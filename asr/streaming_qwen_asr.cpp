#include "streaming_qwen_asr.h"

#include "qwen_asr_model.h"
#include "silero_vad.h"
#include "streaming_audio_window.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vox::asr {
namespace {

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

int clamp_positive(int value, int default_value) {
    return value > 0 ? value : default_value;
}

void append_tail(
    std::vector<float> & destination,
    const std::vector<float> & source,
    size_t max_samples) {
    if (source.empty() || max_samples == 0) {
        return;
    }

    const size_t samples_to_copy = std::min(max_samples, source.size());
    destination.insert(
        destination.end(),
        source.end() - static_cast<std::ptrdiff_t>(samples_to_copy),
        source.end());
}

void append_bounded(
    std::vector<float> & destination,
    const std::vector<float> & source,
    size_t max_samples) {
    append_tail(destination, source, max_samples);
    if (destination.size() <= max_samples) {
        return;
    }

    destination.erase(
        destination.begin(),
        destination.end() - static_cast<std::ptrdiff_t>(max_samples));
}

} // namespace

class StreamingQwenAsr::Impl {
public:
    explicit Impl(StreamingQwenAsrConfig config)
        : config_(std::move(config)) {
        normalize_config();
        initialize_model();
        initialize_streaming_mode();
    }

    std::vector<Transcript> push_audio(const float * samples, size_t sample_count) {
        if (vad_) {
            return push_vad_audio(samples, sample_count);
        }
        return window_->push_audio(samples, sample_count);
    }

    std::vector<Transcript> flush() {
        if (!vad_) {
            return window_->flush();
        }

        std::vector<Transcript> transcripts;
        if (!has_unflushed_audio_) {
            return transcripts;
        }

        if (!pending_audio_.empty()) {
            std::vector<float> tail;
            tail.swap(pending_audio_);
            append_utterance_step(tail, transcripts);
        }
        if (in_speech_ && !utterance_audio_.empty()) {
            emit_utterance(transcripts);
        }

        pending_audio_.clear();
        pre_speech_audio_.clear();
        silence_ms_ = 0;
        in_speech_ = false;
        has_unflushed_audio_ = false;
        vad_->reset();
        return transcripts;
    }

    void reset() {
        if (window_) {
            window_->reset();
        }
        pending_audio_.clear();
        utterance_audio_.clear();
        pre_speech_audio_.clear();
        chunk_index_ = 0;
        silence_ms_ = 0;
        in_speech_ = false;
        has_unflushed_audio_ = false;
        if (vad_) {
            vad_->reset();
        }
    }

private:
    size_t to_samples(int32_t milliseconds) const {
        return static_cast<size_t>(milliseconds) * kAsrSampleRate / 1000;
    }

    std::vector<Transcript> push_vad_audio(const float * samples, size_t sample_count) {
        if (sample_count == 0) {
            return {};
        }
        if (samples == nullptr) {
            throw std::runtime_error("audio samples pointer is null");
        }

        pending_audio_.reserve(pending_audio_.size() + sample_count);
        for (size_t i = 0; i < sample_count; ++i) {
            pending_audio_.push_back(std::isfinite(samples[i]) ? samples[i] : 0.0f);
        }
        has_unflushed_audio_ = true;

        std::vector<Transcript> transcripts;
        const size_t samples_per_step = to_samples(config_.step_ms);
        while (pending_audio_.size() >= samples_per_step) {
            std::vector<float> step(
                pending_audio_.begin(),
                pending_audio_.begin() + static_cast<std::ptrdiff_t>(samples_per_step));
            pending_audio_.erase(
                pending_audio_.begin(),
                pending_audio_.begin() + static_cast<std::ptrdiff_t>(samples_per_step));

            append_utterance_step(step, transcripts);
        }
        return transcripts;
    }

    void append_utterance_step(
        const std::vector<float> & step,
        std::vector<Transcript> & transcripts) {
        const VadResult vad_result = vad_->analyze(step);
        if (config_.debug) {
            std::cerr << "qwen-vad: samples=" << step.size()
                      << " speech=" << vad_result.has_speech
                      << " max_p=" << vad_result.max_probability
                      << " avg_p=" << vad_result.average_probability
                      << " in_speech=" << in_speech_ << "\n";
        }

        const size_t pad_samples = to_samples(config_.vad_speech_pad_ms);
        if (vad_result.has_speech) {
            if (!in_speech_) {
                utterance_audio_.clear();
                utterance_audio_.insert(
                    utterance_audio_.end(),
                    pre_speech_audio_.begin(),
                    pre_speech_audio_.end());
                in_speech_ = true;
            }
            utterance_audio_.insert(utterance_audio_.end(), step.begin(), step.end());
            pre_speech_audio_.clear();
            silence_ms_ = 0;
        } else if (in_speech_) {
            append_tail(utterance_audio_, step, pad_samples);
            silence_ms_ += config_.step_ms;
            if (silence_ms_ >= config_.vad_min_silence_ms) {
                emit_utterance(transcripts);
            }
        } else {
            append_bounded(pre_speech_audio_, step, pad_samples);
        }

        if (in_speech_ &&
            config_.vad_max_speech_ms > 0 &&
            utterance_audio_.size() >= to_samples(config_.vad_max_speech_ms)) {
            emit_utterance(transcripts);
        }
    }

    void emit_utterance(std::vector<Transcript> & transcripts) {
        if (!utterance_audio_.empty()) {
            const std::string text = model_->transcribe(utterance_audio_);
            if (!text.empty()) {
                transcripts.push_back(Transcript{chunk_index_, text, true});
            }
            ++chunk_index_;
        }

        utterance_audio_.clear();
        pre_speech_audio_.clear();
        silence_ms_ = 0;
        in_speech_ = false;
        vad_->reset();
    }

    void normalize_config() {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing Qwen3-ASR model path");
        }
        if (config_.mmproj_path.empty()) {
            throw std::runtime_error("missing Qwen3-ASR mmproj path");
        }
        if (!file_exists(config_.model_path)) {
            throw std::runtime_error("Qwen3-ASR model file not found: " + config_.model_path);
        }
        if (!file_exists(config_.mmproj_path)) {
            throw std::runtime_error("Qwen3-ASR mmproj file not found: " + config_.mmproj_path);
        }
        if (!config_.vad_model_path.empty() && !file_exists(config_.vad_model_path)) {
            throw std::runtime_error("Silero VAD model file not found: " + config_.vad_model_path);
        }
        config_.threads = clamp_positive(config_.threads, 4);
        config_.step_ms = std::max(250, clamp_positive(config_.step_ms, 2000));
        config_.window_ms = std::max(config_.step_ms, clamp_positive(config_.window_ms, 6000));
        config_.overlap_ms = std::max(0, std::min(config_.overlap_ms, config_.step_ms));
        config_.context_size = std::max(512, config_.context_size);
        config_.batch_size = std::max(1, config_.batch_size);
        config_.max_output_tokens = std::max(1, config_.max_output_tokens);
        config_.min_audio_rms = std::max(0.0f, config_.min_audio_rms);
        config_.temperature = std::max(0.0f, config_.temperature);
        config_.top_p = std::max(0.0f, std::min(1.0f, config_.top_p));
        config_.top_k = std::max(1, config_.top_k);
        config_.vad_threshold = std::max(0.0f, std::min(1.0f, config_.vad_threshold));
        config_.vad_min_silence_ms =
            std::max(config_.step_ms, clamp_positive(config_.vad_min_silence_ms, 600));
        config_.vad_speech_pad_ms = std::max(0, config_.vad_speech_pad_ms);
        config_.vad_max_speech_ms = std::max(0, config_.vad_max_speech_ms);
    }

    void initialize_model() {
        QwenAsrModelConfig model_config;
        model_config.model_path = config_.model_path;
        model_config.mmproj_path = config_.mmproj_path;
        model_config.language = config_.language;
        model_config.context = config_.context;
        model_config.threads = config_.threads;
        model_config.context_size = config_.context_size;
        model_config.batch_size = config_.batch_size;
        model_config.max_output_tokens = config_.max_output_tokens;
        model_config.gpu_layers = config_.gpu_layers;
        model_config.top_k = config_.top_k;
        model_config.top_p = config_.top_p;
        model_config.temperature = config_.temperature;
        model_config.debug = config_.debug;
        model_config.use_gpu = config_.use_gpu;
        model_config.mmproj_use_gpu = config_.mmproj_use_gpu;
        model_config.flash_attention = config_.flash_attention;
        model_config.use_mmap = config_.use_mmap;
        model_config.use_mlock = config_.use_mlock;
        model_ = std::make_unique<QwenAsrModel>(std::move(model_config));
    }

    void initialize_streaming_mode() {
        if (config_.utterance_mode && config_.use_vad && !config_.vad_model_path.empty()) {
            SileroVadConfig vad_config;
            vad_config.model_path = config_.vad_model_path;
            vad_config.threads = config_.threads;
            vad_config.threshold = config_.vad_threshold;
            vad_config.use_gpu = false;
            vad_ = std::make_unique<SileroVad>(std::move(vad_config));
            return;
        }

        StreamingAudioWindowConfig window_config;
        window_config.sample_rate = kAsrSampleRate;
        window_config.step_ms = config_.step_ms;
        window_config.window_ms = config_.window_ms;
        window_config.overlap_ms = config_.overlap_ms;
        window_config.min_audio_rms = config_.min_audio_rms;
        window_ = std::make_unique<StreamingAudioWindow>(
            window_config,
            [this](const std::vector<float> & pcm) {
                return model_->transcribe(pcm);
            });
    }

    StreamingQwenAsrConfig config_;
    std::unique_ptr<QwenAsrModel> model_;
    std::unique_ptr<StreamingAudioWindow> window_;
    std::unique_ptr<SileroVad> vad_;
    std::vector<float> pending_audio_;
    std::vector<float> utterance_audio_;
    std::vector<float> pre_speech_audio_;
    uint64_t chunk_index_ = 0;
    int32_t silence_ms_ = 0;
    bool in_speech_ = false;
    bool has_unflushed_audio_ = false;
};

StreamingQwenAsr::StreamingQwenAsr(StreamingQwenAsrConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
}

StreamingQwenAsr::~StreamingQwenAsr() = default;

std::vector<Transcript> StreamingQwenAsr::push_audio(const float * samples, size_t sample_count) {
    return impl_->push_audio(samples, sample_count);
}

std::vector<Transcript> StreamingQwenAsr::push_audio(const std::vector<float> & samples) {
    return push_audio(samples.data(), samples.size());
}

std::vector<Transcript> StreamingQwenAsr::flush() {
    return impl_->flush();
}

void StreamingQwenAsr::reset() {
    impl_->reset();
}

} // namespace vox::asr
