#include "streaming_whisper.h"

#include "whisper_model.h"

#include <algorithm>
#include <fstream>
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

} // namespace

class StreamingWhisper::Impl {
public:
    explicit Impl(StreamingWhisperConfig config)
        : config_(std::move(config)) {
        normalize_config();
        initialize_model();
    }

    std::vector<Transcript> push_audio(const float * samples, size_t sample_count) {
        if (sample_count == 0) {
            return {};
        }
        if (samples == nullptr) {
            throw std::runtime_error("audio samples pointer is null");
        }

        pending_audio_.insert(pending_audio_.end(), samples, samples + sample_count);
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

            append_transcript(step, false, transcripts);
        }
        return transcripts;
    }

    std::vector<Transcript> flush() {
        std::vector<Transcript> transcripts;
        if (!has_unflushed_audio_) {
            return transcripts;
        }

        if (!pending_audio_.empty()) {
            std::vector<float> tail;
            tail.swap(pending_audio_);
            append_transcript(tail, true, transcripts);
        } else if (!window_audio_.empty()) {
            const std::string text = model_->transcribe(window_audio_);
            if (!text.empty()) {
                transcripts.push_back(Transcript{chunk_index_, text, true});
            }
            ++chunk_index_;
        }
        has_unflushed_audio_ = false;
        return transcripts;
    }

    void reset() {
        pending_audio_.clear();
        window_audio_.clear();
        chunk_index_ = 0;
        has_unflushed_audio_ = false;
    }

private:
    size_t to_samples(int32_t milliseconds) const {
        return static_cast<size_t>(milliseconds) * kWhisperSampleRate / 1000;
    }

    void append_transcript(
        const std::vector<float> & new_audio,
        bool is_final,
        std::vector<Transcript> & transcripts) {
        const size_t samples_per_window = to_samples(config_.window_ms);
        const size_t samples_to_overlap = to_samples(config_.overlap_ms);
        const size_t old_samples_to_keep = std::min(
            window_audio_.size(),
            samples_to_overlap + samples_per_window > new_audio.size()
                ? samples_to_overlap + samples_per_window - new_audio.size()
                : size_t{0});

        std::vector<float> next_window;
        next_window.reserve(old_samples_to_keep + new_audio.size());
        next_window.insert(
            next_window.end(),
            window_audio_.end() - static_cast<std::ptrdiff_t>(old_samples_to_keep),
            window_audio_.end());
        next_window.insert(next_window.end(), new_audio.begin(), new_audio.end());
        window_audio_.swap(next_window);

        const std::string text = model_->transcribe(window_audio_);
        if (!text.empty()) {
            transcripts.push_back(Transcript{chunk_index_, text, is_final});
        }
        ++chunk_index_;
    }

    void normalize_config() {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing whisper model path");
        }
        if (!file_exists(config_.model_path)) {
            throw std::runtime_error("whisper model file not found: " + config_.model_path);
        }
        config_.threads = clamp_positive(config_.threads, 4);
        config_.step_ms = std::max(250, clamp_positive(config_.step_ms, 2000));
        config_.window_ms = std::max(config_.step_ms, clamp_positive(config_.window_ms, 6000));
        config_.overlap_ms = std::max(0, std::min(config_.overlap_ms, config_.step_ms));
    }

    void initialize_model() {
        WhisperModelConfig model_config;
        model_config.model_path = config_.model_path;
        model_config.language = config_.language;
        model_config.threads = config_.threads;
        model_config.use_gpu = config_.use_gpu;
        model_config.flash_attention = config_.flash_attention;
        model_ = std::make_unique<WhisperModel>(std::move(model_config));
    }

    StreamingWhisperConfig config_;
    std::unique_ptr<WhisperModel> model_;
    std::vector<float> pending_audio_;
    std::vector<float> window_audio_;
    uint64_t chunk_index_ = 0;
    bool has_unflushed_audio_ = false;
};

StreamingWhisper::StreamingWhisper(StreamingWhisperConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
}

StreamingWhisper::~StreamingWhisper() = default;

std::vector<Transcript> StreamingWhisper::push_audio(const float * samples, size_t sample_count) {
    return impl_->push_audio(samples, sample_count);
}

std::vector<Transcript> StreamingWhisper::push_audio(const std::vector<float> & samples) {
    return push_audio(samples.data(), samples.size());
}

std::vector<Transcript> StreamingWhisper::flush() {
    return impl_->flush();
}

void StreamingWhisper::reset() {
    impl_->reset();
}

} // namespace vox::asr
