#include "streaming_whisper.h"

#include "streaming_audio_window.h"
#include "whisper_model.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace vox::asr {
namespace {

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
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
        return window_->push_audio(samples, sample_count);
    }

    std::vector<Transcript> flush() {
        return window_->flush();
    }

    void reset() {
        window_->reset();
    }

private:
    void normalize_config() {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing whisper model path");
        }
        if (!file_exists(config_.model_path)) {
            throw std::runtime_error("whisper model file not found: " + config_.model_path);
        }
        config_.min_audio_rms = std::max(0.0f, config_.min_audio_rms);
        config_.no_speech_threshold = std::max(0.0f, config_.no_speech_threshold);
        config_.min_token_probability = std::max(0.0f, config_.min_token_probability);
    }

    void initialize_model() {
        WhisperModelConfig model_config;
        model_config.model_path = config_.model_path;
        model_config.language = config_.language;
        model_config.threads = config_.threads;
        model_config.no_speech_threshold = config_.no_speech_threshold;
        model_config.min_token_probability = config_.min_token_probability;
        model_config.debug = config_.debug;
        model_config.no_timestamps = config_.no_timestamps;
        model_config.use_gpu = config_.use_gpu;
        model_config.flash_attention = config_.flash_attention;
        model_ = std::make_unique<WhisperModel>(std::move(model_config));

        StreamingAudioWindowConfig window_config;
        window_config.sample_rate = kWhisperSampleRate;
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

    StreamingWhisperConfig config_;
    std::unique_ptr<WhisperModel> model_;
    std::unique_ptr<StreamingAudioWindow> window_;
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
