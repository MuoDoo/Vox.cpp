#include "whisper_realtime.h"

#include "whisper_model.h"

#include "common-sdl.h"
#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <thread>
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

class RealtimeWhisper::Impl {
public:
    explicit Impl(RealtimeWhisperConfig config)
        : config_(std::move(config)) {
        normalize_config();
        initialize_model();
        initialize_audio();
    }

    ~Impl() {
        if (audio_ != nullptr) {
            audio_->pause();
        }
    }

    void run(
        const std::function<bool(const Transcript &)> & on_transcript,
        const std::function<bool()> & keep_running) {
        if (!audio_->resume()) {
            throw std::runtime_error("failed to start audio capture");
        }

        const int samples_per_step = config_.step_ms * WHISPER_SAMPLE_RATE / 1000;
        const int samples_per_window = config_.window_ms * WHISPER_SAMPLE_RATE / 1000;
        const int samples_to_overlap = config_.overlap_ms * WHISPER_SAMPLE_RATE / 1000;

        std::vector<float> new_audio;
        std::vector<float> old_audio;
        std::vector<float> window_audio;

        uint64_t chunk_index = 0;
        while (keep_running()) {
            if (!sdl_poll_events()) {
                break;
            }

            audio_->get(config_.step_ms, new_audio);
            if (static_cast<int>(new_audio.size()) < samples_per_step) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            audio_->clear();

            const int new_samples = static_cast<int>(new_audio.size());
            const int old_samples_to_keep = std::min(
                static_cast<int>(old_audio.size()),
                std::max(0, samples_to_overlap + samples_per_window - new_samples));

            window_audio.resize(old_samples_to_keep + new_samples);
            if (old_samples_to_keep > 0) {
                std::copy(
                    old_audio.end() - old_samples_to_keep,
                    old_audio.end(),
                    window_audio.begin());
            }
            std::copy(new_audio.begin(), new_audio.end(), window_audio.begin() + old_samples_to_keep);
            old_audio = window_audio;

            const std::string text = model_->transcribe(window_audio);
            if (!text.empty()) {
                const Transcript transcript{chunk_index, text};
                if (!on_transcript(transcript)) {
                    break;
                }
            }

            ++chunk_index;
        }
    }

private:
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

    void initialize_audio() {
        audio_ = std::make_unique<audio_async>(config_.window_ms);
        if (!audio_->init(config_.capture_device_id, WHISPER_SAMPLE_RATE)) {
            throw std::runtime_error("failed to initialize audio capture");
        }
    }

    RealtimeWhisperConfig config_;
    std::unique_ptr<WhisperModel> model_;
    std::unique_ptr<audio_async> audio_;
};

RealtimeWhisper::RealtimeWhisper(RealtimeWhisperConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
}

RealtimeWhisper::~RealtimeWhisper() = default;

void RealtimeWhisper::run(
    const std::function<bool(const Transcript &)> & on_transcript,
    const std::function<bool()> & keep_running) {
    impl_->run(on_transcript, keep_running);
}

} // namespace vox::asr
