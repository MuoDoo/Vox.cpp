#include "microphone_audio_source.h"

#include "streaming_whisper.h"

#include "common-sdl.h"

#include <stdexcept>

namespace vox::app {

class MicrophoneAudioSource::Impl {
public:
    Impl(int32_t capture_device_id, int32_t buffer_ms)
        : audio_(buffer_ms) {
        if (!audio_.init(capture_device_id, vox::asr::kWhisperSampleRate)) {
            throw std::runtime_error("failed to initialize audio capture");
        }
    }

    ~Impl() {
        audio_.pause();
    }

    void start() {
        if (!audio_.resume()) {
            throw std::runtime_error("failed to start audio capture");
        }
    }

    std::vector<float> read(int32_t milliseconds) {
        std::vector<float> samples;
        audio_.get(milliseconds, samples);
        audio_.clear();
        return samples;
    }

    bool poll_events() {
        return sdl_poll_events();
    }

private:
    audio_async audio_;
};

MicrophoneAudioSource::MicrophoneAudioSource(int32_t capture_device_id, int32_t buffer_ms)
    : impl_(std::make_unique<Impl>(capture_device_id, buffer_ms)) {
}

MicrophoneAudioSource::~MicrophoneAudioSource() = default;

void MicrophoneAudioSource::start() {
    impl_->start();
}

std::vector<float> MicrophoneAudioSource::read(int32_t milliseconds) {
    return impl_->read(milliseconds);
}

bool MicrophoneAudioSource::poll_events() {
    return impl_->poll_events();
}

} // namespace vox::app
