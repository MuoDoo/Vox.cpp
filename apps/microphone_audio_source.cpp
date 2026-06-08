#include "microphone_audio_source.h"

#include "asr_types.h"

#include "common-sdl.h"

#include <chrono>
#include <stdexcept>
#include <thread>

namespace vox::app {

class MicrophoneAudioSource::Impl {
public:
    Impl(int32_t capture_device_id, int32_t buffer_ms)
        : buffer_ms_(buffer_ms),
          audio_(buffer_ms) {
        if (!audio_.init(capture_device_id, vox::asr::kAsrSampleRate)) {
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
        audio_.get(capped_milliseconds(milliseconds), samples);
        audio_.clear();
        return samples;
    }

    std::vector<float> read(int32_t milliseconds, const std::function<bool()> & should_continue) {
        if (!should_continue) {
            return read(milliseconds);
        }

        std::vector<float> samples;
        const int32_t capped_ms = capped_milliseconds(milliseconds);
        const size_t requested_samples =
            capped_ms > 0
                ? static_cast<size_t>(capped_ms) * vox::asr::kAsrSampleRate / 1000
                : size_t{0};

        while (should_continue()) {
            audio_.get(capped_ms, samples);
            if (requested_samples == 0 || samples.size() >= requested_samples) {
                audio_.clear();
                return samples;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return {};
    }

    bool poll_events() {
        return sdl_poll_events();
    }

private:
    int32_t capped_milliseconds(int32_t milliseconds) const {
        if (milliseconds <= 0 || buffer_ms_ <= 0 || milliseconds <= buffer_ms_) {
            return milliseconds;
        }
        return buffer_ms_;
    }

    int32_t buffer_ms_ = 0;
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

std::vector<float> MicrophoneAudioSource::read(
    int32_t milliseconds,
    const std::function<bool()> & should_continue) {
    return impl_->read(milliseconds, should_continue);
}

bool MicrophoneAudioSource::poll_events() {
    return impl_->poll_events();
}

} // namespace vox::app
