#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace vox::app {

class MicrophoneAudioSource {
public:
    explicit MicrophoneAudioSource(int32_t capture_device_id = -1, int32_t buffer_ms = 6000);
    ~MicrophoneAudioSource();

    MicrophoneAudioSource(const MicrophoneAudioSource &) = delete;
    MicrophoneAudioSource & operator=(const MicrophoneAudioSource &) = delete;

    void start();
    std::vector<float> read(int32_t milliseconds);
    std::vector<float> read(int32_t milliseconds, const std::function<bool()> & should_continue);
    bool poll_events();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vox::app
