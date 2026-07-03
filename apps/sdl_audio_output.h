#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vox::app {

class SdlAudioOutput {
public:
    SdlAudioOutput() = default;
    ~SdlAudioOutput();

    SdlAudioOutput(const SdlAudioOutput &) = delete;
    SdlAudioOutput & operator=(const SdlAudioOutput &) = delete;

    void open(int32_t playback_device_id, int32_t sample_rate, int32_t channels = 1);
    void close();
    void clear();

    void enqueue_mono(const std::vector<float> & samples, int32_t sample_rate);

    int32_t sample_rate() const;
    int32_t channels() const;
    uint32_t queued_bytes() const;
    std::string device_name() const;

private:
    uint32_t device_id_ = 0;
    int32_t sample_rate_ = 0;
    int32_t channels_ = 0;
    std::string device_name_;
};

} // namespace vox::app
