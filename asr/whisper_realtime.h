#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace vox::asr {

struct RealtimeWhisperConfig {
    std::string model_path;
    std::string language = "auto";

    int32_t threads = 4;
    int32_t capture_device_id = -1;
    int32_t step_ms = 2000;
    int32_t window_ms = 6000;
    int32_t overlap_ms = 300;

    bool use_gpu = true;
    bool flash_attention = true;
};

struct Transcript {
    uint64_t chunk_index = 0;
    std::string text;
};

class RealtimeWhisper {
public:
    explicit RealtimeWhisper(RealtimeWhisperConfig config);
    ~RealtimeWhisper();

    RealtimeWhisper(const RealtimeWhisper &) = delete;
    RealtimeWhisper & operator=(const RealtimeWhisper &) = delete;

    void run(
        const std::function<bool(const Transcript &)> & on_transcript,
        const std::function<bool()> & keep_running);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vox::asr
