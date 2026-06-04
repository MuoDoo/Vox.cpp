#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vox::asr {

inline constexpr int32_t kWhisperSampleRate = 16000;

struct StreamingWhisperConfig {
    std::string model_path;
    std::string language = "auto";

    int32_t threads = 4;
    int32_t step_ms = 2000;
    int32_t window_ms = 6000;
    int32_t overlap_ms = 300;

    bool use_gpu = true;
    bool flash_attention = true;
};

struct Transcript {
    uint64_t chunk_index = 0;
    std::string text;
    bool is_final = false;
};

class StreamingWhisper {
public:
    explicit StreamingWhisper(StreamingWhisperConfig config);
    ~StreamingWhisper();

    StreamingWhisper(const StreamingWhisper &) = delete;
    StreamingWhisper & operator=(const StreamingWhisper &) = delete;

    // Input must be mono float32 PCM at kWhisperSampleRate.
    std::vector<Transcript> push_audio(const float * samples, size_t sample_count);
    std::vector<Transcript> push_audio(const std::vector<float> & samples);

    // Processes any buffered tail as a final chunk.
    std::vector<Transcript> flush();
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vox::asr
