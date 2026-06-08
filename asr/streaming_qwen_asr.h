#pragma once

#include "asr_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vox::asr {

struct StreamingQwenAsrConfig {
    std::string model_path;
    std::string mmproj_path;
    std::string language = "auto";
    std::string context;

    int32_t threads = 4;
    int32_t step_ms = 2000;
    int32_t window_ms = 6000;
    int32_t overlap_ms = 300;

    std::string vad_model_path;
    float vad_threshold = 0.5f;
    int32_t vad_min_silence_ms = 600;
    int32_t vad_speech_pad_ms = 200;
    int32_t vad_max_speech_ms = 30000;

    int32_t context_size = 10240;
    int32_t batch_size = 512;
    int32_t max_output_tokens = 512;
    int32_t gpu_layers = 999;

    float min_audio_rms = 0.001f;
    float temperature = 0.0f;
    float top_p = 0.95f;
    int32_t top_k = 64;

    bool debug = false;
    bool use_gpu = true;
    bool mmproj_use_gpu = true;
    bool flash_attention = true;
    bool use_mmap = true;
    bool use_mlock = false;
    bool use_vad = true;
    bool utterance_mode = true;
};

class StreamingQwenAsr {
public:
    explicit StreamingQwenAsr(StreamingQwenAsrConfig config);
    ~StreamingQwenAsr();

    StreamingQwenAsr(const StreamingQwenAsr &) = delete;
    StreamingQwenAsr & operator=(const StreamingQwenAsr &) = delete;

    // Input must be mono float32 PCM at kAsrSampleRate.
    std::vector<Transcript> push_audio(const float * samples, size_t sample_count);
    std::vector<Transcript> push_audio(const std::vector<float> & samples);

    std::vector<Transcript> flush();
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vox::asr
