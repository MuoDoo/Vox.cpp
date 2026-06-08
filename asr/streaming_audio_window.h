#pragma once

#include "asr_types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vox::asr {

struct StreamingAudioWindowConfig {
    int32_t sample_rate = kAsrSampleRate;
    int32_t step_ms = 2000;
    int32_t window_ms = 6000;
    int32_t overlap_ms = 300;
    float min_audio_rms = 0.001f;
};

class StreamingAudioWindow {
public:
    using TranscribeFunction = std::function<std::string(const std::vector<float> &)>;

    StreamingAudioWindow(StreamingAudioWindowConfig config, TranscribeFunction transcribe);

    std::vector<Transcript> push_audio(const float * samples, size_t sample_count);
    std::vector<Transcript> push_audio(const std::vector<float> & samples);

    std::vector<Transcript> flush();
    void reset();

    bool has_audible_signal(const std::vector<float> & samples) const;

private:
    size_t to_samples(int32_t milliseconds) const;
    void append_transcript(
        const std::vector<float> & new_audio,
        bool is_final,
        std::vector<Transcript> & transcripts);
    void normalize_config();

    StreamingAudioWindowConfig config_;
    TranscribeFunction transcribe_;
    std::vector<float> pending_audio_;
    std::vector<float> window_audio_;
    uint64_t chunk_index_ = 0;
    bool has_unflushed_audio_ = false;
};

} // namespace vox::asr
