#pragma once

#include <cstdint>
#include <string>

namespace vox::asr {

inline constexpr int32_t kAsrSampleRate = 16000;
inline constexpr int32_t kWhisperSampleRate = kAsrSampleRate;

struct Transcript {
    uint64_t chunk_index = 0;
    std::string text;
    bool is_final = false;
};

} // namespace vox::asr
