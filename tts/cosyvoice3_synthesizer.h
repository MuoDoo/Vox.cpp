#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vox::tts {

inline constexpr int32_t kCosyVoice3SampleRate = 24000;

struct CosyVoice3TtsConfig {
    std::string model_path;
    std::string flow_model_path;
    std::string hift_model_path;
    std::string voices_model_path;
    std::string voice = "zero_shot";
    std::string output_dir = "tts-output";
    std::string play_command;

    int32_t threads = 4;
    int32_t max_tokens = 0;
    int32_t flow_steps = 0; // 0 = model default (10); fewer = faster, lower quality
    float temperature = 0.8f;
    uint64_t seed = 42;

    bool use_gpu = false;
    bool flash_attention = true;
    bool play_after_synthesis = false;
};

class CosyVoice3Synthesizer {
public:
    explicit CosyVoice3Synthesizer(CosyVoice3TtsConfig config);
    ~CosyVoice3Synthesizer();

    CosyVoice3Synthesizer(const CosyVoice3Synthesizer &) = delete;
    CosyVoice3Synthesizer & operator=(const CosyVoice3Synthesizer &) = delete;

    std::string synthesize(const std::string & text, uint64_t chunk_index, bool is_final);
    std::vector<float> synthesize_pcm(const std::string & text);

private:
    class Impl;
    Impl * impl_;
};

std::string default_tts_play_command();
void play_audio_file(const std::string & command, const std::string & path);
void write_wav_mono_16(const std::string & path, const float * samples, int32_t sample_count, int32_t sample_rate);

} // namespace vox::tts
