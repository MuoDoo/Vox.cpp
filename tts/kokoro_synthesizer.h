#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vox::tts {

inline constexpr int32_t kKokoroSampleRate = 24000;

struct KokoroTtsConfig {
    std::string model_path;
    std::string voice_model_path;
    std::string language = "en-us";
    std::string output_dir = "tts-output";
    std::string play_command;

    int32_t threads = 4;
    float length_scale = 1.0f;

    bool use_gpu = false;
    bool flash_attention = true;
    bool play_after_synthesis = false;
    bool phonemes_input = false;
};

class KokoroSynthesizer {
public:
    explicit KokoroSynthesizer(KokoroTtsConfig config);
    ~KokoroSynthesizer();

    KokoroSynthesizer(const KokoroSynthesizer &) = delete;
    KokoroSynthesizer & operator=(const KokoroSynthesizer &) = delete;

    std::string synthesize(const std::string & text, uint64_t chunk_index, bool is_final);
    std::vector<float> synthesize_pcm(const std::string & text);

private:
    class Impl;
    Impl * impl_;
};

} // namespace vox::tts
