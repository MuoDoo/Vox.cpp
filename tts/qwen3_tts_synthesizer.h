#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vox::tts {

inline constexpr int32_t kQwen3TtsSampleRate = 24000;

struct Qwen3TtsConfig {
    std::string model_path;
    std::string codec_model_path;
    std::string voice_model_path;
    std::string ref_text;
    std::string instruct;
    std::string language = "auto";
    std::string voice = "aiden";
    std::string output_dir = "tts-output";
    std::string play_command;

    int32_t threads = 4;
    int32_t max_codec_steps = 0;
    float temperature = 0.8f;
    uint64_t seed = 42;

    bool use_gpu = false;
    bool flash_attention = true;
    bool play_after_synthesis = false;
};

class Qwen3TtsSynthesizer {
public:
    explicit Qwen3TtsSynthesizer(Qwen3TtsConfig config);
    ~Qwen3TtsSynthesizer();

    Qwen3TtsSynthesizer(const Qwen3TtsSynthesizer &) = delete;
    Qwen3TtsSynthesizer & operator=(const Qwen3TtsSynthesizer &) = delete;

    std::string synthesize(const std::string & text, uint64_t chunk_index, bool is_final);
    std::vector<float> synthesize_pcm(const std::string & text);

private:
    class Impl;
    Impl * impl_;
};

} // namespace vox::tts
