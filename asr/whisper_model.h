#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vox::asr {

struct WhisperModelConfig {
    std::string model_path;
    std::string language = "auto";

    int32_t threads = 4;

    bool use_gpu = true;
    bool flash_attention = true;
};

class WhisperModel {
public:
    explicit WhisperModel(WhisperModelConfig config);
    ~WhisperModel();

    WhisperModel(const WhisperModel &) = delete;
    WhisperModel & operator=(const WhisperModel &) = delete;

    std::string transcribe(const std::vector<float> & pcm);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vox::asr
