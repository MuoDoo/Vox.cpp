#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vox::asr {

struct QwenAsrModelConfig {
    std::string model_path;
    std::string mmproj_path;
    std::string language = "auto";
    std::string context;

    int32_t threads = 4;
    int32_t context_size = 10240;
    int32_t batch_size = 512;
    int32_t max_output_tokens = 512;
    int32_t gpu_layers = 999;

    int32_t top_k = 64;
    float top_p = 0.95f;
    float temperature = 0.0f;
    uint32_t seed = 1;

    bool debug = false;
    bool use_mmap = true;
    bool use_mlock = false;
    bool use_gpu = true;
    bool mmproj_use_gpu = true;
    bool flash_attention = true;
};

class QwenAsrModel {
public:
    explicit QwenAsrModel(QwenAsrModelConfig config);
    ~QwenAsrModel();

    QwenAsrModel(const QwenAsrModel &) = delete;
    QwenAsrModel & operator=(const QwenAsrModel &) = delete;

    std::string transcribe(const std::vector<float> & pcm);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::string normalize_qwen_asr_language(const std::string & language);
std::string parse_qwen_asr_output(const std::string & raw, const std::string & forced_language);

} // namespace vox::asr
