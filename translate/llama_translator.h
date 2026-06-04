#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace vox::translate {

struct LlamaTranslatorConfig {
    std::string model_path;
    std::string source_language = "auto";
    std::string target_language = "Chinese";

    int context_size = 2048;
    int batch_size = 512;
    int max_output_tokens = 256;
    int thread_count = 0;
    int gpu_layers = 999;

    int top_k = 20;
    float top_p = 0.6f;
    float temperature = 0.7f;
    float repeat_penalty = 1.05f;
    int penalty_last_n = 64;
    uint32_t seed = 1;

    bool use_mmap = true;
    bool use_mlock = false;
};

class LlamaTranslator {
public:
    explicit LlamaTranslator(LlamaTranslatorConfig config);
    ~LlamaTranslator();

    LlamaTranslator(const LlamaTranslator &) = delete;
    LlamaTranslator & operator=(const LlamaTranslator &) = delete;

    std::string translate(const std::string & text);
    std::string translate(const std::string & text,
                          const std::string & source_language,
                          const std::string & target_language);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::string make_hymt_prompt(const std::string & text,
                             const std::string & source_language,
                             const std::string & target_language);

} // namespace vox::translate
