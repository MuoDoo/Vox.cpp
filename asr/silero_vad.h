#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vox::asr {

struct SileroVadConfig {
    std::string model_path;
    int32_t threads = 4;
    float threshold = 0.5f;
    bool use_gpu = false;
};

struct VadResult {
    bool has_speech = false;
    float max_probability = 0.0f;
    float average_probability = 0.0f;
    size_t probability_count = 0;
};

class SileroVad {
public:
    explicit SileroVad(SileroVadConfig config);
    ~SileroVad();

    SileroVad(const SileroVad &) = delete;
    SileroVad & operator=(const SileroVad &) = delete;

    VadResult analyze(const std::vector<float> & samples);
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vox::asr
