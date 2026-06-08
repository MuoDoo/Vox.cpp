#include "silero_vad.h"

#include "whisper.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace vox::asr {
namespace {

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

int32_t clamp_positive(int32_t value, int32_t default_value) {
    return value > 0 ? value : default_value;
}

void quiet_whisper_log(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR && text != nullptr) {
        fputs(text, stderr);
    }
}

struct VadContextDeleter {
    void operator()(whisper_vad_context * context) const {
        whisper_vad_free(context);
    }
};

using VadContextPtr = std::unique_ptr<whisper_vad_context, VadContextDeleter>;

} // namespace

class SileroVad::Impl {
public:
    explicit Impl(SileroVadConfig config)
        : config_(std::move(config)) {
        normalize_config();
        initialize();
    }

    VadResult analyze(const std::vector<float> & samples) {
        if (samples.empty()) {
            return {};
        }

        const bool ok = whisper_vad_detect_speech_no_reset(
            context_.get(),
            samples.data(),
            static_cast<int>(samples.size()));
        if (!ok) {
            throw std::runtime_error("Silero VAD inference failed");
        }

        const int probability_count = whisper_vad_n_probs(context_.get());
        const float * probabilities = whisper_vad_probs(context_.get());
        if (probability_count <= 0 || probabilities == nullptr) {
            return {};
        }

        VadResult result;
        result.probability_count = static_cast<size_t>(probability_count);
        double probability_sum = 0.0;
        for (int i = 0; i < probability_count; ++i) {
            const float probability = probabilities[i];
            result.max_probability = std::max(result.max_probability, probability);
            probability_sum += static_cast<double>(probability);
        }
        result.average_probability =
            static_cast<float>(probability_sum / static_cast<double>(probability_count));
        result.has_speech = result.max_probability >= config_.threshold;
        return result;
    }

    void reset() {
        whisper_vad_reset_state(context_.get());
    }

private:
    void normalize_config() {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing Silero VAD model path");
        }
        if (!file_exists(config_.model_path)) {
            throw std::runtime_error("Silero VAD model file not found: " + config_.model_path);
        }
        config_.threads = clamp_positive(config_.threads, 4);
        config_.threshold = std::max(0.0f, std::min(1.0f, config_.threshold));
    }

    void initialize() {
        whisper_log_set(quiet_whisper_log, nullptr);

        whisper_vad_context_params params = whisper_vad_default_context_params();
        params.n_threads = config_.threads;
        params.use_gpu = config_.use_gpu;

        context_.reset(whisper_vad_init_from_file_with_params(config_.model_path.c_str(), params));
        if (!context_) {
            throw std::runtime_error("failed to initialize Silero VAD: " + config_.model_path);
        }
    }

    SileroVadConfig config_;
    VadContextPtr context_;
};

SileroVad::SileroVad(SileroVadConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
}

SileroVad::~SileroVad() = default;

VadResult SileroVad::analyze(const std::vector<float> & samples) {
    return impl_->analyze(samples);
}

void SileroVad::reset() {
    impl_->reset();
}

} // namespace vox::asr
