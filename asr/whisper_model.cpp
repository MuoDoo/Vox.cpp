#include "whisper_model.h"

#include "ggml-backend.h"
#include "whisper.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vox::asr {
namespace {

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }

    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string collect_text(whisper_context * ctx) {
    std::ostringstream output;
    const int segment_count = whisper_full_n_segments(ctx);

    for (int i = 0; i < segment_count; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        if (text != nullptr) {
            output << text;
        }
    }

    return trim(output.str());
}

int clamp_positive(int value, int default_value) {
    return value > 0 ? value : default_value;
}

} // namespace

class WhisperModel::Impl {
public:
    explicit Impl(WhisperModelConfig config)
        : config_(std::move(config)) {
        normalize_config();
        initialize();
    }

    ~Impl() {
        if (ctx_ != nullptr) {
            whisper_free(ctx_);
        }
    }

    std::string transcribe(const std::vector<float> & pcm) {
        if (pcm.empty()) {
            return "";
        }

        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.print_progress = false;
        params.print_realtime = false;
        params.print_special = false;
        params.print_timestamps = false;
        params.translate = false;
        params.single_segment = true;
        params.max_tokens = 0;
        params.language = config_.language.c_str();
        params.n_threads = config_.threads;
        params.audio_ctx = 0;

        const int result = whisper_full(
            ctx_,
            params,
            pcm.data(),
            static_cast<int>(pcm.size()));
        if (result != 0) {
            throw std::runtime_error("whisper inference failed");
        }

        return collect_text(ctx_);
    }

private:
    void normalize_config() {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing whisper model path");
        }
        if (!file_exists(config_.model_path)) {
            throw std::runtime_error("whisper model file not found: " + config_.model_path);
        }
        if (config_.language != "auto" && whisper_lang_id(config_.language.c_str()) == -1) {
            throw std::runtime_error("unknown whisper language: " + config_.language);
        }

        config_.threads = clamp_positive(config_.threads, 4);
    }

    void initialize() {
        ggml_backend_load_all();

        whisper_context_params context_params = whisper_context_default_params();
        context_params.use_gpu = config_.use_gpu;
        context_params.flash_attn = config_.flash_attention;

        ctx_ = whisper_init_from_file_with_params(config_.model_path.c_str(), context_params);
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize whisper context");
        }

        if (!whisper_is_multilingual(ctx_) && config_.language != "en") {
            config_.language = "en";
        }
    }

    WhisperModelConfig config_;
    whisper_context * ctx_ = nullptr;
};

WhisperModel::WhisperModel(WhisperModelConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
}

WhisperModel::~WhisperModel() = default;

std::string WhisperModel::transcribe(const std::vector<float> & pcm) {
    return impl_->transcribe(pcm);
}

} // namespace vox::asr
