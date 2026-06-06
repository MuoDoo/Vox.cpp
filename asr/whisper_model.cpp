#include "whisper_model.h"

#include "ggml-backend.h"
#include "whisper.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vox::asr {
namespace {

struct SegmentResult {
    std::string text;
    float no_speech_probability = 0.0f;
    float average_token_probability = 0.0f;
    float minimum_token_probability = 0.0f;
    int token_count = 0;
    bool accepted = false;
};

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

SegmentResult collect_segment(whisper_context * ctx, int segment_index, float min_token_probability) {
    SegmentResult result;
    result.no_speech_probability = whisper_full_get_segment_no_speech_prob(ctx, segment_index);

    const char * text = whisper_full_get_segment_text(ctx, segment_index);
    if (text != nullptr) {
        result.text = trim(text);
    }

    const int token_count = whisper_full_n_tokens(ctx, segment_index);
    float min_probability = 1.0f;
    double sum_probability = 0.0;
    int probability_count = 0;
    for (int i = 0; i < token_count; ++i) {
        const float probability = whisper_full_get_token_p(ctx, segment_index, i);
        if (probability < 0.0f) {
            continue;
        }
        min_probability = std::min(min_probability, probability);
        sum_probability += probability;
        ++probability_count;
    }

    result.token_count = probability_count;
    if (probability_count > 0) {
        result.average_token_probability =
            static_cast<float>(sum_probability / static_cast<double>(probability_count));
        result.minimum_token_probability = min_probability;
    }

    result.accepted =
        !result.text.empty() &&
        (probability_count == 0 || result.average_token_probability >= min_token_probability);
    return result;
}

std::string collect_text(whisper_context * ctx, float min_token_probability, bool debug) {
    std::ostringstream output;
    const int segment_count = whisper_full_n_segments(ctx);

    for (int i = 0; i < segment_count; ++i) {
        const SegmentResult segment = collect_segment(ctx, i, min_token_probability);
        if (debug) {
            std::cerr << " segment[" << i << "].no_speech=" << segment.no_speech_probability
                      << " avg_p=" << segment.average_token_probability
                      << " min_p=" << segment.minimum_token_probability
                      << " tokens=" << segment.token_count
                      << " accepted=" << segment.accepted;
        }
        if (segment.accepted) {
            output << segment.text;
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
        params.no_timestamps = config_.no_timestamps;
        params.debug_mode = config_.debug;
        params.translate = false;
        params.single_segment = true;
        params.max_tokens = 0;
        params.language = config_.language.c_str();
        params.n_threads = config_.threads;
        params.audio_ctx = 0;
        params.no_speech_thold = config_.no_speech_threshold;

        const int result = whisper_full(
            ctx_,
            params,
            pcm.data(),
            static_cast<int>(pcm.size()));
        if (result != 0) {
            throw std::runtime_error("whisper inference failed");
        }

        if (config_.debug) {
            const int segment_count = whisper_full_n_segments(ctx_);
            std::cerr << "whisper: samples=" << pcm.size()
                      << " segments=" << segment_count;
            collect_text(ctx_, config_.min_token_probability, true);
            std::cerr << "\n";
        }

        return collect_text(ctx_, config_.min_token_probability, false);
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
        config_.no_speech_threshold = std::max(0.0f, config_.no_speech_threshold);
        config_.min_token_probability = std::max(0.0f, config_.min_token_probability);
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
