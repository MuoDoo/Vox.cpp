#include "qwen_asr_model.h"

#include "asr_types.h"
#include "llama_runtime.h"

#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vox::asr {
namespace {

void quiet_mtmd_log(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR && text != nullptr) {
        fputs(text, stderr);
    }
}

struct ModelDeleter {
    void operator()(llama_model * model) const {
        llama_model_free(model);
    }
};

struct ContextDeleter {
    void operator()(llama_context * context) const {
        llama_free(context);
    }
};

struct SamplerDeleter {
    void operator()(llama_sampler * sampler) const {
        llama_sampler_free(sampler);
    }
};

struct MtmdContextDeleter {
    void operator()(mtmd_context * context) const {
        mtmd_free(context);
    }
};

struct MtmdBitmapDeleter {
    void operator()(mtmd_bitmap * bitmap) const {
        mtmd_bitmap_free(bitmap);
    }
};

struct MtmdInputChunksDeleter {
    void operator()(mtmd_input_chunks * chunks) const {
        mtmd_input_chunks_free(chunks);
    }
};

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;
using MtmdContextPtr = std::unique_ptr<mtmd_context, MtmdContextDeleter>;
using MtmdBitmapPtr = std::unique_ptr<mtmd_bitmap, MtmdBitmapDeleter>;
using MtmdInputChunksPtr = std::unique_ptr<mtmd_input_chunks, MtmdInputChunksDeleter>;

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

int clamp_positive(int value, int default_value) {
    return value > 0 ? value : default_value;
}

std::string trim_ascii(std::string value) {
    const auto is_space = [](unsigned char c) {
        return std::isspace(c) != 0;
    };

    const auto begin = std::find_if_not(value.begin(), value.end(), is_space);
    const auto end = std::find_if_not(value.rbegin(), value.rend(), is_space).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool starts_with(const std::string & value, const std::string & prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string strip_token(std::string text, const std::string & token) {
    size_t position = text.find(token);
    while (position != std::string::npos) {
        text.erase(position, token.size());
        position = text.find(token, position);
    }
    return text;
}

std::string strip_qwen_chat_tokens(std::string text) {
    text = strip_token(std::move(text), "<|im_end|>");

    const std::string assistant_prefix = "<|im_start|>assistant";
    std::string lowered = lower_ascii(text);
    size_t position = lowered.find(assistant_prefix);
    while (position != std::string::npos) {
        text.erase(position, assistant_prefix.size());
        lowered.erase(position, assistant_prefix.size());
        position = lowered.find(assistant_prefix, position);
    }

    return trim_ascii(std::move(text));
}

} // namespace

std::string parse_qwen_asr_output(const std::string & raw, const std::string & forced_language) {
    std::string text = trim_ascii(raw);
    if (text.empty()) {
        return {};
    }

    text = strip_qwen_chat_tokens(std::move(text));
    if (text.empty()) {
        return {};
    }

    const std::string asr_text_marker = "<asr_text>";
    const size_t marker_position = text.find(asr_text_marker);
    if (marker_position != std::string::npos) {
        const std::string header = lower_ascii(trim_ascii(text.substr(0, marker_position)));
        if (starts_with(header, "language none")) {
            return {};
        }
        return strip_qwen_chat_tokens(text.substr(marker_position + asr_text_marker.size()));
    }

    if (!forced_language.empty()) {
        return strip_qwen_chat_tokens(std::move(text));
    }

    const std::string lowered = lower_ascii(text);
    if (starts_with(lowered, "language none")) {
        const size_t line_end = text.find('\n');
        return line_end == std::string::npos ? std::string() : trim_ascii(text.substr(line_end + 1));
    }

    if (starts_with(lowered, "language ")) {
        const size_t line_end = text.find('\n');
        if (line_end != std::string::npos) {
            return trim_ascii(text.substr(line_end + 1));
        }
    }

    return text;
}

namespace {

std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    if (tokens.empty()) {
        return {};
    }

    std::string text;
    text.resize(std::max<size_t>(tokens.size() * 8, 32));
    int32_t bytes = llama_detokenize(
        vocab,
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        text.data(),
        static_cast<int32_t>(text.size()),
        false,
        false);
    if (bytes < 0) {
        text.resize(static_cast<size_t>(-bytes));
        bytes = llama_detokenize(
            vocab,
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            text.data(),
            static_cast<int32_t>(text.size()),
            false,
            false);
    }
    if (bytes < 0) {
        throw std::runtime_error("llama.cpp detokenization failed");
    }

    text.resize(static_cast<size_t>(bytes));
    return text;
}

} // namespace

std::string normalize_qwen_asr_language(const std::string & language) {
    const std::string value = trim_ascii(language);
    if (value.empty() || value == "auto") {
        return {};
    }

    const std::string lower = lower_ascii(value);
    if (lower == "zh" || lower == "zho" || lower == "chinese") {
        return "Chinese";
    }
    if (lower == "en" || lower == "eng" || lower == "english") {
        return "English";
    }
    if (lower == "yue" || lower == "cantonese") {
        return "Cantonese";
    }
    if (lower == "ar" || lower == "arabic") {
        return "Arabic";
    }
    if (lower == "de" || lower == "german") {
        return "German";
    }
    if (lower == "fr" || lower == "french") {
        return "French";
    }
    if (lower == "es" || lower == "spanish") {
        return "Spanish";
    }
    if (lower == "pt" || lower == "portuguese") {
        return "Portuguese";
    }
    if (lower == "id" || lower == "indonesian") {
        return "Indonesian";
    }
    if (lower == "it" || lower == "italian") {
        return "Italian";
    }
    if (lower == "ko" || lower == "korean") {
        return "Korean";
    }
    if (lower == "ru" || lower == "russian") {
        return "Russian";
    }
    if (lower == "th" || lower == "thai") {
        return "Thai";
    }
    if (lower == "vi" || lower == "vietnamese") {
        return "Vietnamese";
    }
    if (lower == "ja" || lower == "japanese") {
        return "Japanese";
    }
    if (lower == "tr" || lower == "turkish") {
        return "Turkish";
    }
    if (lower == "hi" || lower == "hindi") {
        return "Hindi";
    }
    if (lower == "ms" || lower == "malay") {
        return "Malay";
    }
    if (lower == "nl" || lower == "dutch") {
        return "Dutch";
    }
    if (lower == "sv" || lower == "swedish") {
        return "Swedish";
    }
    if (lower == "da" || lower == "danish") {
        return "Danish";
    }
    if (lower == "fi" || lower == "finnish") {
        return "Finnish";
    }
    if (lower == "pl" || lower == "polish") {
        return "Polish";
    }
    if (lower == "cs" || lower == "czech") {
        return "Czech";
    }
    if (lower == "fil" || lower == "filipino") {
        return "Filipino";
    }
    if (lower == "fa" || lower == "persian") {
        return "Persian";
    }
    if (lower == "el" || lower == "greek") {
        return "Greek";
    }
    if (lower == "ro" || lower == "romanian") {
        return "Romanian";
    }
    if (lower == "hu" || lower == "hungarian") {
        return "Hungarian";
    }
    if (lower == "mk" || lower == "macedonian") {
        return "Macedonian";
    }

    std::string normalized = lower;
    normalized[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(normalized[0])));
    return normalized;
}

class QwenAsrModel::Impl {
public:
    explicit Impl(QwenAsrModelConfig config)
        : config_(std::move(config)) {
        normalize_config();
        initialize();
    }

    std::string transcribe(const std::vector<float> & pcm) {
        if (pcm.empty()) {
            return {};
        }

        MtmdBitmapPtr bitmap(mtmd_bitmap_init_from_audio(pcm.size(), pcm.data()));
        if (!bitmap) {
            throw std::runtime_error("failed to create Qwen3-ASR audio bitmap");
        }

        const mtmd_bitmap * bitmaps[] = {bitmap.get()};
        MtmdInputChunksPtr chunks(mtmd_input_chunks_init());
        if (!chunks) {
            throw std::runtime_error("failed to create Qwen3-ASR input chunks");
        }

        const std::string prompt = format_prompt();
        mtmd_input_text input_text;
        input_text.text = prompt.c_str();
        input_text.add_special = true;
        input_text.parse_special = true;

        const int32_t tokenize_result = mtmd_tokenize(
            mtmd_context_.get(),
            chunks.get(),
            &input_text,
            bitmaps,
            1);
        if (tokenize_result != 0) {
            throw std::runtime_error("Qwen3-ASR prompt/audio tokenization failed: " +
                                     std::to_string(tokenize_result));
        }

        const int required_context =
            static_cast<int>(mtmd_helper_get_n_pos(chunks.get())) + config_.max_output_tokens;
        if (required_context > static_cast<int>(llama_n_ctx(context_.get()))) {
            throw std::runtime_error("Qwen3-ASR prompt/audio exceeds llama.cpp context window");
        }

        llama_memory_clear(llama_get_memory(context_.get()), true);
        llama_sampler_reset(sampler_.get());

        llama_pos n_past = 0;
        const int32_t eval_result = mtmd_helper_eval_chunks(
            mtmd_context_.get(),
            context_.get(),
            chunks.get(),
            n_past,
            0,
            config_.batch_size,
            true,
            &n_past);
        if (eval_result != 0) {
            throw std::runtime_error("Qwen3-ASR prompt/audio evaluation failed: " +
                                     std::to_string(eval_result));
        }

        std::vector<llama_token> output_tokens;
        output_tokens.reserve(static_cast<size_t>(config_.max_output_tokens));

        llama_batch batch = llama_batch_init(1, 0, 1);
        for (int32_t i = 0; i < config_.max_output_tokens; ++i) {
            const llama_token token = llama_sampler_sample(sampler_.get(), context_.get(), -1);
            llama_sampler_accept(sampler_.get(), token);
            if (llama_vocab_is_eog(vocab_, token)) {
                break;
            }

            output_tokens.push_back(token);
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = n_past++;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            const int32_t decode_result = llama_decode(context_.get(), batch);
            if (decode_result != 0) {
                llama_batch_free(batch);
                throw std::runtime_error("Qwen3-ASR decode failed: " + std::to_string(decode_result));
            }
        }
        llama_batch_free(batch);

        return parse_qwen_asr_output(detokenize(vocab_, output_tokens), forced_language_);
    }

private:
    void normalize_config() {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing Qwen3-ASR model path");
        }
        if (config_.mmproj_path.empty()) {
            throw std::runtime_error("missing Qwen3-ASR mmproj path");
        }
        if (!file_exists(config_.model_path)) {
            throw std::runtime_error("Qwen3-ASR model file not found: " + config_.model_path);
        }
        if (!file_exists(config_.mmproj_path)) {
            throw std::runtime_error("Qwen3-ASR mmproj file not found: " + config_.mmproj_path);
        }
        config_.threads = clamp_positive(config_.threads, 4);
        config_.context_size = std::max(512, config_.context_size);
        config_.batch_size = std::max(1, config_.batch_size);
        config_.max_output_tokens = std::max(1, config_.max_output_tokens);
        config_.top_k = std::max(1, config_.top_k);
        config_.top_p = std::max(0.0f, std::min(1.0f, config_.top_p));
        config_.temperature = std::max(0.0f, config_.temperature);
        forced_language_ = normalize_qwen_asr_language(config_.language);
    }

    void initialize() {
        runtime_ = std::make_unique<vox::llama::LlamaRuntime>();
        mtmd_helper_log_set(quiet_mtmd_log, nullptr);

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config_.use_gpu ? config_.gpu_layers : 0;
        model_params.use_mmap = config_.use_mmap;
        model_params.use_mlock = config_.use_mlock;
        if (!config_.use_gpu || config_.gpu_layers <= 0) {
            devices_[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            devices_[1] = nullptr;
            if (devices_[0] == nullptr) {
                throw std::runtime_error("failed to find llama.cpp CPU backend device");
            }
            model_params.devices = devices_.data();
        }

        model_.reset(llama_model_load_from_file(config_.model_path.c_str(), model_params));
        if (!model_) {
            throw std::runtime_error("failed to load Qwen3-ASR model: " + config_.model_path);
        }

        vocab_ = llama_model_get_vocab(model_.get());
        if (!vocab_) {
            throw std::runtime_error("Qwen3-ASR model has no vocabulary");
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(config_.context_size);
        context_params.n_batch =
            static_cast<uint32_t>(std::min(config_.batch_size, config_.context_size));
        context_params.n_ubatch = context_params.n_batch;
        context_params.n_threads = config_.threads > 0 ? config_.threads : vox::llama::default_thread_count();
        context_params.n_threads_batch = context_params.n_threads;
        context_params.no_perf = true;
        context_params.offload_kqv = config_.use_gpu && config_.gpu_layers > 0;
        context_params.op_offload = config_.use_gpu && config_.gpu_layers > 0;
        context_params.flash_attn_type =
            config_.flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

        context_.reset(llama_init_from_model(model_.get(), context_params));
        if (!context_) {
            throw std::runtime_error("failed to create Qwen3-ASR llama.cpp context");
        }

        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = config_.use_gpu && config_.mmproj_use_gpu;
        mtmd_params.print_timings = config_.debug;
        mtmd_params.n_threads = context_params.n_threads;
        mtmd_params.flash_attn_type = context_params.flash_attn_type;
        mtmd_context_.reset(mtmd_init_from_file(config_.mmproj_path.c_str(), model_.get(), mtmd_params));
        if (!mtmd_context_) {
            throw std::runtime_error("failed to load Qwen3-ASR mmproj: " + config_.mmproj_path);
        }
        if (!mtmd_support_audio(mtmd_context_.get())) {
            throw std::runtime_error("Qwen3-ASR mmproj does not support audio input");
        }
        const int sample_rate = mtmd_get_audio_sample_rate(mtmd_context_.get());
        if (sample_rate != kAsrSampleRate) {
            throw std::runtime_error("Qwen3-ASR expects " + std::to_string(sample_rate) +
                                     " Hz audio, but Vox captures " +
                                     std::to_string(kAsrSampleRate) + " Hz");
        }

        sampler_.reset(make_sampler());
        if (!sampler_) {
            throw std::runtime_error("failed to create Qwen3-ASR sampler");
        }
    }

    llama_sampler * make_sampler() const {
        if (config_.temperature <= 0.0f) {
            return llama_sampler_init_greedy();
        }

        llama_sampler_chain_params params = llama_sampler_chain_default_params();
        params.no_perf = true;

        llama_sampler * sampler = llama_sampler_chain_init(params);
        if (!sampler) {
            return nullptr;
        }

        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(config_.top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config_.top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(config_.temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(config_.seed));
        return sampler;
    }

    std::string format_prompt() const {
        const char * chat_template = llama_model_chat_template(model_.get(), nullptr);
        if (chat_template == nullptr) {
            throw std::runtime_error("Qwen3-ASR model does not provide a chat template");
        }

        const std::string user_content = mtmd_default_marker();
        std::array<llama_chat_message, 2> messages{};
        int32_t message_count = 0;
        if (!config_.context.empty()) {
            messages[message_count++] = llama_chat_message{"system", config_.context.c_str()};
        }
        messages[message_count++] = llama_chat_message{"user", user_content.c_str()};

        int32_t length = llama_chat_apply_template(
            chat_template,
            messages.data(),
            message_count,
            true,
            nullptr,
            0);
        if (length <= 0) {
            throw std::runtime_error("failed to apply Qwen3-ASR chat template");
        }

        std::string formatted(static_cast<size_t>(length), '\0');
        length = llama_chat_apply_template(
            chat_template,
            messages.data(),
            message_count,
            true,
            formatted.data(),
            static_cast<int32_t>(formatted.size()));
        if (length <= 0) {
            throw std::runtime_error("failed to apply Qwen3-ASR chat template");
        }

        formatted.resize(static_cast<size_t>(length));
        if (!forced_language_.empty()) {
            formatted += "language " + forced_language_;
        }
        return formatted;
    }

    QwenAsrModelConfig config_;
    std::string forced_language_;
    std::array<ggml_backend_dev_t, 2> devices_{};
    std::unique_ptr<vox::llama::LlamaRuntime> runtime_;
    ModelPtr model_;
    ContextPtr context_;
    MtmdContextPtr mtmd_context_;
    SamplerPtr sampler_;
    const llama_vocab * vocab_ = nullptr;
};

QwenAsrModel::QwenAsrModel(QwenAsrModelConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
}

QwenAsrModel::~QwenAsrModel() = default;

std::string QwenAsrModel::transcribe(const std::vector<float> & pcm) {
    return impl_->transcribe(pcm);
}

} // namespace vox::asr
