#include "llama_translator.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <clocale>
#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "llama.h"

namespace vox::translate {
namespace {

void quiet_llama_log(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR && text != nullptr) {
        fputs(text, stderr);
    }
}

class LlamaBackend {
public:
    LlamaBackend() {
        std::lock_guard<std::mutex> lock(mutex());
        if (ref_count() == 0) {
            std::setlocale(LC_NUMERIC, "C");
            llama_log_set(quiet_llama_log, nullptr);
            llama_backend_init();
            ggml_backend_load_all();
        }
        ++ref_count();
    }

    ~LlamaBackend() {
        std::lock_guard<std::mutex> lock(mutex());
        --ref_count();
        if (ref_count() == 0) {
            llama_backend_free();
        }
    }

    LlamaBackend(const LlamaBackend &) = delete;
    LlamaBackend & operator=(const LlamaBackend &) = delete;

private:
    static std::mutex & mutex() {
        static std::mutex value;
        return value;
    }

    static int & ref_count() {
        static int value = 0;
        return value;
    }
};

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

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

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

int default_thread_count() {
    const unsigned int count = std::thread::hardware_concurrency();
    return count == 0 ? 4 : static_cast<int>(count);
}

} // namespace

std::string make_hymt_prompt(const std::string & text,
                             const std::string &,
                             const std::string & target_language) {
    return "Translate the following segment into " + target_language +
           ", without additional explanation.\n\n" + text;
}

class LlamaTranslator::Impl {
public:
    explicit Impl(LlamaTranslatorConfig config)
        : config_(std::move(config)) {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing llama.cpp model path");
        }
        if (config_.context_size <= 0) {
            throw std::runtime_error("llama.cpp context size must be positive");
        }
        if (config_.batch_size <= 0) {
            throw std::runtime_error("llama.cpp batch size must be positive");
        }
        if (config_.max_output_tokens <= 0) {
            throw std::runtime_error("llama.cpp max output tokens must be positive");
        }

        backend_ = std::make_unique<LlamaBackend>();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config_.gpu_layers;
        model_params.use_mmap = config_.use_mmap;
        model_params.use_mlock = config_.use_mlock;
        if (config_.gpu_layers <= 0) {
            devices_[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            devices_[1] = nullptr;
            if (devices_[0] == nullptr) {
                throw std::runtime_error("failed to find llama.cpp CPU backend device");
            }
            model_params.devices = devices_.data();
        }

        model_.reset(llama_model_load_from_file(config_.model_path.c_str(), model_params));
        if (!model_) {
            throw std::runtime_error("failed to load llama.cpp model: " + config_.model_path);
        }

        vocab_ = llama_model_get_vocab(model_.get());
        if (!vocab_) {
            throw std::runtime_error("llama.cpp model has no vocabulary");
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(config_.context_size);
        context_params.n_batch = static_cast<uint32_t>(std::min(config_.batch_size, config_.context_size));
        context_params.n_ubatch = context_params.n_batch;
        context_params.n_threads = config_.thread_count > 0 ? config_.thread_count : default_thread_count();
        context_params.n_threads_batch = context_params.n_threads;
        context_params.no_perf = true;
        context_params.offload_kqv = config_.gpu_layers > 0;
        context_params.op_offload = config_.gpu_layers > 0;

        context_.reset(llama_init_from_model(model_.get(), context_params));
        if (!context_) {
            throw std::runtime_error("failed to create llama.cpp context");
        }

        sampler_.reset(make_sampler());
        if (!sampler_) {
            throw std::runtime_error("failed to create llama.cpp sampler");
        }
    }

    std::string translate(const std::string & text) {
        return translate(text, config_.source_language, config_.target_language);
    }

    std::string translate(const std::string & text,
                          const std::string & source_language,
                          const std::string & target_language) {
        if (text.empty()) {
            return {};
        }

        const std::string prompt = format_chat_prompt(make_hymt_prompt(text, source_language, target_language));
        std::vector<llama_token> prompt_tokens = tokenize(prompt, false, true);
        if (prompt_tokens.empty()) {
            throw std::runtime_error("llama.cpp prompt tokenization produced no tokens");
        }

        const int required_context = static_cast<int>(prompt_tokens.size()) + config_.max_output_tokens;
        if (required_context > static_cast<int>(llama_n_ctx(context_.get()))) {
            throw std::runtime_error("llama.cpp prompt exceeds context window");
        }

        llama_memory_clear(llama_get_memory(context_.get()), true);
        llama_sampler_reset(sampler_.get());

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));

        if (llama_model_has_encoder(model_.get())) {
            if (llama_encode(context_.get(), batch) != 0) {
                throw std::runtime_error("llama.cpp encoder evaluation failed");
            }

            llama_token decoder_start = llama_model_decoder_start_token(model_.get());
            if (decoder_start == LLAMA_TOKEN_NULL) {
                decoder_start = llama_vocab_bos(vocab_);
            }
            batch = llama_batch_get_one(&decoder_start, 1);
        }

        std::vector<llama_token> output_tokens;
        output_tokens.reserve(static_cast<size_t>(config_.max_output_tokens));

        int decoded_tokens = 0;
        int position = 0;
        while (decoded_tokens < config_.max_output_tokens) {
            if (position + batch.n_tokens > static_cast<int>(llama_n_ctx(context_.get()))) {
                break;
            }

            const int result = llama_decode(context_.get(), batch);
            if (result != 0) {
                throw std::runtime_error("llama.cpp decode failed: " + std::to_string(result));
            }
            position += batch.n_tokens;

            const llama_token token = llama_sampler_sample(sampler_.get(), context_.get(), -1);
            if (llama_vocab_is_eog(vocab_, token)) {
                break;
            }

            output_tokens.push_back(token);
            ++decoded_tokens;
            batch = llama_batch_get_one(&output_tokens.back(), 1);
        }

        return trim_ascii(detokenize(output_tokens));
    }

private:
    llama_sampler * make_sampler() const {
        llama_sampler_chain_params params = llama_sampler_chain_default_params();
        params.no_perf = true;

        llama_sampler * sampler = llama_sampler_chain_init(params);
        if (!sampler) {
            return nullptr;
        }

        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_penalties(
                config_.penalty_last_n,
                config_.repeat_penalty,
                0.0f,
                0.0f));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(config_.top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config_.top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(config_.temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(config_.seed));

        return sampler;
    }

    std::string format_chat_prompt(const std::string & prompt) const {
        const char * chat_template = llama_model_chat_template(model_.get(), nullptr);
        if (chat_template == nullptr) {
            throw std::runtime_error("llama.cpp model does not provide a chat template");
        }

        const llama_chat_message message{
            "user",
            prompt.c_str(),
        };
        int32_t length = llama_chat_apply_template(
            chat_template,
            &message,
            1,
            true,
            nullptr,
            0);
        if (length <= 0) {
            throw std::runtime_error("failed to apply llama.cpp chat template");
        }

        std::string formatted(static_cast<size_t>(length), '\0');
        length = llama_chat_apply_template(
            chat_template,
            &message,
            1,
            true,
            formatted.data(),
            static_cast<int32_t>(formatted.size()));
        if (length <= 0) {
            throw std::runtime_error("failed to apply llama.cpp chat template");
        }

        formatted.resize(static_cast<size_t>(length));
        return formatted;
    }

    std::vector<llama_token> tokenize(const std::string & text, bool add_special, bool parse_special) const {
        int32_t token_count = llama_tokenize(
            vocab_,
            text.data(),
            static_cast<int32_t>(text.size()),
            nullptr,
            0,
            add_special,
            parse_special);
        if (token_count == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("llama.cpp tokenization overflow");
        }
        if (token_count < 0) {
            token_count = -token_count;
        }

        std::vector<llama_token> tokens(static_cast<size_t>(token_count));
        const int32_t written = llama_tokenize(
            vocab_,
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            token_count,
            add_special,
            parse_special);
        if (written < 0) {
            throw std::runtime_error("llama.cpp tokenization failed");
        }

        tokens.resize(static_cast<size_t>(written));
        return tokens;
    }

    std::string detokenize(const std::vector<llama_token> & tokens) const {
        if (tokens.empty()) {
            return {};
        }

        std::string text;
        text.resize(std::max<size_t>(tokens.size() * 8, 32));
        int32_t bytes = llama_detokenize(
            vocab_,
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            text.data(),
            static_cast<int32_t>(text.size()),
            false,
            false);
        if (bytes < 0) {
            text.resize(static_cast<size_t>(-bytes));
            bytes = llama_detokenize(
                vocab_,
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

    LlamaTranslatorConfig config_;
    std::array<ggml_backend_dev_t, 2> devices_{};
    std::unique_ptr<LlamaBackend> backend_;
    ModelPtr model_;
    ContextPtr context_;
    SamplerPtr sampler_;
    const llama_vocab * vocab_ = nullptr;
};

LlamaTranslator::LlamaTranslator(LlamaTranslatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
}

LlamaTranslator::~LlamaTranslator() = default;

std::string LlamaTranslator::translate(const std::string & text) {
    return impl_->translate(text);
}

std::string LlamaTranslator::translate(const std::string & text,
                                       const std::string & source_language,
                                       const std::string & target_language) {
    return impl_->translate(text, source_language, target_language);
}

} // namespace vox::translate
