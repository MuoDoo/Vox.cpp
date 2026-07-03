#include "qwen3_tts_synthesizer.h"

#include "cosyvoice3_synthesizer.h"
#include "qwen3_tts.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vox::tts {
namespace {

struct Qwen3TtsContextDeleter {
    void operator()(qwen3_tts_context * context) const {
        qwen3_tts_free(context);
    }
};

using Qwen3TtsContextPtr = std::unique_ptr<qwen3_tts_context, Qwen3TtsContextDeleter>;

std::string make_output_path(const std::string & output_dir, uint64_t chunk_index, bool is_final) {
    std::ostringstream filename;
    filename << "chunk-" << std::setw(6) << std::setfill('0') << chunk_index
             << (is_final ? "-final" : "-partial") << ".wav";
    return (std::filesystem::path(output_dir) / filename.str()).string();
}

bool file_exists(const std::string & path) {
    return !path.empty() && std::filesystem::exists(path);
}

std::string parent_dir(const std::string & path) {
    const std::filesystem::path file(path);
    const std::filesystem::path parent = file.parent_path();
    return parent.empty() ? "." : parent.string();
}

std::string first_existing(const std::string & dir, const std::vector<std::string> & filenames) {
    for (const std::string & filename : filenames) {
        const std::string path = (std::filesystem::path(dir) / filename).string();
        if (file_exists(path)) {
            return path;
        }
    }
    return {};
}

void require_file(const std::string & path, const std::string & label) {
    if (!file_exists(path)) {
        throw std::runtime_error("missing " + label + ": " + path);
    }
}

std::string lowercase_ascii(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    return value;
}

std::string normalize_language(std::string language) {
    language = lowercase_ascii(std::move(language));
    std::replace(language.begin(), language.end(), '_', '-');
    return language;
}

bool ends_with_ci(const std::string & value, const std::string & suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return lowercase_ascii(value.substr(value.size() - suffix.size())) == lowercase_ascii(suffix);
}

bool parse_i32_exact(const std::string & value, int32_t & out) {
    size_t parsed = 0;
    int parsed_value = 0;
    try {
        parsed_value = std::stoi(value, &parsed);
    } catch (const std::exception &) {
        return false;
    }
    if (parsed != value.size()) {
        return false;
    }
    out = static_cast<int32_t>(parsed_value);
    return true;
}

int32_t qwen3_language_id(const std::string & language) {
    const std::string value = normalize_language(language);
    if (value.empty() || value == "auto" || value == "default") {
        return 0;
    }

    int32_t raw_id = 0;
    if (parse_i32_exact(value, raw_id)) {
        return raw_id;
    }

    if (value == "english" || value == "en" || value == "en-us" || value == "en-gb") {
        return 2050;
    }
    if (value == "chinese" || value == "mandarin" || value == "zh" || value == "zh-cn" ||
        value == "cmn") {
        return 2055;
    }
    if (value == "japanese" || value == "ja" || value == "jp") {
        return 2058;
    }

    return 0;
}

bool is_chinese_language(const std::string & language) {
    return qwen3_language_id(language) == 2055;
}

bool is_default_voice_name(const std::string & voice) {
    const std::string value = normalize_language(voice);
    return value.empty() || value == "auto" || value == "default" || value == "zero-shot" ||
           value == "zero_shot";
}

bool is_chinese_dialect_voice(const std::string & voice) {
    const std::string value = normalize_language(voice);
    return value == "dylan" || value == "eric";
}

std::string resolve_codec_path(const Qwen3TtsConfig & config) {
    if (!config.codec_model_path.empty()) {
        return config.codec_model_path;
    }
    return first_existing(
        parent_dir(config.model_path),
        {
            "qwen3-tts-tokenizer-12hz.gguf",
            "qwen3-tts-tokenizer.gguf",
            "qwen3-tts-codec.gguf",
        });
}

std::string speaker_list(qwen3_tts_context * context) {
    std::ostringstream out;
    const int n_speakers = qwen3_tts_n_speakers(context);
    for (int i = 0; i < n_speakers; ++i) {
        const char * name = qwen3_tts_get_speaker_name(context, i);
        if (name != nullptr) {
            out << (i == 0 ? "" : ", ") << name;
        }
    }
    return out.str();
}

} // namespace

class Qwen3TtsSynthesizer::Impl {
public:
    explicit Impl(Qwen3TtsConfig config)
        : config_(std::move(config)) {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing Qwen3-TTS talker model path");
        }
        if (config_.output_dir.empty()) {
            throw std::runtime_error("missing TTS output directory");
        }
        if (config_.threads <= 0) {
            throw std::runtime_error("Qwen3-TTS thread count must be positive");
        }

        config_.language = normalize_language(config_.language);
        config_.codec_model_path = resolve_codec_path(config_);

        require_file(config_.model_path, "Qwen3-TTS talker model");
        require_file(config_.codec_model_path, "Qwen3-TTS tokenizer/codec model");

        qwen3_tts_context_params params = qwen3_tts_context_default_params();
        params.n_threads = config_.threads;
        params.verbosity = 1;
        params.use_gpu = config_.use_gpu;
        params.flash_attn = config_.flash_attention;
        params.temperature = config_.temperature;
        params.seed = config_.seed;
        params.max_codec_steps = config_.max_codec_steps;

        context_.reset(qwen3_tts_init_from_file(config_.model_path.c_str(), params));
        if (!context_) {
            throw std::runtime_error("failed to load Qwen3-TTS talker model: " + config_.model_path);
        }
        if (qwen3_tts_set_codec_path(context_.get(), config_.codec_model_path.c_str()) != 0) {
            throw std::runtime_error("failed to load Qwen3-TTS tokenizer/codec model: " + config_.codec_model_path);
        }

        configure_voice();
    }

    std::string synthesize(const std::string & text, uint64_t chunk_index, bool is_final) {
        const std::vector<float> pcm = synthesize_pcm(text);
        const std::string output_path = make_output_path(config_.output_dir, chunk_index, is_final);
        write_wav_mono_16(output_path, pcm.data(), static_cast<int32_t>(pcm.size()), kQwen3TtsSampleRate);
        if (config_.play_after_synthesis) {
            play_audio_file(
                config_.play_command.empty() ? default_tts_play_command() : config_.play_command,
                output_path);
        }
        return output_path;
    }

    std::vector<float> synthesize_pcm(const std::string & text) {
        if (text.empty()) {
            throw std::runtime_error("cannot synthesize empty text");
        }

        qwen3_tts_set_temperature(context_.get(), config_.temperature);
        qwen3_tts_set_seed(context_.get(), config_.seed);

        int n_samples = 0;
        float * pcm = qwen3_tts_synthesize(context_.get(), text.c_str(), &n_samples);
        if (pcm == nullptr || n_samples <= 0) {
            qwen3_tts_pcm_free(pcm);
            throw std::runtime_error("Qwen3-TTS synthesis failed");
        }

        std::vector<float> output(pcm, pcm + n_samples);
        qwen3_tts_pcm_free(pcm);
        return output;
    }

private:
    void configure_voice() {
        const bool is_custom_voice = qwen3_tts_is_custom_voice(context_.get()) != 0;
        const bool is_voice_design = qwen3_tts_is_voice_design(context_.get()) != 0;
        const int32_t language_id = qwen3_language_id(config_.language);

        if (is_voice_design) {
            if (language_id > 0) {
                qwen3_tts_set_language(context_.get(), language_id);
            }
            if (config_.instruct.empty()) {
                throw std::runtime_error("Qwen3-TTS VoiceDesign requires --tts-instruct");
            }
            if (qwen3_tts_set_instruct(context_.get(), config_.instruct.c_str()) != 0) {
                throw std::runtime_error("failed to set Qwen3-TTS VoiceDesign instruct");
            }
            return;
        }

        if (is_custom_voice) {
            std::string voice = config_.voice;
            if (is_default_voice_name(voice)) {
                voice = is_chinese_language(config_.language) ? "dylan" : "aiden";
            }

            const bool allow_dialect_override =
                is_chinese_language(config_.language) && is_chinese_dialect_voice(voice);
            if (language_id > 0 && !allow_dialect_override) {
                qwen3_tts_set_language(context_.get(), language_id);
            }
            if (qwen3_tts_set_speaker_by_name(context_.get(), voice.c_str()) != 0) {
                std::string message = "failed to select Qwen3-TTS CustomVoice speaker: " + voice;
                const std::string available = speaker_list(context_.get());
                if (!available.empty()) {
                    message += " (available: " + available + ")";
                }
                throw std::runtime_error(message);
            }
            if (!config_.instruct.empty() &&
                qwen3_tts_set_cv_style_instruct(context_.get(), config_.instruct.c_str()) != 0) {
                throw std::runtime_error("failed to set Qwen3-TTS CustomVoice style instruct");
            }
            active_voice_ = std::move(voice);
            return;
        }

        if (language_id > 0) {
            qwen3_tts_set_language(context_.get(), language_id);
        }
        if (config_.voice_model_path.empty()) {
            throw std::runtime_error(
                "Qwen3-TTS base requires --tts-voice-model PATH to a baked voice GGUF or reference WAV; "
                "use qwen3-tts-customvoice for preset speakers");
        }
        require_file(config_.voice_model_path, "Qwen3-TTS voice model/reference");

        if (ends_with_ci(config_.voice_model_path, ".wav")) {
            if (config_.ref_text.empty()) {
                throw std::runtime_error("Qwen3-TTS reference WAV requires --tts-ref-text");
            }
            if (qwen3_tts_set_voice_prompt_with_text(
                    context_.get(),
                    config_.voice_model_path.c_str(),
                    config_.ref_text.c_str()) != 0) {
                throw std::runtime_error("failed to set Qwen3-TTS reference voice: " + config_.voice_model_path);
            }
        } else {
            if (qwen3_tts_load_voice_pack(context_.get(), config_.voice_model_path.c_str()) != 0) {
                throw std::runtime_error("failed to load Qwen3-TTS voice pack: " + config_.voice_model_path);
            }
            if (!is_default_voice_name(config_.voice) &&
                qwen3_tts_select_voice(context_.get(), config_.voice.c_str()) != 0) {
                throw std::runtime_error("failed to select Qwen3-TTS voice-pack voice: " + config_.voice);
            }
        }
    }

    Qwen3TtsConfig config_;
    Qwen3TtsContextPtr context_;
    std::string active_voice_;
};

Qwen3TtsSynthesizer::Qwen3TtsSynthesizer(Qwen3TtsConfig config)
    : impl_(new Impl(std::move(config))) {
}

Qwen3TtsSynthesizer::~Qwen3TtsSynthesizer() {
    delete impl_;
}

std::string Qwen3TtsSynthesizer::synthesize(const std::string & text, uint64_t chunk_index, bool is_final) {
    return impl_->synthesize(text, chunk_index, is_final);
}

std::vector<float> Qwen3TtsSynthesizer::synthesize_pcm(const std::string & text) {
    return impl_->synthesize_pcm(text);
}

} // namespace vox::tts
