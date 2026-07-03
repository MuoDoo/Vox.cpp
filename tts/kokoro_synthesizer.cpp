#include "kokoro_synthesizer.h"

#include "cosyvoice3_synthesizer.h"
#include "kokoro.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vox::tts {
namespace {

struct KokoroContextDeleter {
    void operator()(kokoro_context * context) const {
        kokoro_free(context);
    }
};

using KokoroContextPtr = std::unique_ptr<kokoro_context, KokoroContextDeleter>;

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

std::string normalize_language(const std::string & language) {
    std::string value = lowercase_ascii(language);
    std::replace(value.begin(), value.end(), '_', '-');

    if (value.empty() || value == "auto" || value == "english" || value == "en") {
        return "en-us";
    }
    if (value == "chinese" || value == "mandarin" || value == "zh" || value == "zh-cn") {
        return "cmn";
    }
    if (value == "japanese") {
        return "ja";
    }
    if (value == "french") {
        return "fr";
    }
    if (value == "german") {
        return "de";
    }
    if (value == "spanish") {
        return "es";
    }
    if (value == "italian") {
        return "it";
    }
    if (value == "portuguese") {
        return "pt";
    }
    if (value == "hindi") {
        return "hi";
    }
    return value;
}

std::string resolve_model_path_for_language(const KokoroTtsConfig & config) {
    std::array<char, 4096> resolved{};
    const int rc = crispasr_kokoro_resolve_model_for_lang(
        config.model_path.c_str(),
        config.language.c_str(),
        resolved.data(),
        static_cast<int>(resolved.size()));
    if (rc < 0) {
        throw std::runtime_error("resolved Kokoro model path is too long");
    }
    return std::string(resolved.data());
}

std::string resolve_voice_path(const KokoroTtsConfig & config, const std::string & model_path) {
    if (!config.voice_model_path.empty()) {
        return config.voice_model_path;
    }

    if (!config.language.empty() && config.language != "auto" &&
        !crispasr_kokoro_lang_has_native_voice(config.language.c_str())) {
        std::array<char, 4096> resolved{};
        std::array<char, 128> picked{};
        const int rc = crispasr_kokoro_resolve_fallback_voice(
            model_path.c_str(),
            config.language.c_str(),
            resolved.data(),
            static_cast<int>(resolved.size()),
            picked.data(),
            static_cast<int>(picked.size()));
        if (rc == 0) {
            return std::string(resolved.data());
        }
        if (rc < 0) {
            throw std::runtime_error("resolved Kokoro fallback voice path is too long");
        }
    }

    return first_existing(
        parent_dir(model_path),
        {
            "kokoro-voice-af_heart.gguf",
            "kokoro-voice-af_bella.gguf",
            "kokoro-voice-am_adam.gguf",
            "kokoro-voice-ff_siwis.gguf",
        });
}

void copy_language(char (&dest)[32], const std::string & language) {
    std::memset(dest, 0, sizeof(dest));
    std::strncpy(dest, language.c_str(), sizeof(dest) - 1);
}

} // namespace

class KokoroSynthesizer::Impl {
public:
    explicit Impl(KokoroTtsConfig config)
        : config_(std::move(config)) {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing Kokoro model path");
        }
        if (config_.output_dir.empty()) {
            throw std::runtime_error("missing TTS output directory");
        }
        if (config_.threads <= 0) {
            throw std::runtime_error("Kokoro thread count must be positive");
        }
        if (config_.length_scale <= 0.0f) {
            throw std::runtime_error("Kokoro length scale must be positive");
        }

        config_.language = normalize_language(config_.language);
        config_.model_path = resolve_model_path_for_language(config_);
        config_.voice_model_path = resolve_voice_path(config_, config_.model_path);

        require_file(config_.model_path, "Kokoro model");
        if (config_.voice_model_path.empty()) {
            throw std::runtime_error(
                "missing Kokoro voice pack: pass --tts-voice-model or place kokoro-voice-af_heart.gguf next to the model");
        }
        require_file(config_.voice_model_path, "Kokoro voice pack");

        kokoro_context_params params = kokoro_context_default_params();
        params.n_threads = config_.threads;
        params.verbosity = 1;
        params.use_gpu = config_.use_gpu;
        params.flash_attn = config_.flash_attention;
        params.length_scale = config_.length_scale;
        copy_language(params.espeak_lang, config_.language);

        context_.reset(kokoro_init_from_file(config_.model_path.c_str(), params));
        if (!context_) {
            throw std::runtime_error("failed to load Kokoro model: " + config_.model_path);
        }
        if (kokoro_load_voice_pack(context_.get(), config_.voice_model_path.c_str()) != 0) {
            throw std::runtime_error("failed to load Kokoro voice pack: " + config_.voice_model_path);
        }
        if (kokoro_set_language(context_.get(), config_.language.c_str()) != 0) {
            throw std::runtime_error("failed to set Kokoro language: " + config_.language);
        }
        kokoro_set_length_scale(context_.get(), config_.length_scale);
    }

    std::string synthesize(const std::string & text, uint64_t chunk_index, bool is_final) {
        const std::vector<float> pcm = synthesize_pcm(text);
        const std::string output_path = make_output_path(config_.output_dir, chunk_index, is_final);
        write_wav_mono_16(output_path, pcm.data(), static_cast<int32_t>(pcm.size()), kKokoroSampleRate);
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

        int n_samples = 0;
        float * pcm = config_.phonemes_input
            ? kokoro_synthesize_phonemes(context_.get(), text.c_str(), &n_samples)
            : kokoro_synthesize(context_.get(), text.c_str(), &n_samples);
        if (pcm == nullptr || n_samples <= 0) {
            kokoro_pcm_free(pcm);
            throw std::runtime_error("Kokoro synthesis failed");
        }

        std::vector<float> output(pcm, pcm + n_samples);
        kokoro_pcm_free(pcm);
        return output;
    }

private:
    KokoroTtsConfig config_;
    KokoroContextPtr context_;
};

KokoroSynthesizer::KokoroSynthesizer(KokoroTtsConfig config)
    : impl_(new Impl(std::move(config))) {
}

KokoroSynthesizer::~KokoroSynthesizer() {
    delete impl_;
}

std::string KokoroSynthesizer::synthesize(const std::string & text, uint64_t chunk_index, bool is_final) {
    return impl_->synthesize(text, chunk_index, is_final);
}

std::vector<float> KokoroSynthesizer::synthesize_pcm(const std::string & text) {
    return impl_->synthesize_pcm(text);
}

} // namespace vox::tts
