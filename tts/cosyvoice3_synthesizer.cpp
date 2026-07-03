#include "cosyvoice3_synthesizer.h"

#include "cosyvoice3_tts.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#endif

namespace vox::tts {
namespace {

struct CosyVoice3ContextDeleter {
    void operator()(cosyvoice3_tts_context * context) const {
        cosyvoice3_tts_free(context);
    }
};

using CosyVoice3ContextPtr = std::unique_ptr<cosyvoice3_tts_context, CosyVoice3ContextDeleter>;

std::string shell_quote(const std::string & value) {
    std::string quoted = "'";
    for (const char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

void run_player(const std::string & command, const std::string & output_path) {
    if (command.empty()) {
        throw std::runtime_error("missing TTS playback command");
    }

    const std::string full_command = shell_quote(command) + " " + shell_quote(output_path);
    const int status = std::system(full_command.c_str());
    if (status == -1) {
        throw std::runtime_error("failed to run TTS playback command");
    }

#if defined(WIFEXITED) && defined(WEXITSTATUS)
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        return;
    }
    if (WIFEXITED(status)) {
        throw std::runtime_error("TTS playback exited with status " + std::to_string(WEXITSTATUS(status)));
    }
    throw std::runtime_error("TTS playback did not exit normally");
#else
    if (status == 0) {
        return;
    }
    throw std::runtime_error("TTS playback exited with status " + std::to_string(status));
#endif
}

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

std::string resolve_flow_path(const CosyVoice3TtsConfig & config) {
    if (!config.flow_model_path.empty()) {
        return config.flow_model_path;
    }
    return first_existing(
        parent_dir(config.model_path),
        {
            "cosyvoice3-flow-q8_0.gguf",
            "cosyvoice3-flow-f16.gguf",
            "cosyvoice3-flow.gguf",
        });
}

std::string resolve_hift_path(const CosyVoice3TtsConfig & config) {
    if (!config.hift_model_path.empty()) {
        return config.hift_model_path;
    }
    return first_existing(
        parent_dir(config.model_path),
        {
            "cosyvoice3-hift-f16.gguf",
            "cosyvoice3-hift.gguf",
        });
}

std::string resolve_voices_path(const CosyVoice3TtsConfig & config) {
    if (!config.voices_model_path.empty()) {
        return config.voices_model_path;
    }
    return first_existing(
        parent_dir(config.model_path),
        {
            "cosyvoice3-voices.gguf",
            "voices.gguf",
        });
}

void require_file(const std::string & path, const std::string & label) {
    if (!file_exists(path)) {
        throw std::runtime_error("missing " + label + ": " + path);
    }
}

template <typename T>
void write_le(std::ofstream & out, T value) {
    for (size_t i = 0; i < sizeof(T); ++i) {
        out.put(static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 0xff));
    }
}

} // namespace

std::string default_tts_play_command() {
#if defined(__APPLE__)
    return "afplay";
#else
    return "aplay";
#endif
}

void play_audio_file(const std::string & command, const std::string & path) {
    run_player(command, path);
}

void write_wav_mono_16(const std::string & path, const float * samples, int32_t sample_count, int32_t sample_rate) {
    if (path.empty()) {
        throw std::runtime_error("missing wav output path");
    }
    if (samples == nullptr || sample_count <= 0) {
        throw std::runtime_error("cannot write empty wav");
    }
    if (sample_rate <= 0) {
        throw std::runtime_error("invalid wav sample rate");
    }

    const std::filesystem::path output_file(path);
    const std::filesystem::path output_parent = output_file.parent_path();
    if (!output_parent.empty()) {
        std::filesystem::create_directories(output_parent);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open wav output: " + path);
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
    const uint32_t data_size = static_cast<uint32_t>(sample_count) * block_align;
    const uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    write_le<uint32_t>(out, riff_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_le<uint32_t>(out, 16);
    write_le<uint16_t>(out, 1);
    write_le<uint16_t>(out, channels);
    write_le<uint32_t>(out, static_cast<uint32_t>(sample_rate));
    write_le<uint32_t>(out, byte_rate);
    write_le<uint16_t>(out, block_align);
    write_le<uint16_t>(out, bits_per_sample);
    out.write("data", 4);
    write_le<uint32_t>(out, data_size);

    for (int32_t i = 0; i < sample_count; ++i) {
        const float clipped = std::max(-1.0f, std::min(1.0f, samples[i]));
        const int16_t sample = static_cast<int16_t>(clipped * 32767.0f);
        write_le<int16_t>(out, sample);
    }
}

class CosyVoice3Synthesizer::Impl {
public:
    explicit Impl(CosyVoice3TtsConfig config)
        : config_(std::move(config)) {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing CosyVoice3 LLM model path");
        }
        if (config_.output_dir.empty()) {
            throw std::runtime_error("missing TTS output directory");
        }
        if (config_.voice.empty()) {
            throw std::runtime_error("missing CosyVoice3 voice name");
        }

        config_.flow_model_path = resolve_flow_path(config_);
        config_.hift_model_path = resolve_hift_path(config_);
        config_.voices_model_path = resolve_voices_path(config_);

        require_file(config_.model_path, "CosyVoice3 LLM model");
        require_file(config_.flow_model_path, "CosyVoice3 flow model");
        require_file(config_.hift_model_path, "CosyVoice3 HiFT model");
        require_file(config_.voices_model_path, "CosyVoice3 voices model");

        cosyvoice3_tts_context_params params = cosyvoice3_tts_context_default_params();
        params.n_threads = config_.threads;
        params.verbosity = 1;
        params.use_gpu = config_.use_gpu;
        params.flash_attn = config_.flash_attention;
        params.temperature = config_.temperature;
        params.seed = config_.seed;
        params.max_tokens = config_.max_tokens;

        context_.reset(cosyvoice3_tts_init_from_file(config_.model_path.c_str(), params));
        if (!context_) {
            throw std::runtime_error("failed to load CosyVoice3 LLM model: " + config_.model_path);
        }
        if (cosyvoice3_tts_init_flow_from_file(context_.get(), config_.flow_model_path.c_str()) != 0) {
            throw std::runtime_error("failed to load CosyVoice3 flow model: " + config_.flow_model_path);
        }
        if (cosyvoice3_tts_init_hift_from_file(context_.get(), config_.hift_model_path.c_str()) != 0) {
            throw std::runtime_error("failed to load CosyVoice3 HiFT model: " + config_.hift_model_path);
        }
        if (cosyvoice3_tts_init_voices_from_file(context_.get(), config_.voices_model_path.c_str()) != 0) {
            throw std::runtime_error("failed to load CosyVoice3 voices model: " + config_.voices_model_path);
        }

        if (!voice_exists(config_.voice)) {
            throw std::runtime_error("CosyVoice3 voice not found: " + config_.voice);
        }
    }

    std::string synthesize(const std::string & text, uint64_t chunk_index, bool is_final) {
        const std::vector<float> pcm = synthesize_pcm(text);
        const std::string output_path = make_output_path(config_.output_dir, chunk_index, is_final);
        write_wav_mono_16(output_path, pcm.data(), static_cast<int32_t>(pcm.size()), kCosyVoice3SampleRate);
        if (config_.play_after_synthesis) {
            run_player(
                config_.play_command.empty() ? default_tts_play_command() : config_.play_command,
                output_path);
        }
        return output_path;
    }

    std::vector<float> synthesize_pcm(const std::string & text) {
        if (text.empty()) {
            throw std::runtime_error("cannot synthesize empty text");
        }
        if (config_.temperature != 0.0f) {
            cosyvoice3_tts_set_temperature(context_.get(), config_.temperature);
        }
        cosyvoice3_tts_set_seed(context_.get(), config_.seed);

        int n_samples = 0;
        float * pcm = cosyvoice3_tts_synth(context_.get(), text.c_str(), config_.voice.c_str(), &n_samples);
        if (pcm == nullptr || n_samples <= 0) {
            std::free(pcm);
            throw std::runtime_error("CosyVoice3 synthesis failed");
        }

        std::vector<float> output(pcm, pcm + n_samples);
        std::free(pcm);
        return output;
    }

private:
    bool voice_exists(const std::string & voice) const {
        const int n_voices = cosyvoice3_tts_n_voices(context_.get());
        for (int i = 0; i < n_voices; ++i) {
            const char * name = cosyvoice3_tts_voice_name(context_.get(), i);
            if (name != nullptr && voice == name) {
                return true;
            }
        }
        return false;
    }

    CosyVoice3TtsConfig config_;
    CosyVoice3ContextPtr context_;
};

CosyVoice3Synthesizer::CosyVoice3Synthesizer(CosyVoice3TtsConfig config)
    : impl_(new Impl(std::move(config))) {
}

CosyVoice3Synthesizer::~CosyVoice3Synthesizer() {
    delete impl_;
}

std::string CosyVoice3Synthesizer::synthesize(const std::string & text, uint64_t chunk_index, bool is_final) {
    return impl_->synthesize(text, chunk_index, is_final);
}

std::vector<float> CosyVoice3Synthesizer::synthesize_pcm(const std::string & text) {
    return impl_->synthesize_pcm(text);
}

} // namespace vox::tts
