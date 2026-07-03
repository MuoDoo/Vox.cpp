#include "async_transcript_translator.h"
#include "async_text_to_speech.h"
#include "async_audio_player.h"
#include "microphone_audio_source.h"
#include "streaming_qwen_asr.h"
#include "streaming_whisper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic_bool g_running{true};

#ifndef VOX_PROJECT_ROOT
#define VOX_PROJECT_ROOT "."
#endif

struct CliOptions {
    std::string asr_engine = "qwen3";
    std::string asr_mmproj_path;
    std::string vad_model_path;
    std::string tts_model_path;
    std::string tts_flow_model_path;
    std::string tts_hift_model_path;
    std::string tts_voices_model_path;
    std::string tts_voice = "zero_shot";
    std::string tts_output_dir = "tts-output";
    std::string tts_play_command;
    int32_t capture_device_id = -1;
    int32_t step_ms = 0;
    int32_t window_ms = 0;
    int32_t overlap_ms = -1;
    int32_t final_silence_steps = 0;
    int32_t vad_min_silence_ms = 0;
    int32_t vad_speech_pad_ms = -1;
    int32_t vad_max_speech_ms = 0;
    int32_t tts_threads = 0;
    int32_t tts_max_tokens = 0;
    int32_t tts_flow_steps = 0;
    float rms_threshold = -1.0f;
    float no_speech_threshold = -1.0f;
    float vad_threshold = -1.0f;
    float min_token_probability = 0.0f;
    float input_gain = 1.0f;
    float tts_temperature = 0.8f;
    uint64_t tts_seed = 42;
    bool show_help = false;
    bool disable_silence_gate = false;
    bool debug_audio = false;
    bool whisper_debug = false;
    bool disable_gpu = false;
    bool disable_flash_attention = false;
    bool final_only = false;
    bool tts_play = false;
    bool tts_partials = false;
    std::vector<std::string> positional;
};

enum class AsrEngine {
    Whisper,
    Qwen3,
};

struct CommonAsrConfig {
    std::string model_path;
    std::string language = "auto";
    int32_t threads = 4;
    int32_t step_ms = 2000;
    int32_t window_ms = 6000;
    int32_t overlap_ms = 300;
    float min_audio_rms = 0.001f;
    bool debug = false;
    bool use_gpu = true;
    bool flash_attention = true;
};

struct AudioStats {
    double rms = 0.0;
    float peak = 0.0f;
};

void stop(int) {
    g_running = false;
}

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

std::string resolve_model_path(const std::string & model_path) {
    namespace fs = std::filesystem;

    const fs::path requested(model_path);
    if (requested.is_absolute() || file_exists(model_path)) {
        return model_path;
    }

    const fs::path project_path = fs::path(VOX_PROJECT_ROOT) / requested;
    return project_path.string();
}

AsrEngine parse_asr_engine(const std::string & value) {
    if (value == "whisper" || value == "whisper.cpp") {
        return AsrEngine::Whisper;
    }
    if (value == "qwen3" || value == "qwen3-asr" || value == "llama" || value == "llama.cpp") {
        return AsrEngine::Qwen3;
    }
    throw std::runtime_error("unknown ASR engine: " + value);
}

const char * asr_engine_name(AsrEngine engine) {
    switch (engine) {
    case AsrEngine::Whisper:
        return "whisper";
    case AsrEngine::Qwen3:
        return "qwen3";
    }
    return "unknown";
}

int32_t parse_i32(const std::string & value, const std::string & option_name) {
    size_t parsed = 0;
    int parsed_value = 0;
    try {
        parsed_value = std::stoi(value, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid value for " + option_name + ": " + value);
    }

    if (parsed != value.size()) {
        throw std::runtime_error("invalid value for " + option_name + ": " + value);
    }
    return static_cast<int32_t>(parsed_value);
}

float parse_float(const std::string & value, const std::string & option_name) {
    size_t parsed = 0;
    float parsed_value = 0.0f;
    try {
        parsed_value = std::stof(value, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid value for " + option_name + ": " + value);
    }

    if (parsed != value.size()) {
        throw std::runtime_error("invalid value for " + option_name + ": " + value);
    }
    return parsed_value;
}

uint64_t parse_u64(const std::string & value, const std::string & option_name) {
    size_t parsed = 0;
    unsigned long long parsed_value = 0;
    try {
        parsed_value = std::stoull(value, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid value for " + option_name + ": " + value);
    }

    if (parsed != value.size()) {
        throw std::runtime_error("invalid value for " + option_name + ": " + value);
    }
    return static_cast<uint64_t>(parsed_value);
}

std::string option_value(
    const std::string & arg,
    const std::string & option_name,
    int & index,
    int argc,
    char ** argv) {
    const std::string prefix = option_name + "=";
    if (arg.rfind(prefix, 0) == 0) {
        return arg.substr(prefix.size());
    }
    if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + option_name);
    }
    return argv[++index];
}

CliOptions parse_cli(int argc, char ** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            options.show_help = true;
        } else if (arg == "--debug-audio") {
            options.debug_audio = true;
        } else if (arg == "--whisper-debug") {
            options.whisper_debug = true;
        } else if (arg == "--asr-engine" || arg.rfind("--asr-engine=", 0) == 0) {
            options.asr_engine = option_value(arg, "--asr-engine", i, argc, argv);
        } else if (arg == "--asr-mmproj" || arg.rfind("--asr-mmproj=", 0) == 0) {
            options.asr_mmproj_path = option_value(arg, "--asr-mmproj", i, argc, argv);
        } else if (arg == "--vad-model" || arg.rfind("--vad-model=", 0) == 0) {
            options.vad_model_path = option_value(arg, "--vad-model", i, argc, argv);
        } else if (arg == "--tts-model" || arg.rfind("--tts-model=", 0) == 0) {
            options.tts_model_path = option_value(arg, "--tts-model", i, argc, argv);
        } else if (arg == "--tts-flow-model" || arg.rfind("--tts-flow-model=", 0) == 0) {
            options.tts_flow_model_path = option_value(arg, "--tts-flow-model", i, argc, argv);
        } else if (arg == "--tts-hift-model" || arg.rfind("--tts-hift-model=", 0) == 0) {
            options.tts_hift_model_path = option_value(arg, "--tts-hift-model", i, argc, argv);
        } else if (arg == "--tts-voices-model" || arg.rfind("--tts-voices-model=", 0) == 0) {
            options.tts_voices_model_path = option_value(arg, "--tts-voices-model", i, argc, argv);
        } else if (arg == "--tts-voice" || arg.rfind("--tts-voice=", 0) == 0) {
            options.tts_voice = option_value(arg, "--tts-voice", i, argc, argv);
        } else if (arg == "--tts-output-dir" || arg.rfind("--tts-output-dir=", 0) == 0) {
            options.tts_output_dir = option_value(arg, "--tts-output-dir", i, argc, argv);
        } else if (arg == "--tts-play-command" || arg.rfind("--tts-play-command=", 0) == 0) {
            options.tts_play_command = option_value(arg, "--tts-play-command", i, argc, argv);
        } else if (arg == "--tts-threads" || arg.rfind("--tts-threads=", 0) == 0) {
            options.tts_threads = parse_i32(option_value(arg, "--tts-threads", i, argc, argv), "--tts-threads");
        } else if (arg == "--tts-max-tokens" || arg.rfind("--tts-max-tokens=", 0) == 0) {
            options.tts_max_tokens =
                parse_i32(option_value(arg, "--tts-max-tokens", i, argc, argv), "--tts-max-tokens");
        } else if (arg == "--tts-flow-steps" || arg.rfind("--tts-flow-steps=", 0) == 0) {
            options.tts_flow_steps =
                parse_i32(option_value(arg, "--tts-flow-steps", i, argc, argv), "--tts-flow-steps");
        } else if (arg == "--tts-temperature" || arg.rfind("--tts-temperature=", 0) == 0) {
            options.tts_temperature =
                parse_float(option_value(arg, "--tts-temperature", i, argc, argv), "--tts-temperature");
        } else if (arg == "--tts-seed" || arg.rfind("--tts-seed=", 0) == 0) {
            options.tts_seed = parse_u64(option_value(arg, "--tts-seed", i, argc, argv), "--tts-seed");
        } else if (arg == "--no-gpu") {
            options.disable_gpu = true;
        } else if (arg == "--no-flash-attn") {
            options.disable_flash_attention = true;
        } else if (arg == "--final-only") {
            options.final_only = true;
        } else if (arg == "--tts-play") {
            options.tts_play = true;
        } else if (arg == "--tts-partials") {
            options.tts_partials = true;
        } else if (arg == "--no-vad" || arg == "--no-silence-gate") {
            options.disable_silence_gate = true;
        } else if (arg == "-c" || arg == "--capture" || arg.rfind("--capture=", 0) == 0) {
            options.capture_device_id = parse_i32(option_value(arg, "--capture", i, argc, argv), "--capture");
        } else if (arg == "--step" || arg.rfind("--step=", 0) == 0) {
            options.step_ms = parse_i32(option_value(arg, "--step", i, argc, argv), "--step");
        } else if (arg == "--length" || arg.rfind("--length=", 0) == 0) {
            options.window_ms = parse_i32(option_value(arg, "--length", i, argc, argv), "--length");
        } else if (arg == "--keep" || arg.rfind("--keep=", 0) == 0) {
            options.overlap_ms = parse_i32(option_value(arg, "--keep", i, argc, argv), "--keep");
        } else if (arg == "--final-on-silence" || arg.rfind("--final-on-silence=", 0) == 0) {
            options.final_silence_steps =
                parse_i32(option_value(arg, "--final-on-silence", i, argc, argv), "--final-on-silence");
        } else if (arg == "--vad-min-silence-ms" || arg.rfind("--vad-min-silence-ms=", 0) == 0) {
            options.vad_min_silence_ms =
                parse_i32(option_value(arg, "--vad-min-silence-ms", i, argc, argv), "--vad-min-silence-ms");
        } else if (arg == "--vad-speech-pad-ms" || arg.rfind("--vad-speech-pad-ms=", 0) == 0) {
            options.vad_speech_pad_ms =
                parse_i32(option_value(arg, "--vad-speech-pad-ms", i, argc, argv), "--vad-speech-pad-ms");
        } else if (arg == "--vad-max-speech-ms" || arg.rfind("--vad-max-speech-ms=", 0) == 0) {
            options.vad_max_speech_ms =
                parse_i32(option_value(arg, "--vad-max-speech-ms", i, argc, argv), "--vad-max-speech-ms");
        } else if (arg == "--rms-threshold" || arg.rfind("--rms-threshold=", 0) == 0) {
            options.rms_threshold =
                parse_float(option_value(arg, "--rms-threshold", i, argc, argv), "--rms-threshold");
        } else if (arg == "--vad-threshold" || arg.rfind("--vad-threshold=", 0) == 0) {
            options.vad_threshold =
                parse_float(option_value(arg, "--vad-threshold", i, argc, argv), "--vad-threshold");
        } else if (arg == "--gain" || arg.rfind("--gain=", 0) == 0) {
            options.input_gain = parse_float(option_value(arg, "--gain", i, argc, argv), "--gain");
        } else if (arg == "--min-token-p" || arg.rfind("--min-token-p=", 0) == 0) {
            options.min_token_probability =
                parse_float(option_value(arg, "--min-token-p", i, argc, argv), "--min-token-p");
        } else if (arg == "--min-confidence" || arg.rfind("--min-confidence=", 0) == 0) {
            options.min_token_probability =
                parse_float(option_value(arg, "--min-confidence", i, argc, argv), "--min-confidence");
        } else if (arg == "--no-speech-thold" || arg.rfind("--no-speech-thold=", 0) == 0) {
            options.no_speech_threshold =
                parse_float(option_value(arg, "--no-speech-thold", i, argc, argv), "--no-speech-thold");
        } else if (arg == "--no-speech-threshold" || arg.rfind("--no-speech-threshold=", 0) == 0) {
            options.no_speech_threshold = parse_float(
                option_value(arg, "--no-speech-threshold", i, argc, argv),
                "--no-speech-threshold");
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown option: " + arg);
        } else {
            options.positional.push_back(arg);
        }
    }

    if (options.capture_device_id < -1) {
        throw std::runtime_error("capture device id must be -1 for default or a non-negative device index");
    }
    if (options.step_ms < 0 || options.window_ms < 0 || options.overlap_ms < -1) {
        throw std::runtime_error("audio window options must be non-negative");
    }
    if (options.final_silence_steps < 0) {
        throw std::runtime_error("final silence steps must be non-negative");
    }
    if (options.vad_min_silence_ms < 0 || options.vad_speech_pad_ms < -1 || options.vad_max_speech_ms < 0) {
        throw std::runtime_error("VAD timing options must be non-negative");
    }
    if (options.rms_threshold < -1.0f) {
        throw std::runtime_error("rms threshold must be non-negative");
    }
    if (options.no_speech_threshold < -1.0f) {
        throw std::runtime_error("no-speech threshold must be non-negative");
    }
    if (options.vad_threshold < -1.0f || options.vad_threshold > 1.0f) {
        throw std::runtime_error("VAD threshold must be between 0 and 1");
    }
    if (options.input_gain <= 0.0f) {
        throw std::runtime_error("input gain must be positive");
    }
    if (options.min_token_probability < 0.0f || options.min_token_probability > 1.0f) {
        throw std::runtime_error("minimum token probability must be between 0 and 1");
    }
    if (options.positional.size() > 4) {
        throw std::runtime_error("too many positional arguments");
    }
    if (!options.tts_model_path.empty()) {
        if (options.tts_voice.empty()) {
            throw std::runtime_error("TTS voice must not be empty");
        }
        if (options.tts_output_dir.empty()) {
            throw std::runtime_error("TTS output directory must not be empty");
        }
        if (options.tts_threads < 0) {
            throw std::runtime_error("TTS thread count must be positive");
        }
        if (options.tts_max_tokens < 0) {
            throw std::runtime_error("TTS max tokens must be non-negative");
        }
        if (options.tts_flow_steps < 0) {
            throw std::runtime_error("TTS flow steps must be non-negative");
        }
        if (options.tts_temperature < 0.0f) {
            throw std::runtime_error("TTS temperature must be non-negative");
        }
    }
    parse_asr_engine(options.asr_engine);
    return options;
}

void print_usage(const char * program) {
    std::cout << "usage: " << program << " [options] [asr_model] [language] [translation_model] [target_language]\n"
              << "\n"
              << "options:\n"
              << "      --asr-engine NAME\n"
              << "                      ASR backend: qwen3 or whisper; default qwen3\n"
              << "      --asr-mmproj PATH\n"
              << "                      Qwen3-ASR multimodal projector GGUF path\n"
              << "      --vad-model PATH\n"
              << "                      Silero VAD model for Qwen3-ASR utterance mode\n"
              << "      --tts-model PATH\n"
              << "                      enable native CosyVoice3 TTS with the LLM GGUF\n"
              << "      --tts-flow-model PATH\n"
              << "                      optional CosyVoice3 flow GGUF path\n"
              << "      --tts-hift-model PATH\n"
              << "                      optional CosyVoice3 HiFT vocoder GGUF path\n"
              << "      --tts-voices-model PATH\n"
              << "                      optional CosyVoice3 baked voices GGUF path\n"
              << "      --tts-voice NAME\n"
              << "                      CosyVoice3 baked voice name; default zero_shot\n"
              << "      --tts-output-dir DIR\n"
              << "                      directory for synthesized wav files; default tts-output\n"
              << "      --tts-threads N\n"
              << "                      TTS CPU threads; default ASR thread count\n"
              << "      --tts-temperature N\n"
              << "                      TTS speech-token sampling temperature; default 0.8\n"
              << "      --tts-seed N\n"
              << "                      TTS speech-token RNG seed; default 42\n"
              << "      --tts-max-tokens N\n"
              << "                      TTS speech-token decode cap; default model runtime value\n"
              << "      --tts-flow-steps N\n"
              << "                      CFM flow Euler steps; default model value (10); fewer = faster, lower quality\n"
              << "      --tts-play\n"
              << "                      play each synthesized wav after generation\n"
              << "      --tts-play-command PATH\n"
              << "                      player command for --tts-play; default afplay on macOS, aplay elsewhere\n"
              << "      --tts-partials\n"
              << "                      synthesize partial updates too; default final translations only\n"
              << "  -c, --capture ID    capture device index, or -1 for system default\n"
              << "      --step MS       audio step size in milliseconds\n"
              << "      --length MS     audio window size in milliseconds\n"
              << "      --keep MS       overlap to keep from the previous window in milliseconds\n"
              << "      --final-on-silence N\n"
              << "                      emit latest transcript as final after N silent windows\n"
              << "      --final-only    suppress partial ASR output; implies --final-on-silence 1\n"
              << "      --vad-threshold N\n"
              << "                      Qwen3-ASR VAD speech probability threshold; default 0.5\n"
              << "      --vad-min-silence-ms MS\n"
              << "                      Qwen3-ASR silence duration before closing an utterance\n"
              << "      --vad-speech-pad-ms MS\n"
              << "                      Qwen3-ASR audio padding kept around detected speech\n"
              << "      --vad-max-speech-ms MS\n"
              << "                      Qwen3-ASR maximum utterance length before forced finalization\n"
              << "      --rms-threshold N\n"
              << "                      silence gate RMS threshold; 0 disables it\n"
              << "      --gain N        multiply microphone samples before ASR\n"
              << "      --min-token-p N minimum average token probability for emitted segments\n"
              << "      --no-speech-thold N\n"
              << "                      Whisper no-speech threshold; 1.0 disables most no-speech filtering\n"
              << "      --no-vad        disable the silence gate\n"
              << "      --no-gpu        disable ASR GPU acceleration\n"
              << "      --no-flash-attn disable ASR flash attention\n"
              << "      --debug-audio   print microphone sample count, RMS, and peak\n"
              << "      --whisper-debug print Whisper diagnostics, or Qwen3-ASR mtmd timings\n"
              << "  -h, --help          show this help\n";
}

AudioStats measure_audio(const std::vector<float> & samples) {
    AudioStats stats;
    if (samples.empty()) {
        return stats;
    }

    double sum_squares = 0.0;
    size_t finite_count = 0;
    for (const float sample : samples) {
        if (!std::isfinite(sample)) {
            continue;
        }
        const float abs_sample = std::fabs(sample);
        stats.peak = std::max(stats.peak, abs_sample);
        sum_squares += static_cast<double>(sample) * static_cast<double>(sample);
        ++finite_count;
    }

    if (finite_count > 0) {
        stats.rms = std::sqrt(sum_squares / static_cast<double>(finite_count));
    }
    return stats;
}

void apply_gain(std::vector<float> & samples, float gain) {
    if (gain == 1.0f) {
        return;
    }
    for (float & sample : samples) {
        sample = std::max(-1.0f, std::min(1.0f, sample * gain));
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);

    try {
        const CliOptions cli = parse_cli(argc, argv);
        if (cli.show_help) {
            print_usage(argv[0]);
            return 0;
        }

        const AsrEngine asr_engine = parse_asr_engine(cli.asr_engine);
        const std::string default_asr_model =
            asr_engine == AsrEngine::Whisper
                ? "models/ggml-base.bin"
                : "models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf";
        const std::string default_qwen_mmproj =
            "models/asr/qwen3-asr-1.7b/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf";
        const std::string default_qwen_vad_model = "models/vad/ggml-silero-v6.2.0.bin";
        const std::string bundled_qwen_vad_model =
            "external/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin";

        CommonAsrConfig common_config;
        common_config.model_path = resolve_model_path(
            !cli.positional.empty() ? cli.positional[0] : default_asr_model);
        if (cli.positional.size() > 1) {
            common_config.language = cli.positional[1];
        }
        if (cli.step_ms > 0) {
            common_config.step_ms = cli.step_ms;
        }
        if (cli.window_ms > 0) {
            common_config.window_ms = cli.window_ms;
        }
        if (cli.overlap_ms >= 0) {
            common_config.overlap_ms = cli.overlap_ms;
        }
        if (cli.disable_silence_gate) {
            common_config.min_audio_rms = 0.0f;
        } else if (cli.rms_threshold >= 0.0f) {
            common_config.min_audio_rms = cli.rms_threshold;
        } else if (common_config.language != "auto") {
            common_config.min_audio_rms = 0.0f;
        }
        common_config.debug = cli.whisper_debug;
        if (cli.disable_gpu) {
            common_config.use_gpu = false;
        }
        if (cli.disable_flash_attention) {
            common_config.flash_attention = false;
        }

        if (!file_exists(common_config.model_path)) {
            std::cerr << "Missing ASR model: " << common_config.model_path << "\n";
            if (asr_engine == AsrEngine::Whisper) {
                std::cerr << "Download: ./external/whisper.cpp/models/download-ggml-model.sh base models\n";
            } else {
                std::cerr << "Download: scripts/download-qwen3-asr-gguf.sh\n";
            }
            return 1;
        }

        std::string qwen_mmproj_path;
        if (asr_engine == AsrEngine::Qwen3) {
            qwen_mmproj_path = resolve_model_path(
                !cli.asr_mmproj_path.empty() ? cli.asr_mmproj_path : default_qwen_mmproj);
            if (!file_exists(qwen_mmproj_path)) {
                std::cerr << "Missing Qwen3-ASR mmproj: " << qwen_mmproj_path << "\n"
                          << "Download: scripts/download-qwen3-asr-gguf.sh\n";
                return 1;
            }
        }

        std::string qwen_vad_model_path;
        if (asr_engine == AsrEngine::Qwen3 && !cli.disable_silence_gate) {
            if (!cli.vad_model_path.empty()) {
                qwen_vad_model_path = resolve_model_path(cli.vad_model_path);
                if (!file_exists(qwen_vad_model_path)) {
                    std::cerr << "Missing Qwen3-ASR VAD model: " << qwen_vad_model_path << "\n";
                    return 1;
                }
            } else {
                const std::string default_path = resolve_model_path(default_qwen_vad_model);
                const std::string bundled_path = resolve_model_path(bundled_qwen_vad_model);
                if (file_exists(default_path)) {
                    qwen_vad_model_path = default_path;
                } else if (file_exists(bundled_path)) {
                    qwen_vad_model_path = bundled_path;
                }
            }
            if (!qwen_vad_model_path.empty() && cli.step_ms == 0) {
                common_config.step_ms = 500;
            }
        }

        const std::string translation_model_path =
            cli.positional.size() > 2 ? resolve_model_path(cli.positional[2]) : std::string();
        if (!translation_model_path.empty() && !file_exists(translation_model_path)) {
            std::cerr << "Missing translation model: " << translation_model_path << "\n"
                      << "Download: scripts/download-hymt-gguf.sh\n";
            return 1;
        }

        std::mutex output_mutex;
        std::unique_ptr<vox::pipeline::AsyncTextToSpeech> tts;
        std::unique_ptr<vox::pipeline::AsyncAudioPlayer> player;
        std::string tts_model_path;
        std::string tts_flow_model_path;
        std::string tts_hift_model_path;
        std::string tts_voices_model_path;
        std::string tts_output_dir;
        std::unique_ptr<vox::pipeline::AsyncTranscriptTranslator> translator;
        const int32_t final_silence_steps =
            std::max(cli.final_silence_steps, cli.final_only ? int32_t{1} : int32_t{0});
        const float whisper_no_speech_threshold =
            cli.no_speech_threshold >= 0.0f ? cli.no_speech_threshold : 0.95f;

        if (!cli.tts_model_path.empty()) {
            tts_model_path = resolve_model_path(cli.tts_model_path);
            if (!file_exists(tts_model_path)) {
                std::cerr << "Missing TTS model: " << tts_model_path << "\n"
                          << "Download: scripts/download-cosyvoice3-tts-gguf.sh\n";
                return 1;
            }

            if (!cli.tts_flow_model_path.empty()) {
                tts_flow_model_path = resolve_model_path(cli.tts_flow_model_path);
                if (!file_exists(tts_flow_model_path)) {
                    std::cerr << "Missing TTS flow model: " << tts_flow_model_path << "\n";
                    return 1;
                }
            }
            if (!cli.tts_hift_model_path.empty()) {
                tts_hift_model_path = resolve_model_path(cli.tts_hift_model_path);
                if (!file_exists(tts_hift_model_path)) {
                    std::cerr << "Missing TTS HiFT model: " << tts_hift_model_path << "\n";
                    return 1;
                }
            }
            if (!cli.tts_voices_model_path.empty()) {
                tts_voices_model_path = resolve_model_path(cli.tts_voices_model_path);
                if (!file_exists(tts_voices_model_path)) {
                    std::cerr << "Missing TTS voices model: " << tts_voices_model_path << "\n";
                    return 1;
                }
            }

            tts_output_dir = cli.tts_output_dir;
            vox::tts::CosyVoice3TtsConfig tts_config;
            tts_config.model_path = tts_model_path;
            tts_config.flow_model_path = tts_flow_model_path;
            tts_config.hift_model_path = tts_hift_model_path;
            tts_config.voices_model_path = tts_voices_model_path;
            tts_config.voice = cli.tts_voice;
            tts_config.output_dir = tts_output_dir;
            tts_config.threads = cli.tts_threads > 0 ? cli.tts_threads : common_config.threads;
            tts_config.max_tokens = cli.tts_max_tokens;
            tts_config.flow_steps = cli.tts_flow_steps;
            tts_config.temperature = cli.tts_temperature;
            tts_config.seed = cli.tts_seed;
            tts_config.use_gpu = common_config.use_gpu;
            tts_config.flash_attention = common_config.flash_attention;
            tts_config.play_after_synthesis = false;

            if (cli.tts_play) {
                const std::string play_command =
                    cli.tts_play_command.empty() ? vox::tts::default_tts_play_command()
                                                 : cli.tts_play_command;
                player = std::make_unique<vox::pipeline::AsyncAudioPlayer>(
                    [&output_mutex, play_command](const std::string & path) {
                        const auto start = std::chrono::steady_clock::now();
                        vox::tts::play_audio_file(play_command, path);
                        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                    std::chrono::steady_clock::now() - start)
                                                    .count();
                        std::lock_guard<std::mutex> lock(output_mutex);
                        std::cout << "play: " << path << " (play " << elapsed_ms << " ms)\n";
                        std::cout.flush();
                    });
            }

            tts = std::make_unique<vox::pipeline::AsyncTextToSpeech>(
                std::move(tts_config),
                [&output_mutex, &player](vox::pipeline::TextToSpeechResult result) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    if (!result.error.empty()) {
                        std::cerr << "[" << result.request.chunk_index
                                  << "] tts failed: " << result.error << "\n";
                        return;
                    }
                    std::cout << "[" << result.request.chunk_index << "] "
                              << (result.request.is_final ? "tts final: " : "tts: ")
                              << result.output_path
                              << " (synth " << result.elapsed_ms << " ms)\n";
                    std::cout.flush();
                    if (player) {
                        player->submit(result.output_path);
                    }
                });
        }

        const auto submit_tts =
            [&tts, &cli](uint64_t chunk_index, const std::string & text, bool is_final) {
            if (!tts || text.empty()) {
                return;
            }
            if (!is_final && !cli.tts_partials) {
                return;
            }
            tts->submit({chunk_index, text, is_final});
        };

        if (!translation_model_path.empty()) {
            vox::translate::LlamaTranslatorConfig translate_config;
            translate_config.model_path = translation_model_path;
            translate_config.source_language = common_config.language;
            if (cli.positional.size() > 3) {
                translate_config.target_language = cli.positional[3];
            }

            translator = std::make_unique<vox::pipeline::AsyncTranscriptTranslator>(
                std::move(translate_config),
                [&output_mutex, submit_tts](vox::pipeline::TranslationResult result) {
                    if (!result.error.empty()) {
                        std::lock_guard<std::mutex> lock(output_mutex);
                        std::cerr << "[" << result.transcript.chunk_index
                                  << "] translation failed: " << result.error << "\n";
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> lock(output_mutex);
                        std::cout << "[" << result.transcript.chunk_index << "] "
                                  << (result.transcript.is_final ? "translation final: " : "translation: ")
                                  << result.translation
                                  << " (translate " << result.elapsed_ms << " ms)\n";
                        std::cout.flush();
                    }
                    submit_tts(result.transcript.chunk_index, result.translation, result.transcript.is_final);
                });
        }

        std::unique_ptr<vox::asr::StreamingWhisper> whisper_recognizer;
        std::unique_ptr<vox::asr::StreamingQwenAsr> qwen_recognizer;
        if (asr_engine == AsrEngine::Whisper) {
            vox::asr::StreamingWhisperConfig whisper_config;
            whisper_config.model_path = common_config.model_path;
            whisper_config.language = common_config.language;
            whisper_config.threads = common_config.threads;
            whisper_config.step_ms = common_config.step_ms;
            whisper_config.window_ms = common_config.window_ms;
            whisper_config.overlap_ms = common_config.overlap_ms;
            whisper_config.min_audio_rms = common_config.min_audio_rms;
            whisper_config.no_speech_threshold = whisper_no_speech_threshold;
            whisper_config.min_token_probability = cli.min_token_probability;
            whisper_config.debug = common_config.debug;
            whisper_config.use_gpu = common_config.use_gpu;
            whisper_config.flash_attention = common_config.flash_attention;
            whisper_recognizer = std::make_unique<vox::asr::StreamingWhisper>(std::move(whisper_config));
        } else {
            vox::asr::StreamingQwenAsrConfig qwen_config;
            qwen_config.model_path = common_config.model_path;
            qwen_config.mmproj_path = qwen_mmproj_path;
            qwen_config.language = common_config.language;
            qwen_config.threads = common_config.threads;
            qwen_config.step_ms = common_config.step_ms;
            qwen_config.window_ms = common_config.window_ms;
            qwen_config.overlap_ms = common_config.overlap_ms;
            qwen_config.min_audio_rms = common_config.min_audio_rms;
            qwen_config.debug = common_config.debug;
            qwen_config.use_gpu = common_config.use_gpu;
            qwen_config.mmproj_use_gpu = common_config.use_gpu;
            qwen_config.flash_attention = common_config.flash_attention;
            qwen_config.vad_model_path = qwen_vad_model_path;
            qwen_config.use_vad = !cli.disable_silence_gate && !qwen_vad_model_path.empty();
            qwen_config.utterance_mode = qwen_config.use_vad;
            if (cli.vad_threshold >= 0.0f) {
                qwen_config.vad_threshold = cli.vad_threshold;
            }
            if (cli.vad_min_silence_ms > 0) {
                qwen_config.vad_min_silence_ms = cli.vad_min_silence_ms;
            }
            if (cli.vad_speech_pad_ms >= 0) {
                qwen_config.vad_speech_pad_ms = cli.vad_speech_pad_ms;
            }
            if (cli.vad_max_speech_ms > 0) {
                qwen_config.vad_max_speech_ms = cli.vad_max_speech_ms;
            }
            qwen_recognizer = std::make_unique<vox::asr::StreamingQwenAsr>(std::move(qwen_config));
        }

        const auto push_asr_audio =
            [&whisper_recognizer, &qwen_recognizer, asr_engine](const std::vector<float> & samples) {
            if (asr_engine == AsrEngine::Whisper) {
                return whisper_recognizer->push_audio(samples);
            }
            return qwen_recognizer->push_audio(samples);
        };
        const auto flush_asr =
            [&whisper_recognizer, &qwen_recognizer, asr_engine]() {
            if (asr_engine == AsrEngine::Whisper) {
                return whisper_recognizer->flush();
            }
            return qwen_recognizer->flush();
        };

        vox::app::MicrophoneAudioSource microphone(cli.capture_device_id, common_config.window_ms);
        microphone.start();

        {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "vox listening: " << common_config.model_path
                      << " asr_engine=" << asr_engine_name(asr_engine)
                      << " language=" << common_config.language
                      << " capture=" << cli.capture_device_id
                      << " step_ms=" << common_config.step_ms
                      << " window_ms=" << common_config.window_ms
                      << " keep_ms=" << common_config.overlap_ms
                      << " rms_threshold=" << common_config.min_audio_rms
                      << " final_on_silence=" << final_silence_steps
                      << " final_only=" << cli.final_only
                      << " gain=" << cli.input_gain
                      << " gpu=" << common_config.use_gpu
                      << " flash_attn=" << common_config.flash_attention;
            if (asr_engine == AsrEngine::Whisper) {
                std::cout << " no_speech_thold=" << whisper_no_speech_threshold
                          << " min_token_p=" << cli.min_token_probability;
            } else {
                std::cout << " mmproj=" << qwen_mmproj_path
                          << " qwen_vad=" << (!qwen_vad_model_path.empty() ? "on" : "off");
                if (!qwen_vad_model_path.empty()) {
                    const float qwen_vad_threshold =
                        cli.vad_threshold >= 0.0f ? cli.vad_threshold : 0.5f;
                    std::cout << " vad_model=" << qwen_vad_model_path
                              << " vad_threshold=" << qwen_vad_threshold;
                }
            }
            if (translator) {
                std::cout << " translation_model=" << translation_model_path;
            }
            if (tts) {
                std::cout << " tts_model=" << tts_model_path
                          << " tts_voice=" << cli.tts_voice
                          << " tts_output_dir=" << tts_output_dir
                          << " tts_threads=" << (cli.tts_threads > 0 ? cli.tts_threads : common_config.threads)
                          << " tts_temperature=" << cli.tts_temperature
                          << " tts_seed=" << cli.tts_seed
                          << " tts_max_tokens=" << cli.tts_max_tokens
                          << " tts_flow_steps=" << (cli.tts_flow_steps > 0 ? std::to_string(cli.tts_flow_steps) : std::string("default"))
                          << " tts_partials=" << cli.tts_partials
                          << " tts_play=" << cli.tts_play;
                if (!tts_flow_model_path.empty()) {
                    std::cout << " tts_flow_model=" << tts_flow_model_path;
                }
                if (!tts_hift_model_path.empty()) {
                    std::cout << " tts_hift_model=" << tts_hift_model_path;
                }
                if (!tts_voices_model_path.empty()) {
                    std::cout << " tts_voices_model=" << tts_voices_model_path;
                }
            }
            std::cout << "\n";
        }

        const auto handle_transcripts =
            [&output_mutex, &translator, &submit_tts](const std::vector<vox::asr::Transcript> & transcripts) {
            for (const vox::asr::Transcript & transcript : transcripts) {
                {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << "[" << transcript.chunk_index << "] "
                              << (transcript.is_final ? "asr final: " : "asr: ")
                              << transcript.text
                              << " (asr " << transcript.elapsed_ms << " ms)\n";
                    std::cout.flush();
                }
                if (translator) {
                    translator->submit(transcript);
                } else {
                    submit_tts(transcript.chunk_index, transcript.text, transcript.is_final);
                }
            }
        };

        const auto emit_final_transcript =
            [&handle_transcripts](const vox::asr::Transcript & transcript) {
            vox::asr::Transcript final_transcript = transcript;
            final_transcript.is_final = true;
            handle_transcripts(std::vector<vox::asr::Transcript>{final_transcript});
        };

        bool has_pending_final = false;
        int32_t silence_steps = 0;
        vox::asr::Transcript pending_final;

        auto last_audio_debug = std::chrono::steady_clock::now();
        bool capture_running = true;
        const auto should_continue_capture = [&microphone, &capture_running]() {
            capture_running = g_running.load() && microphone.poll_events();
            return capture_running;
        };

        while (should_continue_capture()) {
            std::vector<float> samples = microphone.read(common_config.step_ms, should_continue_capture);
            if (!capture_running || !g_running.load()) {
                break;
            }
            if (samples.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            apply_gain(samples, cli.input_gain);
            const AudioStats stats = measure_audio(samples);
            const bool is_silent =
                common_config.min_audio_rms > 0.0f &&
                stats.rms < static_cast<double>(common_config.min_audio_rms);

            if (cli.debug_audio) {
                const auto now = std::chrono::steady_clock::now();
                const auto elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_audio_debug).count();
                if (elapsed_ms >= 1000) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cerr << "audio: samples=" << samples.size()
                              << " rms=" << stats.rms
                              << " peak=" << stats.peak << "\n";
                    last_audio_debug = now;
                }
            }
            const auto asr_start = std::chrono::steady_clock::now();
            std::vector<vox::asr::Transcript> transcripts = push_asr_audio(samples);
            const int64_t asr_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::steady_clock::now() - asr_start)
                                               .count();
            for (vox::asr::Transcript & transcript : transcripts) {
                transcript.elapsed_ms = asr_elapsed_ms;
            }
            if (final_silence_steps > 0) {
                if (!transcripts.empty()) {
                    const bool has_final_transcript = std::any_of(
                        transcripts.begin(),
                        transcripts.end(),
                        [](const vox::asr::Transcript & transcript) {
                            return transcript.is_final;
                        });
                    if (has_final_transcript) {
                        has_pending_final = false;
                        silence_steps = 0;
                        handle_transcripts(transcripts);
                        continue;
                    }

                    pending_final = transcripts.back();
                    pending_final.is_final = false;
                    has_pending_final = true;
                    silence_steps = 0;
                    if (!cli.final_only) {
                        handle_transcripts(transcripts);
                    }
                    continue;
                }

                if (is_silent || common_config.min_audio_rms <= 0.0f) {
                    ++silence_steps;
                } else {
                    silence_steps = 0;
                }

                if (has_pending_final && silence_steps >= final_silence_steps) {
                    emit_final_transcript(pending_final);
                    has_pending_final = false;
                    silence_steps = 0;
                    continue;
                }
            }
            if (cli.debug_audio && transcripts.empty()) {
                std::lock_guard<std::mutex> lock(output_mutex);
                std::cerr << "asr: no transcript\n";
            }
            handle_transcripts(transcripts);
        }
        if (has_pending_final) {
            emit_final_transcript(pending_final);
        } else {
            const auto flush_start = std::chrono::steady_clock::now();
            std::vector<vox::asr::Transcript> flushed = flush_asr();
            const int64_t flush_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                 std::chrono::steady_clock::now() - flush_start)
                                                 .count();
            for (vox::asr::Transcript & transcript : flushed) {
                transcript.elapsed_ms = flush_elapsed_ms;
            }
            handle_transcripts(flushed);
        }
        if (translator) {
            translator->close();
        }
        if (tts) {
            tts->close();
        }
        if (player) {
            player->close();
        }
    } catch (const std::exception & error) {
        std::cerr << "vox failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
