#include "async_transcript_translator.h"
#include "microphone_audio_source.h"
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
    int32_t capture_device_id = -1;
    int32_t step_ms = 0;
    int32_t window_ms = 0;
    int32_t overlap_ms = -1;
    int32_t final_silence_steps = 0;
    float rms_threshold = -1.0f;
    float no_speech_threshold = -1.0f;
    float min_token_probability = 0.0f;
    float input_gain = 1.0f;
    bool show_help = false;
    bool disable_silence_gate = false;
    bool debug_audio = false;
    bool whisper_debug = false;
    bool disable_gpu = false;
    bool disable_flash_attention = false;
    bool final_only = false;
    std::vector<std::string> positional;
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
        } else if (arg == "--no-gpu") {
            options.disable_gpu = true;
        } else if (arg == "--no-flash-attn") {
            options.disable_flash_attention = true;
        } else if (arg == "--final-only") {
            options.final_only = true;
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
        } else if (arg == "--rms-threshold" || arg.rfind("--rms-threshold=", 0) == 0) {
            options.rms_threshold =
                parse_float(option_value(arg, "--rms-threshold", i, argc, argv), "--rms-threshold");
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
    if (options.rms_threshold < -1.0f) {
        throw std::runtime_error("rms threshold must be non-negative");
    }
    if (options.no_speech_threshold < -1.0f) {
        throw std::runtime_error("no-speech threshold must be non-negative");
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
    return options;
}

void print_usage(const char * program) {
    std::cout << "usage: " << program << " [options] [asr_model] [language] [translation_model] [target_language]\n"
              << "\n"
              << "options:\n"
              << "  -c, --capture ID    capture device index, or -1 for system default\n"
              << "      --step MS       audio step size in milliseconds\n"
              << "      --length MS     audio window size in milliseconds\n"
              << "      --keep MS       overlap to keep from the previous window in milliseconds\n"
              << "      --final-on-silence N\n"
              << "                      emit latest transcript as final after N silent windows\n"
              << "      --final-only    suppress partial ASR output; implies --final-on-silence 1\n"
              << "      --rms-threshold N\n"
              << "                      silence gate RMS threshold; 0 disables it\n"
              << "      --gain N        multiply microphone samples before ASR\n"
              << "      --min-token-p N minimum average token probability for emitted segments\n"
              << "      --no-speech-thold N\n"
              << "                      Whisper no-speech threshold; 1.0 disables most no-speech filtering\n"
              << "      --no-vad        disable the silence gate\n"
              << "      --no-gpu        disable Whisper GPU acceleration\n"
              << "      --no-flash-attn disable Whisper flash attention\n"
              << "      --debug-audio   print microphone sample count, RMS, and peak\n"
              << "      --whisper-debug print Whisper segment diagnostics\n"
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

        vox::asr::StreamingWhisperConfig asr_config;
        asr_config.model_path = resolve_model_path(
            !cli.positional.empty() ? cli.positional[0] : "models/ggml-base.bin");
        if (cli.positional.size() > 1) {
            asr_config.language = cli.positional[1];
        }
        if (cli.step_ms > 0) {
            asr_config.step_ms = cli.step_ms;
        }
        if (cli.window_ms > 0) {
            asr_config.window_ms = cli.window_ms;
        }
        if (cli.overlap_ms >= 0) {
            asr_config.overlap_ms = cli.overlap_ms;
        }
        if (cli.disable_silence_gate) {
            asr_config.min_audio_rms = 0.0f;
        } else if (cli.rms_threshold >= 0.0f) {
            asr_config.min_audio_rms = cli.rms_threshold;
        } else if (asr_config.language != "auto") {
            asr_config.min_audio_rms = 0.0f;
        }
        asr_config.no_speech_threshold =
            cli.no_speech_threshold >= 0.0f ? cli.no_speech_threshold : 0.95f;
        asr_config.min_token_probability = cli.min_token_probability;
        asr_config.debug = cli.whisper_debug;
        if (cli.disable_gpu) {
            asr_config.use_gpu = false;
        }
        if (cli.disable_flash_attention) {
            asr_config.flash_attention = false;
        }

        if (!file_exists(asr_config.model_path)) {
            std::cerr << "Missing model: " << asr_config.model_path << "\n"
                      << "Download: ./external/whisper.cpp/models/download-ggml-model.sh base models\n";
            return 1;
        }

        const std::string translation_model_path =
            cli.positional.size() > 2 ? resolve_model_path(cli.positional[2]) : std::string();
        if (!translation_model_path.empty() && !file_exists(translation_model_path)) {
            std::cerr << "Missing translation model: " << translation_model_path << "\n"
                      << "Download: scripts/download-hymt-gguf.sh\n";
            return 1;
        }

        std::mutex output_mutex;
        std::unique_ptr<vox::pipeline::AsyncTranscriptTranslator> translator;
        const int32_t final_silence_steps =
            cli.final_silence_steps > 0 ? cli.final_silence_steps : (cli.final_only ? 1 : 0);
        if (!translation_model_path.empty()) {
            vox::translate::LlamaTranslatorConfig translate_config;
            translate_config.model_path = translation_model_path;
            translate_config.source_language = asr_config.language;
            if (cli.positional.size() > 3) {
                translate_config.target_language = cli.positional[3];
            }

            translator = std::make_unique<vox::pipeline::AsyncTranscriptTranslator>(
                std::move(translate_config),
                [&output_mutex](vox::pipeline::TranslationResult result) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    if (!result.error.empty()) {
                        std::cerr << "[" << result.transcript.chunk_index
                                  << "] translation failed: " << result.error << "\n";
                        return;
                    }
                    std::cout << "[" << result.transcript.chunk_index << "] "
                              << (result.transcript.is_final ? "translation final: " : "translation: ")
                              << result.translation << "\n";
                    std::cout.flush();
                });
        }

        vox::asr::StreamingWhisper recognizer(asr_config);
        vox::app::MicrophoneAudioSource microphone(cli.capture_device_id, asr_config.window_ms);
        microphone.start();

        {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "vox listening: " << asr_config.model_path
                      << " language=" << asr_config.language
                      << " capture=" << cli.capture_device_id
                      << " step_ms=" << asr_config.step_ms
                      << " window_ms=" << asr_config.window_ms
                      << " keep_ms=" << asr_config.overlap_ms
                      << " rms_threshold=" << asr_config.min_audio_rms
                      << " no_speech_thold=" << asr_config.no_speech_threshold
                      << " min_token_p=" << asr_config.min_token_probability
                      << " final_on_silence=" << final_silence_steps
                      << " final_only=" << cli.final_only
                      << " gain=" << cli.input_gain
                      << " gpu=" << asr_config.use_gpu
                      << " flash_attn=" << asr_config.flash_attention;
            if (translator) {
                std::cout << " translation_model=" << translation_model_path;
            }
            std::cout << "\n";
        }

        const auto handle_transcripts =
            [&output_mutex, &translator](const std::vector<vox::asr::Transcript> & transcripts) {
            for (const vox::asr::Transcript & transcript : transcripts) {
                {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << "[" << transcript.chunk_index << "] "
                              << (transcript.is_final ? "asr final: " : "asr: ")
                              << transcript.text << "\n";
                    std::cout.flush();
                }
                if (translator) {
                    translator->submit(transcript);
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
        while (g_running.load() && microphone.poll_events()) {
            std::vector<float> samples = microphone.read(asr_config.step_ms);
            if (samples.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            apply_gain(samples, cli.input_gain);
            const AudioStats stats = measure_audio(samples);
            const bool is_silent =
                asr_config.min_audio_rms > 0.0f &&
                stats.rms < static_cast<double>(asr_config.min_audio_rms);

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
            std::vector<vox::asr::Transcript> transcripts = recognizer.push_audio(samples);
            if (final_silence_steps > 0) {
                if (!transcripts.empty()) {
                    pending_final = transcripts.back();
                    pending_final.is_final = false;
                    has_pending_final = true;
                    silence_steps = 0;
                    if (!cli.final_only) {
                        handle_transcripts(transcripts);
                    }
                    continue;
                }

                if (is_silent || asr_config.min_audio_rms <= 0.0f) {
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
            handle_transcripts(recognizer.flush());
        }
        if (translator) {
            translator->close();
        }
    } catch (const std::exception & error) {
        std::cerr << "vox failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
