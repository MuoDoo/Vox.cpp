#include "realtime_session.h"

#include "async_text_to_speech.h"
#include "async_transcript_translator.h"
#include "microphone_audio_source.h"
#include "sdl_audio_output.h"
#include "streaming_qwen_asr.h"
#include "streaming_whisper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vox::app {
namespace {

struct AudioStats {
    double rms = 0.0;
    float peak = 0.0f;
};

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

std::string resolve_path(const RealtimeSessionConfig & config, const std::string & path) {
    namespace fs = std::filesystem;
    if (path.empty()) {
        return {};
    }

    const fs::path requested(path);
    if (requested.is_absolute() || file_exists(path)) {
        return path;
    }
    return (fs::path(config.project_root) / requested).string();
}

std::string default_vad_path(const RealtimeSessionConfig & config) {
    const std::string local = resolve_path(config, "models/vad/ggml-silero-v6.2.0.bin");
    if (file_exists(local)) {
        return local;
    }
    const std::string bundled = resolve_path(config, "external/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin");
    return file_exists(bundled) ? bundled : std::string();
}

AudioStats measure_audio(const std::vector<float> & samples) {
    AudioStats stats;
    double sum_squares = 0.0;
    size_t finite_count = 0;
    for (float sample : samples) {
        if (!std::isfinite(sample)) {
            continue;
        }
        stats.peak = std::max(stats.peak, std::fabs(sample));
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

void require_file(const std::string & path, const std::string & label) {
    if (!file_exists(path)) {
        throw std::runtime_error("missing " + label + ": " + path);
    }
}

int32_t tts_sample_rate(RealtimeTtsEngine engine) {
    switch (engine) {
    case RealtimeTtsEngine::CosyVoice3:
        return vox::tts::kCosyVoice3SampleRate;
    case RealtimeTtsEngine::Kokoro:
        return vox::tts::kKokoroSampleRate;
    case RealtimeTtsEngine::Qwen3Tts:
        return vox::tts::kQwen3TtsSampleRate;
    }
    return vox::tts::kCosyVoice3SampleRate;
}

std::string default_qwen3_tts_backend(const RealtimeSessionConfig & config) {
    if (!config.tts_backend.empty()) {
        return config.tts_backend;
    }
    return config.tts_voice_model_path.empty() ? "qwen3-tts-customvoice" : "qwen3-tts";
}

} // namespace

RealtimeSession::RealtimeSession() = default;

RealtimeSession::~RealtimeSession() {
    stop();
}

bool RealtimeSession::start(RealtimeSessionConfig config) {
    if (running()) {
        return false;
    }
    if (worker_.joinable()) {
        worker_.join();
    }

    stop_requested_ = false;
    running_ = true;
    worker_ = std::thread([this, config = std::move(config)]() mutable {
        run(std::move(config));
    });
    return true;
}

void RealtimeSession::request_stop() {
    stop_requested_ = true;
}

void RealtimeSession::stop() {
    request_stop();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool RealtimeSession::running() const {
    return running_.load();
}

void RealtimeSession::run(RealtimeSessionConfig config) {
    const auto emit_status = [this](std::string message, bool running) {
        if (on_status) {
            on_status({std::move(message), running});
        }
    };
    const auto emit_error = [this](std::string message) {
        if (on_error) {
            on_error(std::move(message));
        }
    };
    const auto emit_text = [this](RealtimeTextEvent event) {
        if (on_text) {
            on_text(std::move(event));
        }
    };
    const auto emit_audio = [this](RealtimeAudioEvent event) {
        if (on_audio) {
            on_audio(event);
        }
    };

    try {
        emit_status("Loading models", true);

        const std::string asr_model_path = resolve_path(config, config.asr_model_path);
        require_file(asr_model_path, "ASR model");

        std::string asr_mmproj_path;
        if (config.asr_engine == RealtimeAsrEngine::Qwen3) {
            asr_mmproj_path = resolve_path(config, config.asr_mmproj_path);
            require_file(asr_mmproj_path, "Qwen3-ASR mmproj");
        }

        std::string vad_model_path;
        if (config.asr_engine == RealtimeAsrEngine::Qwen3 && config.use_vad) {
            vad_model_path = config.vad_model_path.empty()
                                 ? default_vad_path(config)
                                 : resolve_path(config, config.vad_model_path);
            if (!vad_model_path.empty()) {
                require_file(vad_model_path, "VAD model");
            }
        }

        const std::string translation_model_path = resolve_path(config, config.translation_model_path);
        if (!translation_model_path.empty()) {
            require_file(translation_model_path, "translation model");
        }

        const std::string tts_model_path = resolve_path(config, config.tts_model_path);
        require_file(tts_model_path, "TTS model");

        const std::string tts_flow_model_path = resolve_path(config, config.tts_flow_model_path);
        const std::string tts_hift_model_path = resolve_path(config, config.tts_hift_model_path);
        const std::string tts_voices_model_path = resolve_path(config, config.tts_voices_model_path);
        const std::string tts_codec_model_path = resolve_path(config, config.tts_codec_model_path);
        const std::string tts_voice_model_path = resolve_path(config, config.tts_voice_model_path);

        if (!tts_flow_model_path.empty()) {
            require_file(tts_flow_model_path, "TTS flow model");
        }
        if (!tts_hift_model_path.empty()) {
            require_file(tts_hift_model_path, "TTS HiFT model");
        }
        if (!tts_voices_model_path.empty()) {
            require_file(tts_voices_model_path, "TTS voices model");
        }
        if (!tts_codec_model_path.empty()) {
            require_file(tts_codec_model_path, "TTS codec model");
        }
        if (!tts_voice_model_path.empty()) {
            require_file(tts_voice_model_path, "TTS voice/reference model");
        }

        auto output = std::make_shared<SdlAudioOutput>();
        output->open(config.playback_device_id, tts_sample_rate(config.tts_engine), 1);
        emit_status("Opened virtual output: " + output->device_name(), true);

        std::unique_ptr<vox::pipeline::AsyncTextToSpeech> tts;
        auto tts_result_handler = [emit_text](vox::pipeline::TextToSpeechResult result) {
            if (!result.error.empty()) {
                emit_text({
                    result.request.chunk_index,
                    "tts error",
                    result.error,
                    result.request.is_final,
                    result.elapsed_ms,
                });
                return;
            }
            emit_text({
                result.request.chunk_index,
                result.request.is_final ? "tts final" : "tts",
                result.output_path,
                result.request.is_final,
                result.elapsed_ms,
            });
        };

        if (config.tts_engine == RealtimeTtsEngine::Kokoro) {
            vox::tts::KokoroTtsConfig tts_config;
            tts_config.model_path = tts_model_path;
            tts_config.voice_model_path = tts_voice_model_path;
            tts_config.language = config.tts_language.empty() ? config.target_language : config.tts_language;
            tts_config.output_dir = config.tts_output_dir;
            tts_config.threads = config.threads;
            tts_config.length_scale = config.tts_length_scale;
            tts_config.use_gpu = config.use_gpu;
            tts_config.flash_attention = config.flash_attention;
            tts_config.play_after_synthesis = false;

            auto synthesizer = std::make_shared<vox::tts::KokoroSynthesizer>(std::move(tts_config));
            tts = std::make_unique<vox::pipeline::AsyncTextToSpeech>(
                [synthesizer, output](vox::pipeline::TextToSpeechRequest request) {
                    const std::vector<float> pcm = synthesizer->synthesize_pcm(request.text);
                    output->enqueue_mono(pcm, vox::tts::kKokoroSampleRate);
                    return "queued to " + output->device_name();
                },
                std::move(tts_result_handler));
        } else if (config.tts_engine == RealtimeTtsEngine::Qwen3Tts) {
            vox::tts::Qwen3TtsConfig tts_config;
            tts_config.crispasr_path = config.tts_crispasr_path;
            tts_config.backend = default_qwen3_tts_backend(config);
            tts_config.model_path = tts_model_path;
            tts_config.codec_model_path = tts_codec_model_path;
            tts_config.voice_model_path = tts_voice_model_path;
            tts_config.ref_text = config.tts_ref_text;
            tts_config.instruct = config.tts_instruct;
            tts_config.language = config.tts_language.empty() ? config.target_language : config.tts_language;
            tts_config.voice = config.tts_voice.empty() ? "aiden" : config.tts_voice;
            tts_config.output_dir = config.tts_output_dir;
            tts_config.threads = config.threads;
            tts_config.max_codec_steps = config.tts_max_tokens;
            tts_config.temperature = config.tts_temperature;
            tts_config.seed = config.tts_seed;
            tts_config.use_gpu = config.use_gpu;
            tts_config.flash_attention = config.flash_attention;
            tts_config.play_after_synthesis = false;

            auto synthesizer = std::make_shared<vox::tts::Qwen3TtsSynthesizer>(std::move(tts_config));
            tts = std::make_unique<vox::pipeline::AsyncTextToSpeech>(
                [synthesizer, output](vox::pipeline::TextToSpeechRequest request) {
                    const std::vector<float> pcm = synthesizer->synthesize_pcm(request.text);
                    output->enqueue_mono(pcm, vox::tts::kQwen3TtsSampleRate);
                    return "queued to " + output->device_name();
                },
                std::move(tts_result_handler));
        } else {
            vox::tts::CosyVoice3TtsConfig tts_config;
            tts_config.model_path = tts_model_path;
            tts_config.flow_model_path = tts_flow_model_path;
            tts_config.hift_model_path = tts_hift_model_path;
            tts_config.voices_model_path = tts_voices_model_path;
            tts_config.voice = config.tts_voice.empty() ? "zero_shot" : config.tts_voice;
            tts_config.output_dir = config.tts_output_dir;
            tts_config.threads = config.threads;
            tts_config.max_tokens = config.tts_max_tokens;
            tts_config.temperature = config.tts_temperature;
            tts_config.seed = config.tts_seed;
            tts_config.use_gpu = config.use_gpu;
            tts_config.flash_attention = config.flash_attention;
            tts_config.play_after_synthesis = false;

            auto synthesizer = std::make_shared<vox::tts::CosyVoice3Synthesizer>(std::move(tts_config));
            tts = std::make_unique<vox::pipeline::AsyncTextToSpeech>(
                [synthesizer, output](vox::pipeline::TextToSpeechRequest request) {
                    const std::vector<float> pcm = synthesizer->synthesize_pcm(request.text);
                    output->enqueue_mono(pcm, vox::tts::kCosyVoice3SampleRate);
                    return "queued to " + output->device_name();
                },
                std::move(tts_result_handler));
        }

        const auto submit_tts = [&tts, &config](uint64_t chunk_index, const std::string & text, bool is_final) {
            if (!tts || text.empty()) {
                return;
            }
            if (!is_final && !config.speak_partials) {
                return;
            }
            tts->submit({chunk_index, text, is_final});
        };

        std::unique_ptr<vox::pipeline::AsyncTranscriptTranslator> translator;
        if (!translation_model_path.empty()) {
            vox::translate::LlamaTranslatorConfig translate_config;
            translate_config.model_path = translation_model_path;
            translate_config.source_language = config.language;
            translate_config.target_language = config.target_language;
            translate_config.thread_count = config.threads;
            translate_config.gpu_layers = config.use_gpu ? 999 : 0;

            translator = std::make_unique<vox::pipeline::AsyncTranscriptTranslator>(
                std::move(translate_config),
                [emit_text, submit_tts](vox::pipeline::TranslationResult result) {
                    if (!result.error.empty()) {
                        emit_text({
                            result.transcript.chunk_index,
                            "translation error",
                            result.error,
                            result.transcript.is_final,
                            result.elapsed_ms,
                        });
                        return;
                    }
                    emit_text({
                        result.transcript.chunk_index,
                        result.transcript.is_final ? "translation final" : "translation",
                        result.translation,
                        result.transcript.is_final,
                        result.elapsed_ms,
                    });
                    submit_tts(result.transcript.chunk_index, result.translation, result.transcript.is_final);
                });
        }

        std::unique_ptr<vox::asr::StreamingWhisper> whisper;
        std::unique_ptr<vox::asr::StreamingQwenAsr> qwen;
        if (config.asr_engine == RealtimeAsrEngine::Whisper) {
            vox::asr::StreamingWhisperConfig asr_config;
            asr_config.model_path = asr_model_path;
            asr_config.language = config.language;
            asr_config.threads = config.threads;
            asr_config.step_ms = config.step_ms;
            asr_config.window_ms = config.window_ms;
            asr_config.overlap_ms = config.overlap_ms;
            asr_config.min_audio_rms = config.min_audio_rms;
            asr_config.use_gpu = config.use_gpu;
            asr_config.flash_attention = config.flash_attention;
            whisper = std::make_unique<vox::asr::StreamingWhisper>(std::move(asr_config));
        } else {
            vox::asr::StreamingQwenAsrConfig asr_config;
            asr_config.model_path = asr_model_path;
            asr_config.mmproj_path = asr_mmproj_path;
            asr_config.language = config.language;
            asr_config.threads = config.threads;
            asr_config.step_ms = config.step_ms;
            asr_config.window_ms = config.window_ms;
            asr_config.overlap_ms = config.overlap_ms;
            asr_config.min_audio_rms = config.min_audio_rms;
            asr_config.use_gpu = config.use_gpu;
            asr_config.mmproj_use_gpu = config.use_gpu;
            asr_config.flash_attention = config.flash_attention;
            asr_config.vad_model_path = vad_model_path;
            asr_config.use_vad = config.use_vad && !vad_model_path.empty();
            asr_config.utterance_mode = asr_config.use_vad;
            qwen = std::make_unique<vox::asr::StreamingQwenAsr>(std::move(asr_config));
        }

        const auto push_asr_audio = [&whisper, &qwen, &config](const std::vector<float> & samples) {
            if (config.asr_engine == RealtimeAsrEngine::Whisper) {
                return whisper->push_audio(samples);
            }
            return qwen->push_audio(samples);
        };
        const auto flush_asr = [&whisper, &qwen, &config]() {
            if (config.asr_engine == RealtimeAsrEngine::Whisper) {
                return whisper->flush();
            }
            return qwen->flush();
        };
        const auto handle_transcripts =
            [emit_text, &translator, submit_tts](const std::vector<vox::asr::Transcript> & transcripts) {
            for (const vox::asr::Transcript & transcript : transcripts) {
                emit_text({
                    transcript.chunk_index,
                    transcript.is_final ? "asr final" : "asr",
                    transcript.text,
                    transcript.is_final,
                    transcript.elapsed_ms,
                });
                if (translator) {
                    translator->submit(transcript);
                } else {
                    submit_tts(transcript.chunk_index, transcript.text, transcript.is_final);
                }
            }
        };

        MicrophoneAudioSource microphone(config.capture_device_id, config.window_ms);
        microphone.start();
        emit_status("Listening", true);

        bool has_pending_final = false;
        int32_t silence_steps = 0;
        vox::asr::Transcript pending_final;

        while (!stop_requested_.load()) {
            std::vector<float> samples = microphone.read(config.step_ms, [this] {
                return !stop_requested_.load();
            });
            if (samples.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            apply_gain(samples, config.input_gain);
            const AudioStats stats = measure_audio(samples);
            emit_audio({stats.rms, stats.peak, output->queued_bytes()});

            const bool is_silent =
                config.min_audio_rms > 0.0f && stats.rms < static_cast<double>(config.min_audio_rms);

            const auto asr_start = std::chrono::steady_clock::now();
            std::vector<vox::asr::Transcript> transcripts = push_asr_audio(samples);
            const int64_t asr_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::steady_clock::now() - asr_start)
                                               .count();
            for (vox::asr::Transcript & transcript : transcripts) {
                transcript.elapsed_ms = asr_elapsed_ms;
            }

            if (config.final_silence_steps > 0) {
                if (!transcripts.empty()) {
                    const bool has_final = std::any_of(
                        transcripts.begin(),
                        transcripts.end(),
                        [](const vox::asr::Transcript & transcript) {
                            return transcript.is_final;
                        });
                    if (has_final) {
                        has_pending_final = false;
                        silence_steps = 0;
                        handle_transcripts(transcripts);
                        continue;
                    }

                    pending_final = transcripts.back();
                    pending_final.is_final = false;
                    has_pending_final = true;
                    silence_steps = 0;
                    if (config.speak_partials) {
                        handle_transcripts(transcripts);
                    } else {
                        for (const vox::asr::Transcript & transcript : transcripts) {
                            emit_text({
                                transcript.chunk_index,
                                "asr",
                                transcript.text,
                                false,
                                transcript.elapsed_ms,
                            });
                        }
                    }
                    continue;
                }

                silence_steps = is_silent || config.min_audio_rms <= 0.0f ? silence_steps + 1 : 0;
                if (has_pending_final && silence_steps >= config.final_silence_steps) {
                    pending_final.is_final = true;
                    handle_transcripts(std::vector<vox::asr::Transcript>{pending_final});
                    has_pending_final = false;
                    silence_steps = 0;
                    continue;
                }
            }

            handle_transcripts(transcripts);
        }

        if (has_pending_final) {
            pending_final.is_final = true;
            handle_transcripts(std::vector<vox::asr::Transcript>{pending_final});
        } else {
            std::vector<vox::asr::Transcript> flushed = flush_asr();
            for (vox::asr::Transcript & transcript : flushed) {
                transcript.is_final = true;
            }
            handle_transcripts(flushed);
        }

        if (translator) {
            translator->close();
        }
        if (tts) {
            tts->close();
        }
        running_ = false;
        emit_status("Stopped", false);
    } catch (const std::exception & error) {
        emit_error(error.what());
        running_ = false;
        emit_status("Stopped", false);
    } catch (...) {
        emit_error("unknown realtime session error");
        running_ = false;
        emit_status("Stopped", false);
    }
}

} // namespace vox::app
