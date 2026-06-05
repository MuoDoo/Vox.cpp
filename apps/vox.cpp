#include "async_transcript_translator.h"
#include "microphone_audio_source.h"
#include "streaming_whisper.h"

#include <atomic>
#include <chrono>
#include <csignal>
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

} // namespace

int main(int argc, char ** argv) {
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);

    vox::asr::StreamingWhisperConfig asr_config;
    asr_config.model_path = resolve_model_path(argc > 1 ? argv[1] : "models/ggml-base.bin");
    if (argc > 2) {
        asr_config.language = argv[2];
    }

    if (!file_exists(asr_config.model_path)) {
        std::cerr << "Missing model: " << asr_config.model_path << "\n"
                  << "Download: ./external/whisper.cpp/models/download-ggml-model.sh base models\n";
        return 1;
    }

    const std::string translation_model_path =
        argc > 3 ? resolve_model_path(argv[3]) : std::string();
    if (!translation_model_path.empty() && !file_exists(translation_model_path)) {
        std::cerr << "Missing translation model: " << translation_model_path << "\n"
                  << "Download: scripts/download-hymt-gguf.sh\n";
        return 1;
    }

    try {
        std::mutex output_mutex;
        std::unique_ptr<vox::pipeline::AsyncTranscriptTranslator> translator;
        if (!translation_model_path.empty()) {
            vox::translate::LlamaTranslatorConfig translate_config;
            translate_config.model_path = translation_model_path;
            translate_config.source_language = asr_config.language;
            if (argc > 4) {
                translate_config.target_language = argv[4];
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
        vox::app::MicrophoneAudioSource microphone(-1, asr_config.window_ms);
        microphone.start();

        {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "vox listening: " << asr_config.model_path
                      << " language=" << asr_config.language;
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

        while (g_running.load() && microphone.poll_events()) {
            const std::vector<float> samples = microphone.read(asr_config.step_ms);
            if (samples.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            handle_transcripts(recognizer.push_audio(samples));
        }
        handle_transcripts(recognizer.flush());
        if (translator) {
            translator->close();
        }
    } catch (const std::exception & error) {
        std::cerr << "vox failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
