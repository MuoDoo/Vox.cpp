#include "microphone_audio_source.h"
#include "streaming_whisper.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
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

    vox::asr::StreamingWhisperConfig config;
    config.model_path = resolve_model_path(argc > 1 ? argv[1] : "models/ggml-base.bin");
    if (argc > 2) {
        config.language = argv[2];
    }

    if (!file_exists(config.model_path)) {
        std::cerr << "Missing model: " << config.model_path << "\n"
                  << "Download: ./external/whisper.cpp/models/download-ggml-model.sh base models\n";
        return 1;
    }

    try {
        vox::asr::StreamingWhisper recognizer(config);
        vox::app::MicrophoneAudioSource microphone(-1, config.window_ms);
        microphone.start();

        std::cout << "vox listening: " << config.model_path
                  << " language=" << config.language << "\n";

        const auto print_transcripts = [](const std::vector<vox::asr::Transcript> & transcripts) {
            for (const vox::asr::Transcript & transcript : transcripts) {
                std::cout << "[" << transcript.chunk_index << "] " << transcript.text << "\n";
            }
            std::cout.flush();
        };

        while (g_running.load() && microphone.poll_events()) {
            const std::vector<float> samples = microphone.read(config.step_ms);
            if (samples.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            print_transcripts(recognizer.push_audio(samples));
        }
        print_transcripts(recognizer.flush());
    } catch (const std::exception & error) {
        std::cerr << "vox failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
