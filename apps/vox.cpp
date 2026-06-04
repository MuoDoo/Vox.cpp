#include "whisper_realtime.h"

#include <atomic>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

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

    vox::asr::RealtimeWhisperConfig config;
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
        vox::asr::RealtimeWhisper recognizer(config);

        std::cout << "vox listening: " << config.model_path
                  << " language=" << config.language << "\n";

        recognizer.run(
            [](const vox::asr::Transcript & transcript) {
                std::cout << "[" << transcript.chunk_index << "] " << transcript.text << "\n";
                std::cout.flush();
                return true;
            },
            [] {
                return g_running.load();
            });
    } catch (const std::exception & error) {
        std::cerr << "vox failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
