#include "qwen3_tts_synthesizer.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

namespace {

std::filesystem::path path_from_env(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return fallback;
}

std::string string_from_env(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return fallback;
}

bool env_bool(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "off" && text != "no";
}

bool file_has_wav_data(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    return input.is_open() && input.tellg() > 44;
}

} // namespace

int main() {
    if (!env_bool("VOX_TEST_QWEN_TTS", false)) {
        std::cout << "SKIP: set VOX_TEST_QWEN_TTS=1 to run Qwen3-TTS smoke test\n";
        return 77;
    }

    const std::filesystem::path model_path =
        path_from_env("VOX_TEST_QWEN_TTS_MODEL_PATH", VOX_TEST_QWEN_TTS_MODEL_PATH);
    const std::filesystem::path codec_path =
        path_from_env("VOX_TEST_QWEN_TTS_CODEC_PATH", VOX_TEST_QWEN_TTS_CODEC_PATH);
    const std::filesystem::path output_dir =
        path_from_env("VOX_TEST_QWEN_TTS_OUTPUT_DIR", VOX_TEST_QWEN_TTS_OUTPUT_DIR);

    if (!std::filesystem::exists(model_path)) {
        std::cout << "SKIP: missing Qwen3-TTS model " << model_path << "\n";
        return 77;
    }
    if (!std::filesystem::exists(codec_path)) {
        std::cout << "SKIP: missing Qwen3-TTS codec " << codec_path << "\n";
        return 77;
    }

    try {
        vox::tts::Qwen3TtsConfig config;
        config.model_path = model_path.string();
        config.codec_model_path = codec_path.string();
        config.output_dir = output_dir.string();
        config.voice = string_from_env("VOX_TEST_QWEN_TTS_VOICE", "vivian");
        config.language = string_from_env("VOX_TEST_QWEN_TTS_LANGUAGE", "English");
        config.threads = 8;
        config.temperature = 0.8f;
        config.seed = 42;
        config.use_gpu = env_bool("VOX_TEST_QWEN_TTS_USE_GPU", false);
        config.flash_attention = config.use_gpu;

        vox::tts::Qwen3TtsSynthesizer synthesizer(std::move(config));
        const std::string wav_path =
            synthesizer.synthesize(string_from_env("VOX_TEST_QWEN_TTS_TEXT", "Hello."), 0, true);
        if (!file_has_wav_data(wav_path)) {
            std::cerr << "Qwen3-TTS smoke output is missing or empty: " << wav_path << "\n";
            return 1;
        }

        std::cout << "Qwen3-TTS smoke output: " << wav_path << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Qwen3-TTS smoke test failed: " << error.what() << "\n";
        return 1;
    }
}
