#include "llama_translator.h"

#include <array>
#include <cctype>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#ifndef VOX_TEST_TRANSLATE_MODEL_PATH
#define VOX_TEST_TRANSLATE_MODEL_PATH "models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf"
#endif

namespace {

bool file_exists(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

bool contains_non_ascii(const std::string & text) {
    for (const unsigned char c : text) {
        if (c >= 0x80) {
            return true;
        }
    }
    return false;
}

bool contains_ascii_alpha(const std::string & text) {
    for (const unsigned char c : text) {
        if (std::isalpha(c) != 0) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string model_path =
        argc > 1 ? argv[1] : std::string(VOX_TEST_TRANSLATE_MODEL_PATH);

    if (!file_exists(model_path)) {
        std::cerr << "translation model missing, skipping: " << model_path << "\n";
        return 77;
    }

    try {
        vox::translate::LlamaTranslatorConfig config;
        config.model_path = model_path;
        config.source_language = "English";
        config.target_language = "Chinese";
        config.context_size = 1024;
        config.batch_size = 256;
        config.max_output_tokens = 96;
        config.gpu_layers = 0;

        vox::translate::LlamaTranslator translator(config);

        struct TestCase {
            std::string source_language;
            std::string target_language;
            std::string source_text;
            bool expect_non_ascii;
        };

        const std::array<TestCase, 2> cases = {{
            {"English", "Chinese", argc > 2 ? argv[2] : "It's on the house.", true},
            {"Chinese", "English", "今天天气很好。", false},
        }};

        for (const TestCase & test_case : cases) {
            const std::string translation = translator.translate(test_case.source_text,
                                                                 test_case.source_language,
                                                                 test_case.target_language);

            std::cout << test_case.source_language << " -> " << test_case.target_language << "\n";
            std::cout << "source: " << test_case.source_text << "\n";
            std::cout << "translation: " << translation << "\n";

            if (translation.empty()) {
                std::cerr << "empty translation\n";
                return 1;
            }
            if (translation == test_case.source_text) {
                std::cerr << "translation is identical to source\n";
                return 1;
            }
            if (test_case.expect_non_ascii && !contains_non_ascii(translation)) {
                std::cerr << "translation does not look like Chinese text\n";
                return 1;
            }
            if (!test_case.expect_non_ascii && !contains_ascii_alpha(translation)) {
                std::cerr << "translation does not look like English text\n";
                return 1;
            }
        }
    } catch (const std::exception & error) {
        std::cerr << "translation failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
