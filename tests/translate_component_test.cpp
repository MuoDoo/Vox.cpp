#include "llama_translator.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    const std::string zh_source_prompt =
        vox::translate::make_hymt_prompt("今天天气很好。", "Chinese", "English");
    if (zh_source_prompt.find("Translate the following segment into English") == std::string::npos ||
        zh_source_prompt.find("今天天气很好。") == std::string::npos) {
        std::cerr << "unexpected zh source prompt: " << zh_source_prompt << "\n";
        return 1;
    }

    const std::string en_prompt =
        vox::translate::make_hymt_prompt("It is on the house.", "English", "Chinese");
    if (en_prompt.find("Translate the following segment into Chinese") == std::string::npos ||
        en_prompt.find("It is on the house.") == std::string::npos) {
        std::cerr << "unexpected non-zh prompt: " << en_prompt << "\n";
        return 1;
    }

    try {
        vox::translate::LlamaTranslator translator({});
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        if (message.find("missing llama.cpp model path") != std::string::npos) {
            return 0;
        }

        std::cerr << "unexpected error: " << message << "\n";
        return 1;
    } catch (const std::exception & error) {
        std::cerr << "unexpected exception type: " << error.what() << "\n";
        return 1;
    }

    std::cerr << "expected missing model path error\n";
    return 1;
}
