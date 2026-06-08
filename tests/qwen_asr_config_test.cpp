#include "qwen_asr_model.h"

#include <iostream>
#include <string>

namespace {

bool expect_language(const std::string & input, const std::string & expected) {
    const std::string actual = vox::asr::normalize_qwen_asr_language(input);
    if (actual != expected) {
        std::cerr << "normalize_qwen_asr_language(" << input << ") = " << actual
                  << ", expected " << expected << "\n";
        return false;
    }
    return true;
}

bool expect_parse(
    const std::string & input,
    const std::string & forced_language,
    const std::string & expected) {
    const std::string actual = vox::asr::parse_qwen_asr_output(input, forced_language);
    if (actual != expected) {
        std::cerr << "parse_qwen_asr_output(" << input << ") = " << actual
                  << ", expected " << expected << "\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    bool ok = true;
    ok = expect_language("auto", "") && ok;
    ok = expect_language("en", "English") && ok;
    ok = expect_language("zh", "Chinese") && ok;
    ok = expect_language("yue", "Cantonese") && ok;
    ok = expect_language("Japanese", "Japanese") && ok;
    ok = expect_language("  german  ", "German") && ok;
    ok = expect_parse("language English<asr_text>Hello world", "", "Hello world") && ok;
    ok = expect_parse("language None<asr_text>", "", "") && ok;
    ok = expect_parse("language Chinese\n你好", "", "你好") && ok;
    ok = expect_parse("<|im_start|>assistant\nlanguage English<asr_text>Hello<|im_end|>", "", "Hello") && ok;
    ok = expect_parse("plain transcript", "English", "plain transcript") && ok;
    ok = expect_parse("language English<asr_text>Hello", "English", "Hello") && ok;
    return ok ? 0 : 1;
}
