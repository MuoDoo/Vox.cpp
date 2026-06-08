#include "cosyvoice3_synthesizer.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<unsigned char> read_binary(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<unsigned char> bytes;
    char byte = 0;
    while (file.get(byte)) {
        bytes.push_back(static_cast<unsigned char>(byte));
    }
    return bytes;
}

uint16_t le_u16(const std::vector<unsigned char> & bytes, size_t offset) {
    return static_cast<uint16_t>(bytes[offset]) |
           static_cast<uint16_t>(bytes[offset + 1] << 8);
}

uint32_t le_u32(const std::vector<unsigned char> & bytes, size_t offset) {
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

bool has_ascii(const std::vector<unsigned char> & bytes, size_t offset, const std::string & text) {
    if (offset + text.size() > bytes.size()) {
        return false;
    }
    for (size_t i = 0; i < text.size(); ++i) {
        if (bytes[offset + i] != static_cast<unsigned char>(text[i])) {
            return false;
        }
    }
    return true;
}

bool test_wav_writer_header_and_samples() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / ("vox-tts-wav-test-" + std::to_string(now));
    const std::filesystem::path wav_path = temp_dir / "out" / "speech.wav";

    const float samples[] = {-2.0f, -1.0f, 0.0f, 0.5f, 2.0f};
    vox::tts::write_wav_mono_16(wav_path.string(), samples, 5, vox::tts::kCosyVoice3SampleRate);

    const std::vector<unsigned char> bytes = read_binary(wav_path);
    std::filesystem::remove_all(temp_dir);

    if (bytes.size() != 54) {
        std::cerr << "unexpected wav size: " << bytes.size() << "\n";
        return false;
    }
    if (!has_ascii(bytes, 0, "RIFF") || !has_ascii(bytes, 8, "WAVE") ||
        !has_ascii(bytes, 12, "fmt ") || !has_ascii(bytes, 36, "data")) {
        std::cerr << "missing wav chunk markers\n";
        return false;
    }
    if (le_u32(bytes, 4) != 46 || le_u32(bytes, 16) != 16 || le_u16(bytes, 20) != 1 ||
        le_u16(bytes, 22) != 1 || le_u32(bytes, 24) != 24000 || le_u32(bytes, 28) != 48000 ||
        le_u16(bytes, 32) != 2 || le_u16(bytes, 34) != 16 || le_u32(bytes, 40) != 10) {
        std::cerr << "unexpected wav header values\n";
        return false;
    }
    if (le_u16(bytes, 44) != 0x8001 || le_u16(bytes, 46) != 0x8001 ||
        le_u16(bytes, 48) != 0x0000 || le_u16(bytes, 50) != 0x3fff ||
        le_u16(bytes, 52) != 0x7fff) {
        std::cerr << "unexpected wav sample values\n";
        return false;
    }

    return true;
}

} // namespace

int main() {
    if (!test_wav_writer_header_and_samples()) {
        return 1;
    }
    return 0;
}
