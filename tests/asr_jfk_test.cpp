#include "whisper_model.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct WavPcm {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

uint16_t read_u16_le(std::istream & input) {
    unsigned char bytes[2] = {};
    input.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
}

uint32_t read_u32_le(std::istream & input) {
    unsigned char bytes[4] = {};
    input.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    return static_cast<uint32_t>(
        static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24));
}

std::string read_id(std::istream & input) {
    char id[4] = {};
    input.read(id, sizeof(id));
    return std::string(id, sizeof(id));
}

void skip_bytes(std::istream & input, uint32_t bytes) {
    input.seekg(bytes + (bytes % 2), std::ios::cur);
}

WavPcm read_wav_pcm16_mono_16khz(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open wav: " + path.string());
    }

    if (read_id(input) != "RIFF") {
        throw std::runtime_error("wav is missing RIFF header");
    }
    read_u32_le(input);
    if (read_id(input) != "WAVE") {
        throw std::runtime_error("wav is missing WAVE header");
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<int16_t> pcm16;

    while (input.good()) {
        const std::string chunk_id = read_id(input);
        if (chunk_id.size() != 4 || !input.good()) {
            break;
        }

        const uint32_t chunk_size = read_u32_le(input);
        if (chunk_id == "fmt ") {
            audio_format = read_u16_le(input);
            channels = read_u16_le(input);
            sample_rate = read_u32_le(input);
            read_u32_le(input);
            read_u16_le(input);
            bits_per_sample = read_u16_le(input);
            if (chunk_size > 16) {
                skip_bytes(input, chunk_size - 16);
            }
        } else if (chunk_id == "data") {
            if (bits_per_sample != 16) {
                throw std::runtime_error("only PCM16 wav fixtures are supported");
            }

            const uint32_t sample_count = chunk_size / sizeof(int16_t);
            pcm16.resize(sample_count);
            input.read(reinterpret_cast<char *>(pcm16.data()), chunk_size);
            if (chunk_size % 2) {
                input.seekg(1, std::ios::cur);
            }
        } else {
            skip_bytes(input, chunk_size);
        }
    }

    if (audio_format != 1 || channels != 1 || sample_rate != 16000 || bits_per_sample != 16) {
        throw std::runtime_error("expected PCM16 mono 16kHz wav fixture");
    }

    WavPcm wav;
    wav.sample_rate = static_cast<int>(sample_rate);
    wav.channels = static_cast<int>(channels);
    wav.samples.reserve(pcm16.size());
    for (const int16_t sample : pcm16) {
        wav.samples.push_back(static_cast<float>(sample) / 32768.0f);
    }
    return wav;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool contains(const std::string & text, const std::string & needle) {
    return text.find(needle) != std::string::npos;
}

} // namespace

int main() {
    const std::filesystem::path model_path = VOX_TEST_MODEL_PATH;
    const std::filesystem::path audio_path = VOX_TEST_AUDIO_PATH;

    if (!std::filesystem::exists(model_path)) {
        std::cout << "SKIP: missing model " << model_path << "\n";
        return 77;
    }

    try {
        const WavPcm wav = read_wav_pcm16_mono_16khz(audio_path);

        vox::asr::WhisperModelConfig config;
        config.model_path = model_path.string();
        config.language = "en";
        config.threads = 4;
        config.use_gpu = false;
        config.flash_attention = false;

        vox::asr::WhisperModel model(config);
        const std::string transcript = model.transcribe(wav.samples);
        const std::string normalized = lower(transcript);

        std::cout << transcript << "\n";

        if (!contains(normalized, "ask not") ||
            !contains(normalized, "country") ||
            !contains(normalized, "for you")) {
            std::cerr << "unexpected JFK transcript: " << transcript << "\n";
            return 1;
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    return 0;
}
