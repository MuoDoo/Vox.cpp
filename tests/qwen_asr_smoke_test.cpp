#include "asr_types.h"
#include "qwen_asr_model.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

float pcm24_to_float(unsigned char b0, unsigned char b1, unsigned char b2) {
    int32_t value =
        static_cast<int32_t>(b0) |
        (static_cast<int32_t>(b1) << 8) |
        (static_cast<int32_t>(b2) << 16);
    if ((value & 0x00800000) != 0) {
        value |= static_cast<int32_t>(0xff000000);
    }
    return static_cast<float>(value) / 8388608.0f;
}

std::vector<float> downsample_to_16khz(std::vector<float> samples, uint32_t sample_rate) {
    if (sample_rate == static_cast<uint32_t>(vox::asr::kAsrSampleRate)) {
        return samples;
    }
    if (sample_rate % static_cast<uint32_t>(vox::asr::kAsrSampleRate) != 0) {
        throw std::runtime_error("wav sample rate must be 16kHz or an integer multiple of 16kHz");
    }

    const size_t factor = sample_rate / static_cast<uint32_t>(vox::asr::kAsrSampleRate);
    std::vector<float> downsampled;
    downsampled.reserve(samples.size() / factor);
    for (size_t offset = 0; offset + factor <= samples.size(); offset += factor) {
        double sum = 0.0;
        for (size_t i = 0; i < factor; ++i) {
            sum += static_cast<double>(samples[offset + i]);
        }
        downsampled.push_back(static_cast<float>(sum / static_cast<double>(factor)));
    }
    return downsampled;
}

WavPcm read_wav_mono_as_16khz(const std::filesystem::path & path) {
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
    std::vector<float> raw_samples;

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
            if (bits_per_sample == 16) {
                const uint32_t sample_count = chunk_size / sizeof(int16_t);
                std::vector<int16_t> pcm16(sample_count);
                input.read(reinterpret_cast<char *>(pcm16.data()), chunk_size);
                raw_samples.reserve(raw_samples.size() + pcm16.size());
                for (const int16_t sample : pcm16) {
                    raw_samples.push_back(static_cast<float>(sample) / 32768.0f);
                }
            } else if (bits_per_sample == 24) {
                const uint32_t sample_count = chunk_size / 3;
                raw_samples.reserve(raw_samples.size() + sample_count);
                for (uint32_t i = 0; i < sample_count; ++i) {
                    unsigned char bytes[3] = {};
                    input.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
                    raw_samples.push_back(pcm24_to_float(bytes[0], bytes[1], bytes[2]));
                }
            } else {
                throw std::runtime_error("only PCM16 and PCM24 wav fixtures are supported");
            }
            if (chunk_size % 2) {
                input.seekg(1, std::ios::cur);
            }
        } else {
            skip_bytes(input, chunk_size);
        }
    }

    if (audio_format != 1 || channels != 1 || raw_samples.empty()) {
        throw std::runtime_error("expected PCM mono wav fixture");
    }

    WavPcm wav;
    wav.sample_rate = vox::asr::kAsrSampleRate;
    wav.channels = static_cast<int>(channels);
    wav.samples = downsample_to_16khz(std::move(raw_samples), sample_rate);
    return wav;
}

std::filesystem::path path_from_env(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return fallback;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool env_bool(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    const std::string text = lower(value);
    return text != "0" && text != "false" && text != "off" && text != "no";
}

} // namespace

int main() {
    const std::filesystem::path model_path =
        path_from_env("VOX_TEST_QWEN_MODEL_PATH", VOX_TEST_QWEN_MODEL_PATH);
    const std::filesystem::path mmproj_path =
        path_from_env("VOX_TEST_QWEN_MMPROJ_PATH", VOX_TEST_QWEN_MMPROJ_PATH);
    const std::filesystem::path audio_path =
        path_from_env("VOX_TEST_QWEN_AUDIO_PATH", VOX_TEST_QWEN_AUDIO_PATH);

    if (!std::filesystem::exists(model_path)) {
        std::cout << "SKIP: missing Qwen model " << model_path << "\n";
        return 77;
    }
    if (!std::filesystem::exists(mmproj_path)) {
        std::cout << "SKIP: missing Qwen mmproj " << mmproj_path << "\n";
        return 77;
    }
    if (!std::filesystem::exists(audio_path)) {
        std::cout << "SKIP: missing Qwen smoke audio " << audio_path << "\n";
        return 77;
    }

    try {
        const WavPcm wav = read_wav_mono_as_16khz(audio_path);

        vox::asr::QwenAsrModelConfig config;
        config.model_path = model_path.string();
        config.mmproj_path = mmproj_path.string();
        config.language = "en";
        config.threads = 4;
        config.max_output_tokens = 256;
        config.use_gpu = env_bool("VOX_TEST_QWEN_USE_GPU", false);
        config.mmproj_use_gpu = config.use_gpu;
        config.flash_attention = config.use_gpu;

        vox::asr::QwenAsrModel model(config);

        const auto started_at = std::chrono::steady_clock::now();
        const std::string transcript = model.transcribe(wav.samples);
        const auto finished_at = std::chrono::steady_clock::now();

        const double elapsed_seconds =
            std::chrono::duration<double>(finished_at - started_at).count();
        const double audio_seconds =
            static_cast<double>(wav.samples.size()) / static_cast<double>(vox::asr::kAsrSampleRate);
        const double real_time_factor = elapsed_seconds / std::max(audio_seconds, 0.001);

        std::cout << "qwen transcript: " << transcript << "\n";
        std::cout << "qwen perf: audio_s=" << audio_seconds
                  << " elapsed_s=" << elapsed_seconds
                  << " rtf=" << real_time_factor << "\n";

        const std::string normalized = lower(transcript);
        if (transcript.empty()) {
            std::cerr << "Qwen ASR produced an empty transcript\n";
            return 1;
        }
        if (normalized.find("<asr_text>") != std::string::npos ||
            normalized.rfind("language ", 0) == 0) {
            std::cerr << "Qwen ASR output parser leaked protocol text\n";
            return 1;
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    return 0;
}
