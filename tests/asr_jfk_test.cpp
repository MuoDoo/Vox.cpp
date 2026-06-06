#include "streaming_whisper.h"

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

        vox::asr::StreamingWhisperConfig config;
        config.model_path = model_path.string();
        config.language = "en";
        config.threads = 4;
        config.use_gpu = false;
        config.flash_attention = false;
        config.step_ms = 2000;
        config.window_ms = 6000;
        config.overlap_ms = 300;

        vox::asr::StreamingWhisper recognizer(config);
        std::vector<vox::asr::Transcript> transcripts;

        constexpr size_t chunk_samples = 1379;
        for (size_t offset = 0; offset < wav.samples.size(); offset += chunk_samples) {
            const size_t count = std::min(chunk_samples, wav.samples.size() - offset);
            std::vector<vox::asr::Transcript> chunk_transcripts =
                recognizer.push_audio(wav.samples.data() + offset, count);
            transcripts.insert(
                transcripts.end(),
                chunk_transcripts.begin(),
                chunk_transcripts.end());
        }
        std::vector<vox::asr::Transcript> final_transcripts = recognizer.flush();
        transcripts.insert(
            transcripts.end(),
            final_transcripts.begin(),
            final_transcripts.end());

        std::string transcript;
        bool saw_final = false;
        for (const vox::asr::Transcript & result : transcripts) {
            std::cout << "[" << result.chunk_index << "] "
                      << (result.is_final ? "final: " : "partial: ")
                      << result.text << "\n";
            transcript += " " + result.text;
            saw_final = saw_final || result.is_final;
        }
        const std::string normalized = lower(transcript);

        if (!contains(normalized, "ask not") ||
            !contains(normalized, "country") ||
            !contains(normalized, "for you")) {
            std::cerr << "unexpected JFK transcript: " << transcript << "\n";
            return 1;
        }
        if (!saw_final) {
            std::cerr << "streaming ASR did not produce a final transcript\n";
            return 1;
        }
        if (!recognizer.flush().empty()) {
            std::cerr << "repeated flush produced duplicate transcripts\n";
            return 1;
        }

        vox::asr::StreamingWhisperConfig silence_config;
        silence_config.model_path = model_path.string();
        silence_config.language = "auto";
        silence_config.threads = 4;
        silence_config.use_gpu = false;
        silence_config.flash_attention = false;
        silence_config.step_ms = 1000;
        silence_config.window_ms = 1000;
        silence_config.overlap_ms = 0;

        vox::asr::StreamingWhisper silence_recognizer(silence_config);
        const std::vector<float> silence_samples(2 * vox::asr::kWhisperSampleRate, 0.0f);
        std::vector<vox::asr::Transcript> silence_transcripts =
            silence_recognizer.push_audio(silence_samples);
        const std::vector<vox::asr::Transcript> silence_final =
            silence_recognizer.flush();
        silence_transcripts.insert(
            silence_transcripts.end(),
            silence_final.begin(),
            silence_final.end());

        if (!silence_transcripts.empty()) {
            std::cerr << "silence produced transcripts\n";
            return 1;
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    return 0;
}
