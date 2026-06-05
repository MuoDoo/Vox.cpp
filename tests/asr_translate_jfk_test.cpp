#include "async_transcript_translator.h"
#include "streaming_whisper.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
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

bool contains_non_ascii(const std::string & text) {
    for (const unsigned char c : text) {
        if (c >= 0x80) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    const std::filesystem::path asr_model_path = VOX_TEST_MODEL_PATH;
    const std::filesystem::path audio_path = VOX_TEST_AUDIO_PATH;
    const std::filesystem::path translate_model_path = VOX_TEST_TRANSLATE_MODEL_PATH;

    if (!std::filesystem::exists(asr_model_path)) {
        std::cout << "SKIP: missing ASR model " << asr_model_path << "\n";
        return 77;
    }
    if (!std::filesystem::exists(translate_model_path)) {
        std::cout << "SKIP: missing translation model " << translate_model_path << "\n";
        return 77;
    }

    try {
        const WavPcm wav = read_wav_pcm16_mono_16khz(audio_path);

        vox::asr::StreamingWhisperConfig asr_config;
        asr_config.model_path = asr_model_path.string();
        asr_config.language = "en";
        asr_config.threads = 4;
        asr_config.use_gpu = false;
        asr_config.flash_attention = false;
        asr_config.step_ms = 2000;
        asr_config.window_ms = 6000;
        asr_config.overlap_ms = 300;

        vox::asr::StreamingWhisper recognizer(asr_config);
        for (size_t offset = 0; offset < wav.samples.size();) {
            constexpr size_t chunk_samples = 1379;
            const size_t count = std::min(chunk_samples, wav.samples.size() - offset);
            recognizer.push_audio(wav.samples.data() + offset, count);
            offset += count;
        }

        std::vector<vox::asr::Transcript> final_transcripts = recognizer.flush();
        if (final_transcripts.empty()) {
            std::cerr << "ASR produced no final transcript\n";
            return 1;
        }

        vox::asr::Transcript final_transcript;
        for (const vox::asr::Transcript & transcript : final_transcripts) {
            if (transcript.is_final && !transcript.text.empty()) {
                final_transcript = transcript;
            }
        }
        if (final_transcript.text.empty()) {
            std::cerr << "ASR final transcript was empty\n";
            return 1;
        }

        std::cout << "ASR final: " << final_transcript.text << "\n";

        vox::translate::LlamaTranslatorConfig translate_config;
        translate_config.model_path = translate_model_path.string();
        translate_config.source_language = "English";
        translate_config.target_language = "Chinese";
        translate_config.context_size = 1024;
        translate_config.batch_size = 256;
        translate_config.max_output_tokens = 96;
        translate_config.gpu_layers = 0;

        std::mutex mutex;
        std::vector<vox::pipeline::TranslationResult> results;

        vox::pipeline::AsyncTranscriptTranslator translator(
            std::move(translate_config),
            [&mutex, &results](vox::pipeline::TranslationResult result) {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    results.push_back(std::move(result));
                }
            });

        translator.submit(final_transcript);
        translator.close();

        if (results.empty()) {
            std::cerr << "translation pipeline produced no result\n";
            return 1;
        }

        const vox::pipeline::TranslationResult & result = results.front();
        if (!result.error.empty()) {
            std::cerr << "translation failed: " << result.error << "\n";
            return 1;
        }
        if (result.translation.empty()) {
            std::cerr << "translation was empty\n";
            return 1;
        }
        if (result.translation == final_transcript.text) {
            std::cerr << "translation is identical to ASR text\n";
            return 1;
        }
        if (!contains_non_ascii(result.translation)) {
            std::cerr << "translation does not look like Chinese text: " << result.translation << "\n";
            return 1;
        }

        std::cout << "Translation: " << result.translation << "\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }

    return 0;
}
