#pragma once

#include "cosyvoice3_synthesizer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace vox::pipeline {

struct TextToSpeechRequest {
    uint64_t chunk_index = 0;
    std::string text;
    bool is_final = false;
};

struct TextToSpeechResult {
    TextToSpeechRequest request;
    std::string output_path;
    std::string error;
    int64_t elapsed_ms = 0;
};

class AsyncTextToSpeech {
public:
    using SynthesizeFunction = std::function<std::string(TextToSpeechRequest)>;
    using ResultHandler = std::function<void(TextToSpeechResult)>;

    AsyncTextToSpeech(vox::tts::CosyVoice3TtsConfig config, ResultHandler result_handler);
    AsyncTextToSpeech(SynthesizeFunction synthesize, ResultHandler result_handler);
    ~AsyncTextToSpeech();

    AsyncTextToSpeech(const AsyncTextToSpeech &) = delete;
    AsyncTextToSpeech & operator=(const AsyncTextToSpeech &) = delete;

    // Partial requests are coalesced when synthesis is slower than input.
    // Final requests discard pending partials and are always processed.
    void submit(TextToSpeechRequest request);

    // Stops accepting work, processes queued requests, and joins the worker.
    void close();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vox::pipeline
