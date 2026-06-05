#pragma once

#include "llama_translator.h"
#include "streaming_whisper.h"

#include <functional>
#include <memory>
#include <string>

namespace vox::pipeline {

struct TranslationResult {
    vox::asr::Transcript transcript;
    std::string translation;
    std::string error;
};

class AsyncTranscriptTranslator {
public:
    using TranslateFunction = std::function<std::string(const std::string &)>;
    using ResultHandler = std::function<void(TranslationResult)>;

    AsyncTranscriptTranslator(vox::translate::LlamaTranslatorConfig config, ResultHandler result_handler);
    AsyncTranscriptTranslator(TranslateFunction translate, ResultHandler result_handler);
    ~AsyncTranscriptTranslator();

    AsyncTranscriptTranslator(const AsyncTranscriptTranslator &) = delete;
    AsyncTranscriptTranslator & operator=(const AsyncTranscriptTranslator &) = delete;

    // Partial transcripts are coalesced when translation is slower than ASR.
    // Final transcripts discard pending partial updates and are always processed.
    void submit(vox::asr::Transcript transcript);

    // Stops accepting work, processes queued transcripts, and joins the worker.
    void close();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vox::pipeline
