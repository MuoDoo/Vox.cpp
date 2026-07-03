#include "async_text_to_speech.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace vox::pipeline {
namespace {

AsyncTextToSpeech::SynthesizeFunction require_synthesize(
    AsyncTextToSpeech::SynthesizeFunction synthesize) {
    if (!synthesize) {
        throw std::runtime_error("missing text-to-speech synthesize function");
    }
    return synthesize;
}

AsyncTextToSpeech::ResultHandler require_result_handler(
    AsyncTextToSpeech::ResultHandler result_handler) {
    if (!result_handler) {
        throw std::runtime_error("missing text-to-speech result handler");
    }
    return result_handler;
}

} // namespace

class AsyncTextToSpeech::Impl {
public:
    Impl(SynthesizeFunction synthesize, ResultHandler result_handler)
        : synthesize_(require_synthesize(std::move(synthesize))),
          result_handler_(require_result_handler(std::move(result_handler))),
          worker_([this] { run(); }) {
    }

    ~Impl() {
        close();
    }

    void submit(TextToSpeechRequest request) {
        if (request.text.empty()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                throw std::runtime_error("cannot submit text-to-speech request after synthesizer is closed");
            }

            if (request.is_final) {
                pending_.erase(
                    std::remove_if(
                        pending_.begin(),
                        pending_.end(),
                        [](const TextToSpeechRequest & pending) {
                            return !pending.is_final;
                        }),
                    pending_.end());
            } else {
                while (!pending_.empty() && !pending_.back().is_final) {
                    pending_.pop_back();
                }
            }
            pending_.push_back(std::move(request));
        }
        ready_.notify_one();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
        }
        ready_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    void run() {
        while (true) {
            TextToSpeechRequest request;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] {
                    return closed_ || !pending_.empty();
                });
                if (pending_.empty()) {
                    return;
                }

                request = std::move(pending_.front());
                pending_.pop_front();
            }

            TextToSpeechResult result;
            result.request = std::move(request);
            const auto start = std::chrono::steady_clock::now();
            try {
                result.output_path = synthesize_(result.request);
            } catch (const std::exception & error) {
                result.error = error.what();
            } catch (...) {
                result.error = "unknown text-to-speech error";
            }
            result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();

            try {
                result_handler_(std::move(result));
            } catch (...) {
                // Result display must not terminate the synthesis worker.
            }
        }
    }

    SynthesizeFunction synthesize_;
    ResultHandler result_handler_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<TextToSpeechRequest> pending_;
    bool closed_ = false;
    std::thread worker_;
};

AsyncTextToSpeech::AsyncTextToSpeech(
    vox::tts::CosyVoice3TtsConfig config,
    ResultHandler result_handler) {
    auto synthesizer = std::make_shared<vox::tts::CosyVoice3Synthesizer>(std::move(config));
    impl_ = std::make_unique<Impl>(
        [synthesizer = std::move(synthesizer)](TextToSpeechRequest request) {
            return synthesizer->synthesize(request.text, request.chunk_index, request.is_final);
        },
        std::move(result_handler));
}

AsyncTextToSpeech::AsyncTextToSpeech(
    SynthesizeFunction synthesize,
    ResultHandler result_handler)
    : impl_(std::make_unique<Impl>(std::move(synthesize), std::move(result_handler))) {
}

AsyncTextToSpeech::~AsyncTextToSpeech() = default;

void AsyncTextToSpeech::submit(TextToSpeechRequest request) {
    impl_->submit(std::move(request));
}

void AsyncTextToSpeech::close() {
    impl_->close();
}

} // namespace vox::pipeline
