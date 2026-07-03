#include "async_transcript_translator.h"

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

AsyncTranscriptTranslator::TranslateFunction require_translate(
    AsyncTranscriptTranslator::TranslateFunction translate) {
    if (!translate) {
        throw std::runtime_error("missing transcript translate function");
    }
    return translate;
}

AsyncTranscriptTranslator::ResultHandler require_result_handler(
    AsyncTranscriptTranslator::ResultHandler result_handler) {
    if (!result_handler) {
        throw std::runtime_error("missing translation result handler");
    }
    return result_handler;
}

} // namespace

class AsyncTranscriptTranslator::Impl {
public:
    Impl(TranslateFunction translate, ResultHandler result_handler)
        : translate_(require_translate(std::move(translate))),
          result_handler_(require_result_handler(std::move(result_handler))),
          worker_([this] { run(); }) {
    }

    ~Impl() {
        close();
    }

    void submit(vox::asr::Transcript transcript) {
        if (transcript.text.empty()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                throw std::runtime_error("cannot submit transcript after translator is closed");
            }

            if (transcript.is_final) {
                pending_.erase(
                    std::remove_if(
                        pending_.begin(),
                        pending_.end(),
                        [](const vox::asr::Transcript & pending) {
                            return !pending.is_final;
                        }),
                    pending_.end());
            } else {
                while (!pending_.empty() && !pending_.back().is_final) {
                    pending_.pop_back();
                }
            }
            pending_.push_back(std::move(transcript));
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
            vox::asr::Transcript transcript;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] {
                    return closed_ || !pending_.empty();
                });
                if (pending_.empty()) {
                    return;
                }

                transcript = std::move(pending_.front());
                pending_.pop_front();
            }

            TranslationResult result;
            result.transcript = std::move(transcript);
            const auto start = std::chrono::steady_clock::now();
            try {
                result.translation = translate_(result.transcript.text);
            } catch (const std::exception & error) {
                result.error = error.what();
            } catch (...) {
                result.error = "unknown translation error";
            }
            result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();

            try {
                result_handler_(std::move(result));
            } catch (...) {
                // Result display must not terminate the translation worker.
            }
        }
    }

    TranslateFunction translate_;
    ResultHandler result_handler_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<vox::asr::Transcript> pending_;
    bool closed_ = false;
    std::thread worker_;
};

AsyncTranscriptTranslator::AsyncTranscriptTranslator(
    vox::translate::LlamaTranslatorConfig config,
    ResultHandler result_handler) {
    auto translator = std::make_shared<vox::translate::LlamaTranslator>(std::move(config));
    impl_ = std::make_unique<Impl>(
        [translator = std::move(translator)](const std::string & text) {
            return translator->translate(text);
        },
        std::move(result_handler));
}

AsyncTranscriptTranslator::AsyncTranscriptTranslator(
    TranslateFunction translate,
    ResultHandler result_handler)
    : impl_(std::make_unique<Impl>(std::move(translate), std::move(result_handler))) {
}

AsyncTranscriptTranslator::~AsyncTranscriptTranslator() = default;

void AsyncTranscriptTranslator::submit(vox::asr::Transcript transcript) {
    impl_->submit(std::move(transcript));
}

void AsyncTranscriptTranslator::close() {
    impl_->close();
}

} // namespace vox::pipeline
