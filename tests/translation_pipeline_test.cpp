#include "async_transcript_translator.h"

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {

class BlockingTranslator {
public:
    std::string translate(const std::string & text) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!started_) {
            started_ = true;
            started_condition_.notify_one();
            release_condition_.wait(lock, [this] {
                return released_;
            });
        }
        return "translated: " + text;
    }

    void wait_until_started() {
        std::unique_lock<std::mutex> lock(mutex_);
        started_condition_.wait(lock, [this] {
            return started_;
        });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        release_condition_.notify_one();
    }

private:
    std::mutex mutex_;
    std::condition_variable started_condition_;
    std::condition_variable release_condition_;
    bool started_ = false;
    bool released_ = false;
};

bool expect_chunks(
    const std::vector<vox::pipeline::TranslationResult> & results,
    const std::vector<uint64_t> & expected_chunks) {
    if (results.size() != expected_chunks.size()) {
        std::cerr << "unexpected result count: " << results.size() << "\n";
        return false;
    }

    for (size_t index = 0; index < results.size(); ++index) {
        if (results[index].transcript.chunk_index != expected_chunks[index]) {
            std::cerr << "unexpected chunk at result " << index << ": "
                      << results[index].transcript.chunk_index << "\n";
            return false;
        }
        if (!results[index].error.empty()) {
            std::cerr << "unexpected translation error: " << results[index].error << "\n";
            return false;
        }
    }
    return true;
}

bool test_partial_coalescing() {
    BlockingTranslator blocking_translator;
    std::vector<vox::pipeline::TranslationResult> results;

    vox::pipeline::AsyncTranscriptTranslator translator(
        [&blocking_translator](const std::string & text) {
            return blocking_translator.translate(text);
        },
        [&results](vox::pipeline::TranslationResult result) {
            results.push_back(std::move(result));
        });

    translator.submit({0, "first", false});
    blocking_translator.wait_until_started();
    translator.submit({1, "stale", false});
    translator.submit({2, "latest", false});
    blocking_translator.release();
    translator.close();

    return expect_chunks(results, {0, 2});
}

bool test_final_discards_pending_partial() {
    BlockingTranslator blocking_translator;
    std::vector<vox::pipeline::TranslationResult> results;

    vox::pipeline::AsyncTranscriptTranslator translator(
        [&blocking_translator](const std::string & text) {
            return blocking_translator.translate(text);
        },
        [&results](vox::pipeline::TranslationResult result) {
            results.push_back(std::move(result));
        });

    translator.submit({0, "first", false});
    blocking_translator.wait_until_started();
    translator.submit({1, "stale", false});
    translator.submit({2, "final", true});
    blocking_translator.release();
    translator.close();

    return expect_chunks(results, {0, 2}) && results.back().transcript.is_final;
}

bool test_final_transcripts_are_not_coalesced() {
    BlockingTranslator blocking_translator;
    std::vector<vox::pipeline::TranslationResult> results;

    vox::pipeline::AsyncTranscriptTranslator translator(
        [&blocking_translator](const std::string & text) {
            return blocking_translator.translate(text);
        },
        [&results](vox::pipeline::TranslationResult result) {
            results.push_back(std::move(result));
        });

    translator.submit({0, "first", false});
    blocking_translator.wait_until_started();
    translator.submit({1, "final one", true});
    translator.submit({2, "final two", true});
    blocking_translator.release();
    translator.close();

    return expect_chunks(results, {0, 1, 2}) &&
           results[1].transcript.is_final &&
           results[2].transcript.is_final;
}

} // namespace

int main() {
    if (!test_partial_coalescing()) {
        return 1;
    }
    if (!test_final_discards_pending_partial()) {
        return 1;
    }
    if (!test_final_transcripts_are_not_coalesced()) {
        return 1;
    }
    return 0;
}
