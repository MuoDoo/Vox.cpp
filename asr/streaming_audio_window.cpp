#include "streaming_audio_window.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vox::asr {

namespace {

int32_t clamp_positive(int32_t value, int32_t default_value) {
    return value > 0 ? value : default_value;
}

} // namespace

StreamingAudioWindow::StreamingAudioWindow(
    StreamingAudioWindowConfig config,
    TranscribeFunction transcribe)
    : config_(std::move(config)),
      transcribe_(std::move(transcribe)) {
    if (!transcribe_) {
        throw std::runtime_error("missing ASR transcribe function");
    }
    normalize_config();
}

std::vector<Transcript> StreamingAudioWindow::push_audio(const float * samples, size_t sample_count) {
    if (sample_count == 0) {
        return {};
    }
    if (samples == nullptr) {
        throw std::runtime_error("audio samples pointer is null");
    }

    pending_audio_.reserve(pending_audio_.size() + sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        pending_audio_.push_back(std::isfinite(samples[i]) ? samples[i] : 0.0f);
    }
    has_unflushed_audio_ = true;

    std::vector<Transcript> transcripts;
    const size_t samples_per_step = to_samples(config_.step_ms);
    while (pending_audio_.size() >= samples_per_step) {
        std::vector<float> step(
            pending_audio_.begin(),
            pending_audio_.begin() + static_cast<std::ptrdiff_t>(samples_per_step));
        pending_audio_.erase(
            pending_audio_.begin(),
            pending_audio_.begin() + static_cast<std::ptrdiff_t>(samples_per_step));

        append_transcript(step, false, transcripts);
    }
    return transcripts;
}

std::vector<Transcript> StreamingAudioWindow::push_audio(const std::vector<float> & samples) {
    return push_audio(samples.data(), samples.size());
}

std::vector<Transcript> StreamingAudioWindow::flush() {
    std::vector<Transcript> transcripts;
    if (!has_unflushed_audio_) {
        return transcripts;
    }

    if (!pending_audio_.empty()) {
        std::vector<float> tail;
        tail.swap(pending_audio_);
        append_transcript(tail, true, transcripts);
    } else if (!window_audio_.empty() && has_audible_signal(window_audio_)) {
        const std::string text = transcribe_(window_audio_);
        if (!text.empty()) {
            transcripts.push_back(Transcript{chunk_index_, text, true});
        }
        ++chunk_index_;
    }
    has_unflushed_audio_ = false;
    return transcripts;
}

void StreamingAudioWindow::reset() {
    pending_audio_.clear();
    window_audio_.clear();
    chunk_index_ = 0;
    has_unflushed_audio_ = false;
}

bool StreamingAudioWindow::has_audible_signal(const std::vector<float> & samples) const {
    if (config_.min_audio_rms <= 0.0f) {
        return true;
    }

    double sum_squares = 0.0;
    size_t sample_count = 0;
    for (const float sample : samples) {
        if (!std::isfinite(sample)) {
            continue;
        }
        sum_squares += static_cast<double>(sample) * static_cast<double>(sample);
        ++sample_count;
    }

    if (sample_count == 0) {
        return false;
    }

    const double rms = std::sqrt(sum_squares / static_cast<double>(sample_count));
    return rms >= static_cast<double>(config_.min_audio_rms);
}

size_t StreamingAudioWindow::to_samples(int32_t milliseconds) const {
    return static_cast<size_t>(milliseconds) * static_cast<size_t>(config_.sample_rate) / 1000;
}

void StreamingAudioWindow::append_transcript(
    const std::vector<float> & new_audio,
    bool is_final,
    std::vector<Transcript> & transcripts) {
    const size_t samples_per_window = to_samples(config_.window_ms);
    const size_t samples_to_overlap = to_samples(config_.overlap_ms);
    const size_t old_samples_to_keep = std::min(
        window_audio_.size(),
        samples_to_overlap + samples_per_window > new_audio.size()
            ? samples_to_overlap + samples_per_window - new_audio.size()
            : size_t{0});

    std::vector<float> next_window;
    next_window.reserve(old_samples_to_keep + new_audio.size());
    next_window.insert(
        next_window.end(),
        window_audio_.end() - static_cast<std::ptrdiff_t>(old_samples_to_keep),
        window_audio_.end());
    next_window.insert(next_window.end(), new_audio.begin(), new_audio.end());
    window_audio_.swap(next_window);

    if (!has_audible_signal(new_audio)) {
        ++chunk_index_;
        return;
    }

    const std::string text = transcribe_(window_audio_);
    if (!text.empty()) {
        transcripts.push_back(Transcript{chunk_index_, text, is_final});
    }
    ++chunk_index_;
}

void StreamingAudioWindow::normalize_config() {
    config_.sample_rate = std::max(1, clamp_positive(config_.sample_rate, kAsrSampleRate));
    config_.step_ms = std::max(250, clamp_positive(config_.step_ms, 2000));
    config_.window_ms = std::max(config_.step_ms, clamp_positive(config_.window_ms, 6000));
    config_.overlap_ms = std::max(0, std::min(config_.overlap_ms, config_.step_ms));
    config_.min_audio_rms = std::max(0.0f, config_.min_audio_rms);
}

} // namespace vox::asr
