#include "sdl_audio_output.h"

#include "audio_device_utils.h"

#include <SDL.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vox::app {
namespace {

void ensure_sdl_audio_or_throw() {
    if (SDL_WasInit(SDL_INIT_AUDIO) == 0) {
        SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);
        SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);
        if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0 && SDL_Init(SDL_INIT_AUDIO) < 0) {
            throw std::runtime_error(std::string("failed to initialize SDL audio: ") + SDL_GetError());
        }
    }
}

float sanitize(float sample) {
    if (!std::isfinite(sample)) {
        return 0.0f;
    }
    return std::max(-1.0f, std::min(1.0f, sample));
}

std::vector<float> convert_mono(
    const std::vector<float> & input,
    int32_t input_rate,
    int32_t output_rate,
    int32_t output_channels) {
    if (input.empty()) {
        return {};
    }
    if (input_rate <= 0 || output_rate <= 0 || output_channels <= 0) {
        throw std::runtime_error("invalid audio conversion format");
    }

    const double rate_ratio = static_cast<double>(output_rate) / static_cast<double>(input_rate);
    const size_t output_frames = std::max<size_t>(1, static_cast<size_t>(std::ceil(input.size() * rate_ratio)));
    std::vector<float> output(output_frames * static_cast<size_t>(output_channels));

    for (size_t frame = 0; frame < output_frames; ++frame) {
        const double src_pos = static_cast<double>(frame) / rate_ratio;
        const size_t src0 = std::min(input.size() - 1, static_cast<size_t>(src_pos));
        const size_t src1 = std::min(input.size() - 1, src0 + 1);
        const float t = static_cast<float>(src_pos - static_cast<double>(src0));
        const float sample = sanitize(input[src0] + (input[src1] - input[src0]) * t);

        for (int32_t channel = 0; channel < output_channels; ++channel) {
            output[frame * static_cast<size_t>(output_channels) + static_cast<size_t>(channel)] = sample;
        }
    }
    return output;
}

} // namespace

SdlAudioOutput::~SdlAudioOutput() {
    close();
}

void SdlAudioOutput::open(int32_t playback_device_id, int32_t sample_rate, int32_t channels) {
    close();
    ensure_sdl_audio_or_throw();

    if (sample_rate <= 0 || channels <= 0) {
        throw std::runtime_error("invalid playback format");
    }

    SDL_AudioSpec requested;
    SDL_AudioSpec obtained;
    SDL_zero(requested);
    SDL_zero(obtained);

    requested.freq = sample_rate;
    requested.format = AUDIO_F32SYS;
    requested.channels = static_cast<Uint8>(channels);
    requested.samples = 1024;
    requested.callback = nullptr;

    const std::string selected_name = audio_device_name(playback_device_id, false);
    const char * device_name = playback_device_id >= 0 && !selected_name.empty() ? selected_name.c_str() : nullptr;
    device_id_ = SDL_OpenAudioDevice(
        device_name,
        SDL_FALSE,
        &requested,
        &obtained,
        SDL_AUDIO_ALLOW_FREQUENCY_CHANGE | SDL_AUDIO_ALLOW_CHANNELS_CHANGE);
    if (device_id_ == 0) {
        throw std::runtime_error(std::string("failed to open playback device: ") + SDL_GetError());
    }
    if (obtained.format != AUDIO_F32SYS) {
        SDL_CloseAudioDevice(device_id_);
        device_id_ = 0;
        throw std::runtime_error("playback device did not accept float32 audio");
    }

    sample_rate_ = obtained.freq;
    channels_ = obtained.channels;
    device_name_ = selected_name.empty() ? "System default" : selected_name;

    SDL_PauseAudioDevice(device_id_, 0);
}

void SdlAudioOutput::close() {
    if (device_id_ != 0) {
        SDL_ClearQueuedAudio(device_id_);
        SDL_CloseAudioDevice(device_id_);
    }
    device_id_ = 0;
    sample_rate_ = 0;
    channels_ = 0;
    device_name_.clear();
}

void SdlAudioOutput::clear() {
    if (device_id_ != 0) {
        SDL_ClearQueuedAudio(device_id_);
    }
}

void SdlAudioOutput::enqueue_mono(const std::vector<float> & samples, int32_t sample_rate) {
    if (samples.empty()) {
        return;
    }
    if (device_id_ == 0) {
        throw std::runtime_error("playback device is not open");
    }

    const std::vector<float> converted = convert_mono(samples, sample_rate, sample_rate_, channels_);
    if (SDL_QueueAudio(
            device_id_,
            converted.data(),
            static_cast<Uint32>(converted.size() * sizeof(float))) != 0) {
        throw std::runtime_error(std::string("failed to queue audio: ") + SDL_GetError());
    }
}

int32_t SdlAudioOutput::sample_rate() const {
    return sample_rate_;
}

int32_t SdlAudioOutput::channels() const {
    return channels_;
}

uint32_t SdlAudioOutput::queued_bytes() const {
    return device_id_ == 0 ? 0 : SDL_GetQueuedAudioSize(device_id_);
}

std::string SdlAudioOutput::device_name() const {
    return device_name_;
}

} // namespace vox::app
