#include "audio_device_utils.h"

#include <SDL.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace vox::app {
namespace {

bool ensure_sdl_audio() {
    if (SDL_WasInit(SDL_INIT_AUDIO) != 0) {
        return true;
    }

    SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);
    SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);
    return SDL_InitSubSystem(SDL_INIT_AUDIO) == 0 || SDL_Init(SDL_INIT_AUDIO) == 0;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

} // namespace

bool is_likely_virtual_cable_device(const std::string & name) {
    const std::string lower = lowercase(name);
    return contains(lower, "vb-audio") ||
           contains(lower, "vb-cable") ||
           contains(lower, "voicemeeter") ||
           contains(lower, "virtual cable") ||
           contains(lower, "cable input") ||
           contains(lower, "cable-a input") ||
           contains(lower, "cable-b input") ||
           contains(lower, "cable-c input") ||
           contains(lower, "cable-d input");
}

std::vector<AudioDeviceInfo> list_audio_devices(bool capture) {
    std::vector<AudioDeviceInfo> devices;
    devices.push_back({-1, "System default", true, false});

    if (!ensure_sdl_audio()) {
        return devices;
    }

    const int sdl_capture = capture ? SDL_TRUE : SDL_FALSE;
    const int count = SDL_GetNumAudioDevices(sdl_capture);
    if (count < 0) {
        return devices;
    }

    for (int i = 0; i < count; ++i) {
        const char * name = SDL_GetAudioDeviceName(i, sdl_capture);
        if (!name) {
            continue;
        }
        devices.push_back({
            static_cast<int32_t>(i),
            name,
            false,
            is_likely_virtual_cable_device(name),
        });
    }
    return devices;
}

std::string audio_device_name(int32_t id, bool capture) {
    if (id < 0) {
        return "System default";
    }
    if (!ensure_sdl_audio()) {
        return {};
    }
    const char * name = SDL_GetAudioDeviceName(id, capture ? SDL_TRUE : SDL_FALSE);
    return name ? std::string(name) : std::string();
}

} // namespace vox::app
