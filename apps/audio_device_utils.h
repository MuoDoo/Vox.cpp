#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vox::app {

struct AudioDeviceInfo {
    int32_t id = -1;
    std::string name;
    bool is_default = false;
    bool likely_virtual_cable = false;
};

std::vector<AudioDeviceInfo> list_audio_devices(bool capture);
std::string audio_device_name(int32_t id, bool capture);
bool is_likely_virtual_cable_device(const std::string & name);

} // namespace vox::app
