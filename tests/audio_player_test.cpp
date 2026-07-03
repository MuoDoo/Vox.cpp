#include "async_audio_player.h"

#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {

bool test_plays_in_submission_order() {
    std::mutex mutex;
    std::vector<std::string> played;

    {
        vox::pipeline::AsyncAudioPlayer player(
            [&mutex, &played](const std::string & path) {
                std::lock_guard<std::mutex> lock(mutex);
                played.push_back(path);
            });

        player.submit("a.wav");
        player.submit("b.wav");
        player.submit("c.wav");
        player.close();
    }

    const std::vector<std::string> expected{"a.wav", "b.wav", "c.wav"};
    if (played != expected) {
        std::cerr << "unexpected playback order/count: " << played.size() << "\n";
        return false;
    }
    return true;
}

bool test_close_drains_pending() {
    std::mutex mutex;
    std::vector<std::string> played;

    vox::pipeline::AsyncAudioPlayer player(
        [&mutex, &played](const std::string & path) {
            std::lock_guard<std::mutex> lock(mutex);
            played.push_back(path);
        });

    for (int i = 0; i < 16; ++i) {
        player.submit("chunk-" + std::to_string(i) + ".wav");
    }
    player.close();

    std::lock_guard<std::mutex> lock(mutex);
    if (played.size() != 16) {
        std::cerr << "close did not drain all pending audio: " << played.size() << "\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!test_plays_in_submission_order()) {
        return 1;
    }
    if (!test_close_drains_pending()) {
        return 1;
    }
    return 0;
}
