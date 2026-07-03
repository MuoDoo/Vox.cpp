#pragma once

#include <functional>
#include <memory>
#include <string>

namespace vox::pipeline {

class AsyncAudioPlayer {
public:
    using PlayFunction = std::function<void(const std::string &)>;

    explicit AsyncAudioPlayer(std::string play_command);
    explicit AsyncAudioPlayer(PlayFunction play);
    ~AsyncAudioPlayer();

    AsyncAudioPlayer(const AsyncAudioPlayer &) = delete;
    AsyncAudioPlayer & operator=(const AsyncAudioPlayer &) = delete;

    // Files are played in submission order on a dedicated thread so that
    // synthesis of later utterances can proceed while earlier audio plays.
    void submit(std::string path);

    // Stops accepting work, plays queued files, and joins the worker.
    void close();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vox::pipeline
