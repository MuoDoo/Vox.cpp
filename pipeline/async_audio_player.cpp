#include "async_audio_player.h"

#include "cosyvoice3_synthesizer.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace vox::pipeline {
namespace {

AsyncAudioPlayer::PlayFunction require_play(AsyncAudioPlayer::PlayFunction play) {
    if (!play) {
        throw std::runtime_error("missing audio playback function");
    }
    return play;
}

} // namespace

class AsyncAudioPlayer::Impl {
public:
    explicit Impl(PlayFunction play)
        : play_(require_play(std::move(play))),
          worker_([this] { run(); }) {
    }

    ~Impl() {
        close();
    }

    void submit(std::string path) {
        if (path.empty()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                throw std::runtime_error("cannot submit audio after player is closed");
            }
            pending_.push_back(std::move(path));
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
            std::string path;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] {
                    return closed_ || !pending_.empty();
                });
                if (pending_.empty()) {
                    return;
                }

                path = std::move(pending_.front());
                pending_.pop_front();
            }

            try {
                play_(path);
            } catch (const std::exception & error) {
                std::cerr << "audio playback failed: " << error.what() << "\n";
            } catch (...) {
                std::cerr << "audio playback failed: unknown error\n";
            }
        }
    }

    PlayFunction play_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::string> pending_;
    bool closed_ = false;
    std::thread worker_;
};

AsyncAudioPlayer::AsyncAudioPlayer(std::string play_command)
    : impl_(std::make_unique<Impl>(
          [command = std::move(play_command)](const std::string & path) {
              vox::tts::play_audio_file(command, path);
          })) {
}

AsyncAudioPlayer::AsyncAudioPlayer(PlayFunction play)
    : impl_(std::make_unique<Impl>(std::move(play))) {
}

AsyncAudioPlayer::~AsyncAudioPlayer() = default;

void AsyncAudioPlayer::submit(std::string path) {
    impl_->submit(std::move(path));
}

void AsyncAudioPlayer::close() {
    impl_->close();
}

} // namespace vox::pipeline
