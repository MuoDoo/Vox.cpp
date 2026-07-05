#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace vox::app {

enum class RealtimeAsrEngine {
    Whisper,
    Qwen3,
};

enum class RealtimeTtsEngine {
    CosyVoice3,
    Kokoro,
    Qwen3Tts,
};

struct RealtimeSessionConfig {
    std::string project_root = ".";

    RealtimeAsrEngine asr_engine = RealtimeAsrEngine::Qwen3;
    std::string asr_model_path = "models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf";
    std::string asr_mmproj_path = "models/asr/qwen3-asr-1.7b/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf";
    std::string vad_model_path;
    std::string language = "auto";

    std::string translation_model_path = "models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf";
    std::string target_language = "Chinese";

    RealtimeTtsEngine tts_engine = RealtimeTtsEngine::Qwen3Tts;
    std::string tts_model_path =
        "models/tts/qwen3-tts-0.6b-customvoice/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf";
    std::string tts_flow_model_path;
    std::string tts_hift_model_path;
    std::string tts_voices_model_path;
    std::string tts_codec_model_path;
    std::string tts_voice_model_path;
    std::string tts_crispasr_path;
    std::string tts_backend;
    std::string tts_ref_text;
    std::string tts_instruct;
    std::string tts_language = "Chinese";
    std::string tts_voice = "dylan";
    std::string tts_output_dir = "tts-output";

    int32_t capture_device_id = -1;
    int32_t playback_device_id = -1;
    int32_t threads = 4;
    int32_t step_ms = 500;
    int32_t window_ms = 6000;
    int32_t overlap_ms = 300;
    int32_t final_silence_steps = 1;
    int32_t tts_max_tokens = 0;

    float min_audio_rms = 0.001f;
    float input_gain = 1.0f;
    float tts_temperature = 0.8f;
    float tts_length_scale = 1.0f;
    uint64_t tts_seed = 42;

    bool use_gpu = true;
    bool flash_attention = true;
    bool use_vad = true;
    bool speak_partials = false;
};

struct RealtimeStatusEvent {
    std::string message;
    bool running = false;
};

struct RealtimeTextEvent {
    uint64_t chunk_index = 0;
    std::string stage;
    std::string text;
    bool is_final = false;
    int64_t elapsed_ms = 0;
};

struct RealtimeAudioEvent {
    double rms = 0.0;
    float peak = 0.0f;
    uint32_t queued_output_bytes = 0;
};

class RealtimeSession {
public:
    using StatusHandler = std::function<void(RealtimeStatusEvent)>;
    using TextHandler = std::function<void(RealtimeTextEvent)>;
    using AudioHandler = std::function<void(RealtimeAudioEvent)>;
    using ErrorHandler = std::function<void(std::string)>;

    RealtimeSession();
    ~RealtimeSession();

    RealtimeSession(const RealtimeSession &) = delete;
    RealtimeSession & operator=(const RealtimeSession &) = delete;

    bool start(RealtimeSessionConfig config);
    void request_stop();
    void stop();
    bool running() const;

    StatusHandler on_status;
    TextHandler on_text;
    AudioHandler on_audio;
    ErrorHandler on_error;

private:
    void run(RealtimeSessionConfig config);

    std::atomic_bool running_{false};
    std::atomic_bool stop_requested_{false};
    std::thread worker_;
};

} // namespace vox::app
