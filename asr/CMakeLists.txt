add_library(vox_realtime_asr STATIC
    whisper_model.cpp
    silero_vad.cpp
    streaming_audio_window.cpp
    streaming_whisper.cpp
    qwen_asr_model.cpp
    streaming_qwen_asr.cpp
)

target_link_libraries(vox_realtime_asr
    PUBLIC
        vox_llama_runtime
    PRIVATE
        whisper
        mtmd
        Threads::Threads
)

target_include_directories(vox_realtime_asr
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        "${VOX_WHISPER_SOURCE_DIR}/include"
        "${VOX_WHISPER_SOURCE_DIR}/ggml/include"
)
