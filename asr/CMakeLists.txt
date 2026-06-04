add_library(vox_realtime_asr STATIC
    whisper_model.cpp
    streaming_whisper.cpp
)

target_link_libraries(vox_realtime_asr
    PRIVATE
        whisper
        Threads::Threads
)

target_include_directories(vox_realtime_asr
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        "${VOX_WHISPER_SOURCE_DIR}/include"
        "${VOX_WHISPER_SOURCE_DIR}/ggml/include"
)
