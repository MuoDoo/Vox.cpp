add_library(vox_realtime_asr STATIC
    whisper_model.cpp
    whisper_realtime.cpp
)

target_link_libraries(vox_realtime_asr
    PRIVATE
        whisper
        vox_sdl_audio
        Threads::Threads
)

target_include_directories(vox_realtime_asr
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        "${VOX_WHISPER_SOURCE_DIR}/include"
        "${VOX_WHISPER_SOURCE_DIR}/ggml/include"
        "${VOX_WHISPER_SOURCE_DIR}/examples"
)
