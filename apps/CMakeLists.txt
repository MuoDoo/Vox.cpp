add_executable(vox
    vox.cpp
    microphone_audio_source.cpp
)

target_link_libraries(vox
    PRIVATE
        vox_realtime_asr
        vox_sdl_audio
)

target_include_directories(vox
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_definitions(vox
    PRIVATE
        VOX_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)
