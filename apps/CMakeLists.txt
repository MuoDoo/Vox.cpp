add_executable(vox
    vox.cpp
)

target_link_libraries(vox
    PRIVATE
        vox_realtime_asr
)

target_compile_definitions(vox
    PRIVATE
        VOX_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)
