add_library(vox_model_manager STATIC
    model_manager.cpp
)

target_include_directories(vox_model_manager
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(VOX_BUILD_APPS)
add_executable(vox
    vox.cpp
    microphone_audio_source.cpp
)

target_link_libraries(vox
    PRIVATE
        vox_model_manager
        vox_translation_pipeline
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
endif()
