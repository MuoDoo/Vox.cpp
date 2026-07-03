add_library(vox_model_manager STATIC
    model_manager.cpp
)

target_include_directories(vox_model_manager
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(VOX_BUILD_APPS)
add_library(vox_app_audio STATIC
    audio_device_utils.cpp
    microphone_audio_source.cpp
    sdl_audio_output.cpp
)

target_include_directories(vox_app_audio
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        "${CMAKE_SOURCE_DIR}/asr"
)

target_link_libraries(vox_app_audio
    PUBLIC
        vox_sdl_audio
)

add_library(vox_realtime_session STATIC
    realtime_session.cpp
)

target_include_directories(vox_realtime_session
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(vox_realtime_session
    PUBLIC
        vox_app_audio
        vox_translation_pipeline
)

add_executable(vox
    vox.cpp
)

target_link_libraries(vox
    PRIVATE
        vox_model_manager
        vox_translation_pipeline
        vox_app_audio
)

target_include_directories(vox
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_definitions(vox
    PRIVATE
        VOX_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

if(VOX_BUILD_QT_GUI)
    find_package(Qt6 QUIET COMPONENTS Widgets)
    if(Qt6_FOUND)
        add_executable(vox_gui
            gui/main.cpp
        )

        target_link_libraries(vox_gui
            PRIVATE
                Qt6::Widgets
                vox_realtime_session
        )

        target_compile_definitions(vox_gui
            PRIVATE
                VOX_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
        )

        set_target_properties(vox_gui PROPERTIES
            AUTOMOC ON
        )
    else()
        message(STATUS "Qt6 Widgets not found; skipping vox_gui. Set VOX_BUILD_QT_GUI=OFF to silence this.")
    endif()
endif()
endif()
