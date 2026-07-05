set(VOX_CRISPASR_DIR "${CMAKE_SOURCE_DIR}/external/CrispASR")
set(VOX_CRISPASR_SRC_DIR "${VOX_CRISPASR_DIR}/src")
set(VOX_CRISPASR_HELPER_BUILD_DIR "${CMAKE_BINARY_DIR}/crispasr-helper")
set(VOX_CRISPASR_HELPER_EXECUTABLE
    "${VOX_CRISPASR_HELPER_BUILD_DIR}/bin/crispasr${CMAKE_EXECUTABLE_SUFFIX}"
    CACHE FILEPATH "Path to the official CrispASR helper executable used by helper-backed TTS"
)
option(VOX_BUILD_CRISPASR_HELPER_TARGET "Add the vox_crispasr_helper target for official CrispASR CLI builds" ON)

if(NOT EXISTS "${VOX_CRISPASR_SRC_DIR}/cosyvoice3_tts.cpp")
    message(FATAL_ERROR "CrispASR submodule not found. Run: git submodule update --init --recursive.")
endif()

if(VOX_BUILD_CRISPASR_HELPER_TARGET)
    set(VOX_CRISPASR_HELPER_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
    if(NOT VOX_CRISPASR_HELPER_BUILD_TYPE)
        set(VOX_CRISPASR_HELPER_BUILD_TYPE Release)
    endif()

    set(VOX_CRISPASR_HELPER_METAL OFF)
    if(APPLE AND VOX_ENABLE_LLAMA_METAL)
        set(VOX_CRISPASR_HELPER_METAL ON)
    endif()

    add_custom_target(vox_crispasr_helper
        COMMAND "${CMAKE_COMMAND}"
            -S "${VOX_CRISPASR_DIR}"
            -B "${VOX_CRISPASR_HELPER_BUILD_DIR}"
            -DCMAKE_BUILD_TYPE=${VOX_CRISPASR_HELPER_BUILD_TYPE}
            -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${VOX_CRISPASR_HELPER_BUILD_DIR}/bin
            -DBUILD_SHARED_LIBS=OFF
            -DCRISPASR_BUILD_EXAMPLES=ON
            -DCRISPASR_BUILD_TESTS=OFF
            -DCRISPASR_BUILD_SERVER=OFF
            -DCRISPASR_ALL_WARNINGS=OFF
            -DCRISPASR_FATAL_WARNINGS=OFF
            -DGGML_METAL=${VOX_CRISPASR_HELPER_METAL}
        COMMAND "${CMAKE_COMMAND}"
            --build "${VOX_CRISPASR_HELPER_BUILD_DIR}"
            --target crispasr-cli
            --config ${VOX_CRISPASR_HELPER_BUILD_TYPE}
        BYPRODUCTS "${VOX_CRISPASR_HELPER_EXECUTABLE}"
        COMMENT "Building official CrispASR CLI helper"
        VERBATIM
    )
endif()

add_library(vox_tts STATIC
    cosyvoice3_synthesizer.cpp
    kokoro_synthesizer.cpp
    qwen3_tts_synthesizer.cpp
    "${VOX_CRISPASR_SRC_DIR}/cosyvoice3_tts.cpp"
    "${VOX_CRISPASR_SRC_DIR}/chatterbox_campplus.cpp"
    "${VOX_CRISPASR_SRC_DIR}/core/gguf_loader.cpp"
    "${VOX_CRISPASR_SRC_DIR}/core/mel.cpp"
    "${VOX_CRISPASR_SRC_DIR}/core/rvq.cpp"
    "${VOX_CRISPASR_SRC_DIR}/core/audio_resample.cpp"
    "${VOX_CRISPASR_SRC_DIR}/core/kaldi_fbank.cpp"
)

target_include_directories(vox_tts
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        "${VOX_CRISPASR_SRC_DIR}"
        "${VOX_CRISPASR_SRC_DIR}/core"
        "${CMAKE_SOURCE_DIR}/external/llama.cpp/ggml/src"
)

target_link_libraries(vox_tts
    PUBLIC
        ggml
        ggml-base
        ggml-cpu
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

target_compile_definitions(vox_tts
    PRIVATE
        CRISPASR_ESPEAK_DLOPEN=1
        VOX_CRISPASR_HELPER_PATH="${VOX_CRISPASR_HELPER_EXECUTABLE}"
)

if(WIN32)
    target_compile_definitions(vox_tts PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()

if(TARGET ggml-metal)
    target_link_libraries(vox_tts PUBLIC ggml-metal)
endif()
