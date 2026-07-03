set(VOX_CRISPASR_DIR "${CMAKE_SOURCE_DIR}/external/CrispASR")
set(VOX_CRISPASR_SRC_DIR "${VOX_CRISPASR_DIR}/src")

if(NOT EXISTS "${VOX_CRISPASR_SRC_DIR}/cosyvoice3_tts.cpp")
    message(FATAL_ERROR "CrispASR submodule not found. Run: git submodule update --init --recursive.")
endif()
if(NOT EXISTS "${VOX_CRISPASR_SRC_DIR}/kokoro.cpp")
    message(FATAL_ERROR "CrispASR Kokoro runtime not found. Run: git submodule update --init --recursive.")
endif()
if(NOT EXISTS "${VOX_CRISPASR_SRC_DIR}/qwen3_tts.cpp")
    message(FATAL_ERROR "CrispASR Qwen3-TTS runtime not found. Run: git submodule update --init --recursive.")
endif()

add_library(vox_tts STATIC
    cosyvoice3_synthesizer.cpp
    kokoro_synthesizer.cpp
    qwen3_tts_synthesizer.cpp
    "${VOX_CRISPASR_SRC_DIR}/cosyvoice3_tts.cpp"
    "${VOX_CRISPASR_SRC_DIR}/kokoro.cpp"
    "${VOX_CRISPASR_SRC_DIR}/qwen3_tts.cpp"
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

target_compile_definitions(vox_tts PRIVATE CRISPASR_ESPEAK_DLOPEN=1)

if(WIN32)
    target_compile_definitions(vox_tts PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()

if(TARGET ggml-metal)
    target_link_libraries(vox_tts PUBLIC ggml-metal)
endif()
