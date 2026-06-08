add_library(vox_translate STATIC
    llama_translator.cpp
)

target_include_directories(vox_translate
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(vox_translate
    PUBLIC
        vox_llama_runtime
        llama
)
