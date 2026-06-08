add_library(vox_llama_runtime STATIC
    llama_runtime.cpp
)

target_include_directories(vox_llama_runtime
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(vox_llama_runtime
    PUBLIC
        llama
)
