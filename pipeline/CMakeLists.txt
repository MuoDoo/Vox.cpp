add_library(vox_translation_pipeline STATIC
    async_transcript_translator.cpp
)

target_include_directories(vox_translation_pipeline
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(vox_translation_pipeline
    PUBLIC
        vox_realtime_asr
        vox_translate
)
