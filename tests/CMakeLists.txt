add_executable(vox_asr_jfk_test
    asr_jfk_test.cpp
)

target_link_libraries(vox_asr_jfk_test
    PRIVATE
        vox_realtime_asr
)

target_compile_definitions(vox_asr_jfk_test
    PRIVATE
        VOX_TEST_MODEL_PATH="${CMAKE_SOURCE_DIR}/models/ggml-base.bin"
        VOX_TEST_AUDIO_PATH="${VOX_WHISPER_SOURCE_DIR}/samples/jfk.wav"
)

add_test(NAME vox_asr_jfk_test COMMAND vox_asr_jfk_test)
set_tests_properties(vox_asr_jfk_test PROPERTIES SKIP_RETURN_CODE 77)

add_executable(vox_translate_component_test
    translate_component_test.cpp
)

target_link_libraries(vox_translate_component_test
    PRIVATE
        vox_translate
)

add_test(NAME vox_translate_component_test COMMAND vox_translate_component_test)

add_executable(vox_translation_pipeline_test
    translation_pipeline_test.cpp
)

target_link_libraries(vox_translation_pipeline_test
    PRIVATE
        vox_translation_pipeline
)

add_test(NAME vox_translation_pipeline_test COMMAND vox_translation_pipeline_test)

add_executable(vox_translate_hymt_test
    translate_hymt_test.cpp
)

target_link_libraries(vox_translate_hymt_test
    PRIVATE
        vox_translate
)

target_compile_definitions(vox_translate_hymt_test
    PRIVATE
        VOX_TEST_TRANSLATE_MODEL_PATH="${CMAKE_SOURCE_DIR}/models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf"
)

add_test(NAME vox_translate_hymt_test COMMAND vox_translate_hymt_test)
set_tests_properties(vox_translate_hymt_test PROPERTIES SKIP_RETURN_CODE 77)
