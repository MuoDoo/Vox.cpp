cmake_minimum_required(VERSION 3.20)

project(VoxCpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(VOX_BUILD_TESTS "Build Vox tests" ON)
option(VOX_BUILD_APPS "Build Vox command-line applications" ON)
if(APPLE)
    option(VOX_ENABLE_LLAMA_METAL "Enable the llama.cpp Metal backend on Apple platforms" ON)
else()
    option(VOX_ENABLE_LLAMA_METAL "Enable the llama.cpp Metal backend on Apple platforms" OFF)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "Build whisper.cpp tests" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "Build whisper.cpp examples" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "Build whisper.cpp server" FORCE)
set(WHISPER_ALL_WARNINGS OFF CACHE BOOL "Enable whisper.cpp warnings" FORCE)

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Build llama.cpp common utils" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Build llama.cpp tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Build llama.cpp app" FORCE)
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "Enable llama.cpp warnings" FORCE)
set(LLAMA_FATAL_WARNINGS OFF CACHE BOOL "Treat llama.cpp warnings as errors" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "llama.cpp OpenSSL support" FORCE)
if(APPLE AND VOX_ENABLE_LLAMA_METAL)
    set(GGML_METAL ON CACHE BOOL "Enable ggml Metal backend" FORCE)
endif()

set(VOX_WHISPER_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/whisper.cpp")
set(VOX_WHISPER_SOURCE_DIR "")
set(VOX_LLAMA_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp")

if(EXISTS "${VOX_LLAMA_LOCAL_DIR}/CMakeLists.txt")
    add_subdirectory("${VOX_LLAMA_LOCAL_DIR}" EXCLUDE_FROM_ALL)
else()
    message(FATAL_ERROR "llama.cpp submodule not found. Run: git submodule update --init --recursive.")
endif()

if(EXISTS "${VOX_LLAMA_LOCAL_DIR}/tools/mtmd/CMakeLists.txt")
    if(NOT DEFINED LLAMA_INSTALL_VERSION)
        set(LLAMA_INSTALL_VERSION 0.0.0)
    endif()
    add_subdirectory(
        "${VOX_LLAMA_LOCAL_DIR}/tools/mtmd"
        "${CMAKE_BINARY_DIR}/llama-mtmd"
        EXCLUDE_FROM_ALL
    )
else()
    message(FATAL_ERROR "llama.cpp mtmd library not found. Update the llama.cpp submodule.")
endif()

if(EXISTS "${VOX_WHISPER_LOCAL_DIR}/CMakeLists.txt")
    set(VOX_WHISPER_SOURCE_DIR "${VOX_WHISPER_LOCAL_DIR}")
    add_subdirectory("${VOX_WHISPER_SOURCE_DIR}" EXCLUDE_FROM_ALL)
else()
    message(FATAL_ERROR "whisper.cpp submodule not found. Run: git submodule update --init --recursive.")
endif()

find_package(Threads REQUIRED)

if(VOX_BUILD_APPS)
    find_package(SDL2 QUIET)

    set(VOX_SDL2_TARGET "")
    set(VOX_SDL2_CFLAGS "")
    set(VOX_SDL2_LIBS "")
    set(VOX_SDL2_INCLUDE_DIRS "")

    if(TARGET SDL2::SDL2)
        set(VOX_SDL2_TARGET SDL2::SDL2)
    elseif(SDL2_FOUND)
        set(VOX_SDL2_INCLUDE_DIRS ${SDL2_INCLUDE_DIRS})
        set(VOX_SDL2_LIBS ${SDL2_LIBRARIES})
    else()
        find_program(SDL2_CONFIG_EXECUTABLE sdl2-config)
        if(NOT SDL2_CONFIG_EXECUTABLE)
            message(FATAL_ERROR "SDL2 was not found. Install it first, for example: brew install sdl2")
        endif()

        execute_process(
            COMMAND "${SDL2_CONFIG_EXECUTABLE}" --cflags
            OUTPUT_VARIABLE VOX_SDL2_CFLAGS_RAW
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )
        execute_process(
            COMMAND "${SDL2_CONFIG_EXECUTABLE}" --libs
            OUTPUT_VARIABLE VOX_SDL2_LIBS_RAW
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )

        separate_arguments(VOX_SDL2_CFLAGS UNIX_COMMAND "${VOX_SDL2_CFLAGS_RAW}")
        separate_arguments(VOX_SDL2_LIBS UNIX_COMMAND "${VOX_SDL2_LIBS_RAW}")
    endif()

    add_library(vox_sdl_audio STATIC
        "${VOX_WHISPER_SOURCE_DIR}/examples/common-sdl.cpp"
    )
    target_include_directories(vox_sdl_audio
        PUBLIC
            "${VOX_WHISPER_SOURCE_DIR}/examples"
            ${VOX_SDL2_INCLUDE_DIRS}
    )
    target_compile_options(vox_sdl_audio PUBLIC ${VOX_SDL2_CFLAGS})
    if(VOX_SDL2_TARGET)
        target_link_libraries(vox_sdl_audio PUBLIC ${VOX_SDL2_TARGET})
    else()
        target_link_libraries(vox_sdl_audio PUBLIC ${VOX_SDL2_LIBS})
    endif()
endif()

add_subdirectory(llama_runtime)
add_subdirectory(asr)
add_subdirectory(translate)
add_subdirectory(tts)
add_subdirectory(pipeline)
if(VOX_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(VOX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
