#include "llama_runtime.h"

#include "llama.h"

#include <clocale>
#include <cstdio>
#include <mutex>
#include <thread>

namespace vox::llama {
namespace {

void quiet_llama_log(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR && text != nullptr) {
        fputs(text, stderr);
    }
}

std::mutex & runtime_mutex() {
    static std::mutex value;
    return value;
}

int & runtime_ref_count() {
    static int value = 0;
    return value;
}

} // namespace

LlamaRuntime::LlamaRuntime() {
    std::lock_guard<std::mutex> lock(runtime_mutex());
    if (runtime_ref_count() == 0) {
        std::setlocale(LC_NUMERIC, "C");
        llama_log_set(quiet_llama_log, nullptr);
        llama_backend_init();
        ggml_backend_load_all();
    }
    ++runtime_ref_count();
}

LlamaRuntime::~LlamaRuntime() {
    std::lock_guard<std::mutex> lock(runtime_mutex());
    --runtime_ref_count();
    if (runtime_ref_count() == 0) {
        llama_backend_free();
    }
}

int default_thread_count() {
    const unsigned int count = std::thread::hardware_concurrency();
    return count == 0 ? 4 : static_cast<int>(count);
}

} // namespace vox::llama
