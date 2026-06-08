#pragma once

namespace vox::llama {

class LlamaRuntime {
public:
    LlamaRuntime();
    ~LlamaRuntime();

    LlamaRuntime(const LlamaRuntime &) = delete;
    LlamaRuntime & operator=(const LlamaRuntime &) = delete;
};

int default_thread_count();

} // namespace vox::llama
