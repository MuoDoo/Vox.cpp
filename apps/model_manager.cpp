#include "model_manager.h"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <system_error>

namespace vox::app::model {
namespace {

std::filesystem::path model_path(const std::filesystem::path & project_root, const std::string & relative_path) {
    return project_root / std::filesystem::path(relative_path);
}

std::string format_size(uintmax_t bytes) {
    const char * units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }
    char buffer[64];
    if (unit == 0) {
        std::snprintf(buffer, sizeof(buffer), "%llu %s", static_cast<unsigned long long>(bytes), units[unit]);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.1f %s", value, units[unit]);
    }
    return buffer;
}

const char * status_name(const ManagedModelStatus & status) {
    if (status.complete) {
        return "installed";
    }
    if (status.installed || status.has_partial_download) {
        return "incomplete";
    }
    return "missing";
}

void print_model_usage(std::ostream & out) {
    out << "usage: vox model <command> [model-name]\n"
        << "\n"
        << "commands:\n"
        << "  list [--installed]   list supported models and local status\n"
        << "  download <name>      download a supported model\n"
        << "  verify <name>        verify local model files exist and are non-empty\n"
        << "  repair <name>        remove incomplete files, then download missing files\n"
        << "  remove <name>        remove local files for a model\n"
        << "\n"
        << "Run 'vox model list' to see model names.\n";
}

int run_download_command(const ManagedModel & model, const std::filesystem::path & project_root, std::ostream & err) {
    if (model.download_command.empty()) {
        err << "No download command is available for " << model.name << ".\n";
        return 1;
    }

    std::error_code ec;
    const std::filesystem::path previous = std::filesystem::current_path(ec);
    if (ec) {
        err << "Could not read current directory: " << ec.message() << "\n";
        return 1;
    }
    std::filesystem::current_path(project_root, ec);
    if (ec) {
        err << "Could not enter project directory " << project_root << ": " << ec.message() << "\n";
        return 1;
    }
    const int result = std::system(model.download_command.c_str());
    std::error_code restore_ec;
    std::filesystem::current_path(previous, restore_ec);
    if (restore_ec) {
        err << "Could not restore working directory " << previous << ": " << restore_ec.message() << "\n";
        if (result != 0) {
            err << "Download command failed for " << model.name << ".\n";
        }
        return 1;
    }
    if (result != 0) {
        err << "Download command failed for " << model.name << ".\n";
        return 1;
    }
    return 0;
}

void print_model_details(const ManagedModel & model, const ManagedModelStatus & status, std::ostream & out) {
    out << model.name << " (" << status_name(status) << ")\n"
        << "  description: " << model.description << "\n"
        << "  source: " << model.source << "\n"
        << "  version: " << model.version << "\n"
        << "  checksum: " << (model.checksum.empty() ? "unavailable" : model.checksum) << "\n";
    for (size_t i = 0; i < model.files.size(); ++i) {
        const ManagedModelFile & file = model.files[i];
        const ManagedModelFileStatus & file_status = status.files[i];
        out << "  file: " << file_status.path.string()
            << " [" << (file_status.complete ? "ok" : (file_status.exists ? "empty" : "missing")) << "]";
        if (file_status.exists) {
            out << " size=" << format_size(file_status.size);
        } else if (!file.size_hint.empty()) {
            out << " expected=" << file.size_hint;
        }
        out << "\n";
    }
    if (status.has_partial_download) {
        out << "  partial download detected; run 'vox model repair " << model.name << "'.\n";
    }
}

int require_model_name(const std::vector<std::string> & args, std::ostream & err) {
    if (args.size() < 2) {
        err << "Missing model name.\n";
        return 1;
    }
    if (args.size() > 2) {
        err << "Too many arguments for model command.\n";
        return 1;
    }
    return 0;
}

} // namespace

const std::vector<ManagedModel> & supported_models() {
    static const std::vector<ManagedModel> models = {
        {
            "qwen3-asr-1.7b",
            "Default Qwen3-ASR GGUF model and multimodal projector",
            "ggml-org/Qwen3-ASR-1.7B-GGUF",
            "1.7B Q8_0",
            "",
            {
                {"models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf", "about 2.17 GB"},
                {"models/asr/qwen3-asr-1.7b/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", "about 356 MB"},
            },
            "scripts/download-qwen3-asr-gguf.sh",
        },
        {
            "qwen3-asr-0.6b",
            "Smaller Qwen3-ASR GGUF model and multimodal projector",
            "ggml-org/Qwen3-ASR-0.6B-GGUF",
            "0.6B Q8_0",
            "",
            {
                {"models/asr/qwen3-asr-0.6b/Qwen3-ASR-0.6B-Q8_0.gguf", ""},
                {"models/asr/qwen3-asr-0.6b/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf", ""},
            },
            "scripts/download-qwen3-asr-gguf.sh 0.6B Q8_0 models/asr/qwen3-asr-0.6b",
        },
        {
            "whisper-base",
            "Whisper.cpp base GGML ASR model",
            "ggerganov/whisper.cpp",
            "base",
            "",
            {{"models/ggml-base.bin", ""}},
            "./external/whisper.cpp/models/download-ggml-model.sh base models",
        },
        {
            "hymt-translate",
            "Tencent HY-MT1.5 translation GGUF model",
            "tencent/HY-MT1.5-1.8B-GGUF",
            "1.8B Q4_K_M",
            "",
            {{"models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf", "about 1.13 GB"}},
            "scripts/download-hymt-gguf.sh",
        },
        {
            "cosyvoice3-tts",
            "Minimum baked-voice CosyVoice3 TTS GGUF set",
            "cstr/cosyvoice3-0.5b-2512-GGUF",
            "q4_k/q8_0/f16",
            "",
            {
                {"models/tts/cosyvoice3/cosyvoice3-llm-q4_k.gguf", ""},
                {"models/tts/cosyvoice3/cosyvoice3-flow-q8_0.gguf", ""},
                {"models/tts/cosyvoice3/cosyvoice3-hift-f16.gguf", ""},
                {"models/tts/cosyvoice3/cosyvoice3-voices.gguf", ""},
            },
            "scripts/download-cosyvoice3-tts-gguf.sh",
        },
    };
    return models;
}

const ManagedModel * find_model(const std::string & name) {
    for (const ManagedModel & model : supported_models()) {
        if (model.name == name) {
            return &model;
        }
    }
    return nullptr;
}

ManagedModelStatus inspect_model(const ManagedModel & model, const std::filesystem::path & project_root) {
    ManagedModelStatus status;
    status.model = &model;
    status.complete = true;
    for (const ManagedModelFile & file : model.files) {
        ManagedModelFileStatus file_status;
        file_status.path = model_path(project_root, file.relative_path);

        std::error_code ec;
        file_status.exists = std::filesystem::is_regular_file(file_status.path, ec);
        if (file_status.exists) {
            file_status.size = std::filesystem::file_size(file_status.path, ec);
            if (ec) {
                file_status.size = 0;
            }
            file_status.complete = file_status.size > 0;
            status.installed = true;
        }
        status.complete = status.complete && file_status.complete;

        const std::filesystem::path partial = file_status.path.string() + ".part";
        if (std::filesystem::exists(partial, ec)) {
            status.has_partial_download = true;
        }
        status.files.push_back(file_status);
    }
    status.complete = status.complete && !status.has_partial_download;
    return status;
}

int run_model_command(
    const std::vector<std::string> & args,
    const std::filesystem::path & project_root,
    std::ostream & out,
    std::ostream & err) {
    if (args.empty() || args[0] == "-h" || args[0] == "--help" || args[0] == "help") {
        print_model_usage(out);
        return 0;
    }

    const std::string command = args[0];
    if (command == "list") {
        bool installed_only = false;
        if (args.size() > 2 || (args.size() == 2 && args[1] != "--installed")) {
            err << "usage: vox model list [--installed]\n";
            return 1;
        }
        installed_only = args.size() == 2;
        for (const ManagedModel & model : supported_models()) {
            const ManagedModelStatus status = inspect_model(model, project_root);
            if (installed_only && !status.installed && !status.has_partial_download) {
                continue;
            }
            out << model.name << "\t" << status_name(status) << "\t" << model.version << "\t";
            if (!model.files.empty()) {
                out << model_path(project_root, model.files.front().relative_path).string();
            }
            out << "\n";
        }
        return 0;
    }

    if (command != "download" && command != "verify" && command != "repair" && command != "remove") {
        err << "Unknown model command: " << command << "\n";
        print_model_usage(err);
        return 1;
    }
    if (require_model_name(args, err) != 0) {
        return 1;
    }

    const ManagedModel * model = find_model(args[1]);
    if (!model) {
        err << "Unknown model: " << args[1] << "\n"
            << "Run 'vox model list' to see supported models.\n";
        return 1;
    }

    if (command == "verify") {
        const ManagedModelStatus status = inspect_model(*model, project_root);
        print_model_details(*model, status, out);
        if (!status.complete) {
            err << "Model is missing or incomplete. Run 'vox model repair " << model->name << "'.\n";
            return 1;
        }
        return 0;
    }

    if (command == "remove") {
        std::error_code ec;
        for (const ManagedModelFile & file : model->files) {
            const std::filesystem::path path = model_path(project_root, file.relative_path);
            const bool removed = std::filesystem::remove(path, ec);
            if (ec) {
                err << "Could not remove " << path << ": " << ec.message() << "\n";
                return 1;
            }
            std::filesystem::remove(path.string() + ".part", ec);
            out << (removed ? "Removed: " : "Not installed: ") << path.string() << "\n";
        }
        return 0;
    }

    if (command == "repair") {
        std::error_code ec;
        const ManagedModelStatus status = inspect_model(*model, project_root);
        for (size_t i = 0; i < model->files.size(); ++i) {
            const std::filesystem::path path = status.files[i].path;
            if (status.files[i].exists && !status.files[i].complete) {
                std::filesystem::remove(path, ec);
                if (ec) {
                    err << "Could not remove incomplete file " << path << ": " << ec.message() << "\n";
                    return 1;
                }
                out << "Removed incomplete file: " << path.string() << "\n";
            }
            std::filesystem::remove(path.string() + ".part", ec);
        }
    }

    const ManagedModelStatus before = inspect_model(*model, project_root);
    if (before.complete) {
        out << "Model already installed: " << model->name << "\n";
        return 0;
    }
    out << "Downloading " << model->name << " using: " << model->download_command << "\n";
    return run_download_command(*model, project_root, err);
}

} // namespace vox::app::model
