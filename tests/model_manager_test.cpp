#include "model_manager.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

bool expect(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << message << "\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    bool ok = true;
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() / ("vox_model_manager_test_" + std::to_string(now));
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "models", ec);

    const vox::app::model::ManagedModel * model = vox::app::model::find_model("whisper-base");
    ok = expect(model != nullptr, "whisper-base should be supported") && ok;
    ok = expect(vox::app::model::find_model("missing-model") == nullptr, "unknown model should not resolve") && ok;
    if (!model) {
        return 1;
    }

    vox::app::model::ManagedModelStatus status = vox::app::model::inspect_model(*model, root);
    ok = expect(!status.installed && !status.complete, "missing model should not be complete") && ok;

    const fs::path model_path = root / "models/ggml-base.bin";
    {
        std::ofstream empty(model_path, std::ios::binary);
    }
    status = vox::app::model::inspect_model(*model, root);
    ok = expect(status.installed && !status.complete, "empty model file should be incomplete") && ok;

    std::ostringstream out;
    std::ostringstream err;
    int result = vox::app::model::run_model_command({"list", "--installed"}, root, out, err);
    ok = expect(result == 0, "list --installed should succeed for incomplete models") && ok;
    ok = expect(out.str().find("whisper-base") != std::string::npos, "installed list should include incomplete model") && ok;

    {
        std::ofstream file(model_path, std::ios::binary);
        file << "not a real model, but enough to test file completeness";
    }
    status = vox::app::model::inspect_model(*model, root);
    ok = expect(status.installed && status.complete, "non-empty model file should be complete") && ok;

    out.str("");
    out.clear();
    err.str("");
    err.clear();
    result = vox::app::model::run_model_command({"verify", "whisper-base"}, root, out, err);
    ok = expect(result == 0, "verify should succeed for a complete local model") && ok;
    ok = expect(out.str().find("checksum: unavailable") != std::string::npos, "verify should show checksum status") && ok;

    out.str("");
    out.clear();
    err.str("");
    err.clear();
    result = vox::app::model::run_model_command({"list", "--installed"}, root, out, err);
    ok = expect(result == 0, "list --installed should succeed") && ok;
    ok = expect(out.str().find("whisper-base") != std::string::npos, "installed list should include complete model") && ok;

    out.str("");
    out.clear();
    err.str("");
    err.clear();
    result = vox::app::model::run_model_command({"remove", "whisper-base"}, root, out, err);
    ok = expect(result == 0, "remove should succeed") && ok;
    ok = expect(!fs::exists(model_path), "remove should delete model file") && ok;

    {
        std::ofstream partial(model_path.string() + ".part", std::ios::binary);
        partial << "partial";
    }
    status = vox::app::model::inspect_model(*model, root);
    ok = expect(!status.installed && status.has_partial_download && !status.complete, "partial download should be incomplete") && ok;
    out.str("");
    out.clear();
    err.str("");
    err.clear();
    result = vox::app::model::run_model_command({"list", "--installed"}, root, out, err);
    ok = expect(result == 0, "list --installed should succeed for partial downloads") && ok;
    ok = expect(out.str().find("whisper-base") != std::string::npos, "installed list should include partial download") && ok;

    fs::remove_all(root, ec);
    return ok ? 0 : 1;
}
