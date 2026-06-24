#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace vox::app::model {

struct ManagedModelFile {
    std::string relative_path;
    std::string size_hint;
};

struct ManagedModel {
    std::string name;
    std::string description;
    std::string source;
    std::string version;
    std::string checksum;
    std::vector<ManagedModelFile> files;
    std::string download_command;
};

struct ManagedModelFileStatus {
    std::filesystem::path path;
    bool exists = false;
    bool complete = false;
    uintmax_t size = 0;
};

struct ManagedModelStatus {
    const ManagedModel * model = nullptr;
    std::vector<ManagedModelFileStatus> files;
    bool has_partial_download = false;
    bool installed = false;
    bool complete = false;
};

const std::vector<ManagedModel> & supported_models();
const ManagedModel * find_model(const std::string & name);
ManagedModelStatus inspect_model(const ManagedModel & model, const std::filesystem::path & project_root);
int run_model_command(
    const std::vector<std::string> & args,
    const std::filesystem::path & project_root,
    std::ostream & out,
    std::ostream & err);

} // namespace vox::app::model
