#include "qwen3_tts_synthesizer.h"

#include "cosyvoice3_synthesizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef VOX_CRISPASR_HELPER_PATH
#define VOX_CRISPASR_HELPER_PATH ""
#endif

namespace vox::tts {
namespace {

std::atomic<uint64_t> g_temp_index{0};

std::string make_output_path(const std::string & output_dir, uint64_t chunk_index, bool is_final) {
    std::ostringstream filename;
    filename << "chunk-" << std::setw(6) << std::setfill('0') << chunk_index
             << (is_final ? "-final" : "-partial") << ".wav";
    return (std::filesystem::path(output_dir) / filename.str()).string();
}

bool has_path_separator(const std::string & path) {
    return path.find('/') != std::string::npos || path.find('\\') != std::string::npos;
}

bool file_exists(const std::string & path) {
    return !path.empty() && std::filesystem::exists(path);
}

void require_file(const std::string & path, const std::string & label) {
    if (!file_exists(path)) {
        throw std::runtime_error("missing " + label + ": " + path);
    }
}

std::string env_string(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
}

std::string lowercase_ascii(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    return value;
}

bool ends_with_ci(const std::string & value, const std::string & suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return lowercase_ascii(value.substr(value.size() - suffix.size())) == lowercase_ascii(suffix);
}

bool contains_ci(const std::string & value, const std::string & needle) {
    return lowercase_ascii(value).find(lowercase_ascii(needle)) != std::string::npos;
}

bool is_default_voice_name(const std::string & voice) {
    const std::string value = lowercase_ascii(voice);
    return value.empty() || value == "auto" || value == "default" || value == "zero-shot" ||
           value == "zero_shot";
}

bool is_chinese_language(const std::string & language) {
    const std::string value = lowercase_ascii(language);
    return value == "zh" || value == "zh-cn" || value == "cmn" || value == "chinese" ||
           value == "mandarin";
}

std::vector<std::string> split_path_list(const std::string & paths) {
    std::vector<std::string> output;
#if defined(_WIN32)
    constexpr char separator = ';';
#else
    constexpr char separator = ':';
#endif
    size_t start = 0;
    while (start <= paths.size()) {
        const size_t end = paths.find(separator, start);
        output.push_back(paths.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return output;
}

std::vector<std::string> executable_suffixes() {
#if defined(_WIN32)
    std::string pathext = env_string("PATHEXT");
    if (pathext.empty()) {
        pathext = ".COM;.EXE;.BAT;.CMD";
    }
    std::vector<std::string> suffixes;
    for (std::string suffix : split_path_list(pathext)) {
        if (!suffix.empty()) {
            suffixes.push_back(std::move(suffix));
        }
    }
    suffixes.push_back("");
    return suffixes;
#else
    return {""};
#endif
}

std::string find_on_path(const std::string & executable) {
    if (executable.empty() || has_path_separator(executable)) {
        return file_exists(executable) ? executable : std::string();
    }

    const std::string path_env = env_string("PATH");
    if (path_env.empty()) {
        return {};
    }

    for (const std::string & dir : split_path_list(path_env)) {
        if (dir.empty()) {
            continue;
        }
        for (const std::string & suffix : executable_suffixes()) {
            const std::filesystem::path candidate = std::filesystem::path(dir) / (executable + suffix);
            if (std::filesystem::exists(candidate)) {
                return candidate.string();
            }
        }
    }
    return {};
}

std::string find_configured_helper_variant(const std::string & built_path) {
    if (built_path.empty() || file_exists(built_path)) {
        return built_path;
    }

    const std::filesystem::path configured(built_path);
    const std::filesystem::path dir = configured.parent_path();
    const std::filesystem::path filename = configured.filename();
    for (const char * config : {"Release", "RelWithDebInfo", "MinSizeRel", "Debug"}) {
        const std::filesystem::path candidate = dir / config / filename;
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }
    return built_path;
}

std::string resolve_helper_path(const Qwen3TtsConfig & config) {
    if (!config.crispasr_path.empty()) {
        return config.crispasr_path;
    }

    const std::string env_path = env_string("VOX_CRISPASR_CLI");
    if (!env_path.empty()) {
        return env_path;
    }

    const std::string built_path = VOX_CRISPASR_HELPER_PATH;
    const std::string built_variant = find_configured_helper_variant(built_path);
    if (!built_variant.empty() && file_exists(built_variant)) {
        return built_variant;
    }

    const std::string on_path = find_on_path("crispasr");
    if (!on_path.empty()) {
        return on_path;
    }

    return built_variant.empty() ? std::string("crispasr") : built_variant;
}

void validate_helper_path(const std::string & helper_path) {
    if (helper_path.empty()) {
        throw std::runtime_error("missing CrispASR helper path");
    }
    if (has_path_separator(helper_path) && !file_exists(helper_path)) {
        throw std::runtime_error(
            "missing CrispASR helper: " + helper_path +
            " (build it with `cmake --build build --target vox_crispasr_helper`, "
            "or set VOX_CRISPASR_CLI / --tts-crispasr-path)");
    }
    if (!has_path_separator(helper_path) && find_on_path(helper_path).empty()) {
        throw std::runtime_error(
            "CrispASR helper is not on PATH: " + helper_path +
            " (build `vox_crispasr_helper`, set VOX_CRISPASR_CLI, or pass --tts-crispasr-path)");
    }
}

std::string qwen3_backend_name(const Qwen3TtsConfig & config) {
    if (!config.backend.empty()) {
        return config.backend;
    }
    return config.voice_model_path.empty() ? "qwen3-tts-customvoice" : "qwen3-tts";
}

std::string format_float(float value) {
    std::ostringstream out;
    out << std::setprecision(6) << value;
    return out.str();
}

std::string temp_output_path(const std::string & output_dir) {
    const uint64_t tick = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const uint64_t index = ++g_temp_index;
    std::ostringstream filename;
    filename << "qwen3-tts-" << tick << "-" << index << ".wav";
    return (std::filesystem::path(output_dir) / filename.str()).string();
}

#if defined(_WIN32)
std::wstring utf8_to_wide(const std::string & value) {
    if (value.empty()) {
        return {};
    }
    const int size = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.c_str(),
        static_cast<int>(value.size()),
        nullptr,
        0);
    if (size <= 0) {
        throw std::runtime_error("failed to convert UTF-8 argument for CreateProcessW");
    }
    std::wstring output(static_cast<size_t>(size), L'\0');
    MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.c_str(),
        static_cast<int>(value.size()),
        output.data(),
        size);
    return output;
}

std::string wide_to_utf8(const std::wstring & value) {
    if (value.empty()) {
        return {};
    }
    const int size = WideCharToMultiByte(
        CP_UTF8,
        0,
        value.c_str(),
        static_cast<int>(value.size()),
        nullptr,
        0,
        nullptr,
        nullptr);
    if (size <= 0) {
        return {};
    }
    std::string output(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(
        CP_UTF8,
        0,
        value.c_str(),
        static_cast<int>(value.size()),
        output.data(),
        size,
        nullptr,
        nullptr);
    return output;
}

std::wstring windows_quote_arg(const std::wstring & value) {
    std::wstring out = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : value) {
        if (c == L'\\') {
            ++backslashes;
            continue;
        }
        if (c == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(c);
            backslashes = 0;
            continue;
        }
        out.append(backslashes, L'\\');
        backslashes = 0;
        out.push_back(c);
    }
    out.append(backslashes * 2, L'\\');
    out.push_back(L'"');
    return out;
}

std::string windows_error_message(DWORD code) {
    wchar_t * message = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS;
    const DWORD size = FormatMessageW(
        flags,
        nullptr,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPWSTR>(&message),
        0,
        nullptr);
    std::wstring wide;
    if (size > 0 && message != nullptr) {
        wide.assign(message, message + size);
        LocalFree(message);
    }
    while (!wide.empty() && (wide.back() == L'\r' || wide.back() == L'\n' || wide.back() == L' ')) {
        wide.pop_back();
    }
    return wide.empty() ? ("Windows error " + std::to_string(code)) : wide_to_utf8(wide);
}
#endif

void run_process(const std::vector<std::string> & args) {
    if (args.empty() || args.front().empty()) {
        throw std::runtime_error("missing process executable");
    }

#if defined(_WIN32)
    std::wstring command_line;
    for (const std::string & arg : args) {
        if (!command_line.empty()) {
            command_line.push_back(L' ');
        }
        command_line += windows_quote_arg(utf8_to_wide(arg));
    }

    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const BOOL ok = CreateProcessW(
        nullptr,
        mutable_command.data(),
        nullptr,
        nullptr,
        FALSE,
        0,
        nullptr,
        nullptr,
        &startup,
        &process);
    if (!ok) {
        throw std::runtime_error(
            "failed to start CrispASR helper '" + args.front() + "': " +
            windows_error_message(GetLastError()));
    }

    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    if (exit_code != 0) {
        throw std::runtime_error("CrispASR helper exited with status " + std::to_string(exit_code));
    }
#else
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const std::string & arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("failed to fork CrispASR helper");
    }
    if (pid == 0) {
        execvp(argv[0], argv.data());
        _exit(errno == ENOENT ? 127 : 126);
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            throw std::runtime_error("failed to wait for CrispASR helper");
        }
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        return;
    }
    if (WIFEXITED(status)) {
        throw std::runtime_error("CrispASR helper exited with status " + std::to_string(WEXITSTATUS(status)));
    }
    throw std::runtime_error("CrispASR helper did not exit normally");
#endif
}

uint16_t read_u16(std::istream & input) {
    uint8_t bytes[2]{};
    input.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t read_u32(std::istream & input) {
    uint8_t bytes[4]{};
    input.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

std::vector<float> read_wav_mono_float(const std::string & path, int32_t expected_sample_rate) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open Qwen3-TTS wav output: " + path);
    }

    char riff[4]{};
    char wave[4]{};
    input.read(riff, 4);
    (void) read_u32(input);
    input.read(wave, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
        throw std::runtime_error("Qwen3-TTS output is not a RIFF/WAVE file: " + path);
    }

    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::streampos data_pos{};
    uint32_t data_size = 0;

    while (input && (!data_size || !format)) {
        char id[4]{};
        input.read(id, 4);
        if (!input) {
            break;
        }
        const uint32_t chunk_size = read_u32(input);
        const std::streampos chunk_start = input.tellg();
        if (std::strncmp(id, "fmt ", 4) == 0) {
            format = read_u16(input);
            channels = read_u16(input);
            sample_rate = read_u32(input);
            (void) read_u32(input);
            (void) read_u16(input);
            bits_per_sample = read_u16(input);
        } else if (std::strncmp(id, "data", 4) == 0) {
            data_pos = input.tellg();
            data_size = chunk_size;
        }

        input.seekg(chunk_start + static_cast<std::streamoff>(chunk_size + (chunk_size & 1)));
    }

    if (format == 0 || channels == 0 || sample_rate == 0 || bits_per_sample == 0 || data_size == 0) {
        throw std::runtime_error("Qwen3-TTS wav output is missing fmt/data chunks: " + path);
    }
    if (sample_rate != static_cast<uint32_t>(expected_sample_rate)) {
        throw std::runtime_error(
            "unexpected Qwen3-TTS sample rate " + std::to_string(sample_rate) +
            " Hz; expected " + std::to_string(expected_sample_rate) + " Hz");
    }
    if (format != 1 && format != 3) {
        throw std::runtime_error("unsupported Qwen3-TTS wav format: " + std::to_string(format));
    }
    if (!((format == 1 && bits_per_sample == 16) || (format == 3 && bits_per_sample == 32))) {
        throw std::runtime_error("unsupported Qwen3-TTS wav bit depth: " + std::to_string(bits_per_sample));
    }

    const uint32_t bytes_per_sample = bits_per_sample / 8;
    const uint32_t frame_size = bytes_per_sample * channels;
    if (frame_size == 0 || data_size < frame_size) {
        throw std::runtime_error("Qwen3-TTS wav output is empty: " + path);
    }

    input.clear();
    input.seekg(data_pos);

    const uint32_t frames = data_size / frame_size;
    std::vector<float> output;
    output.reserve(frames);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        double mixed = 0.0;
        for (uint16_t channel = 0; channel < channels; ++channel) {
            if (format == 1) {
                const uint16_t raw = read_u16(input);
                const int16_t sample = static_cast<int16_t>(raw);
                mixed += static_cast<double>(sample) / 32768.0;
            } else {
                const uint32_t raw = read_u32(input);
                float sample = 0.0f;
                static_assert(sizeof(sample) == sizeof(raw), "float size must match wav f32 sample size");
                std::memcpy(&sample, &raw, sizeof(sample));
                mixed += sample;
            }
        }
        output.push_back(static_cast<float>(mixed / static_cast<double>(channels)));
    }
    return output;
}

} // namespace

class Qwen3TtsSynthesizer::Impl {
public:
    explicit Impl(Qwen3TtsConfig config)
        : config_(std::move(config)),
          helper_path_(resolve_helper_path(config_)),
          backend_(qwen3_backend_name(config_)) {
        if (config_.model_path.empty()) {
            throw std::runtime_error("missing Qwen3-TTS talker model path");
        }
        if (config_.output_dir.empty()) {
            throw std::runtime_error("missing TTS output directory");
        }
        if (config_.threads <= 0) {
            throw std::runtime_error("Qwen3-TTS thread count must be positive");
        }

        validate_helper_path(helper_path_);
        require_file(config_.model_path, "Qwen3-TTS talker model");
        if (!config_.codec_model_path.empty()) {
            require_file(config_.codec_model_path, "Qwen3-TTS tokenizer/codec model");
        }
        if (!config_.voice_model_path.empty()) {
            require_file(config_.voice_model_path, "Qwen3-TTS voice model/reference");
            if (ends_with_ci(config_.voice_model_path, ".wav") && config_.ref_text.empty()) {
                throw std::runtime_error("Qwen3-TTS reference WAV requires --tts-ref-text");
            }
        }
    }

    std::string synthesize(const std::string & text, uint64_t chunk_index, bool is_final) {
        const std::string output_path = make_output_path(config_.output_dir, chunk_index, is_final);
        run_helper(text, output_path);
        if (config_.play_after_synthesis) {
            play_audio_file(
                config_.play_command.empty() ? default_tts_play_command() : config_.play_command,
                output_path);
        }
        return output_path;
    }

    std::vector<float> synthesize_pcm(const std::string & text) {
        const std::string output_path = temp_output_path(config_.output_dir);
        try {
            run_helper(text, output_path);
            std::vector<float> pcm = read_wav_mono_float(output_path, kQwen3TtsSampleRate);
            std::error_code ec;
            std::filesystem::remove(output_path, ec);
            return pcm;
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove(output_path, ec);
            throw;
        }
    }

private:
    void run_helper(const std::string & text, const std::string & output_path) const {
        if (text.empty()) {
            throw std::runtime_error("cannot synthesize empty text");
        }

        const std::filesystem::path output_file(output_path);
        if (!output_file.parent_path().empty()) {
            std::filesystem::create_directories(output_file.parent_path());
        }

        std::vector<std::string> args = {
            helper_path_,
            "--backend",
            backend_,
            "-m",
            config_.model_path,
            "--tts",
            text,
            "--tts-output",
            output_path,
            "-t",
            std::to_string(config_.threads),
            "--temperature",
            format_float(config_.temperature),
            "--seed",
            std::to_string(config_.seed),
            "-np",
        };

        if (!config_.use_gpu) {
            args.push_back("--no-gpu");
        }
        if (config_.flash_attention) {
            args.push_back("--flash-attn");
        } else {
            args.push_back("--no-flash-attn");
        }
        if (!config_.language.empty()) {
            args.push_back("-l");
            args.push_back(config_.language);
        }
        if (!config_.codec_model_path.empty()) {
            args.push_back("--codec-model");
            args.push_back(config_.codec_model_path);
        }
        if (!config_.voice_model_path.empty()) {
            args.push_back("--voice");
            args.push_back(config_.voice_model_path);
        } else if (contains_ci(backend_, "voicedesign")) {
            // VoiceDesign uses --instruct, not --voice.
        } else if (contains_ci(backend_, "customvoice") && is_default_voice_name(config_.voice)) {
            args.push_back("--voice");
            args.push_back(is_chinese_language(config_.language) ? "dylan" : "vivian");
        } else if (!is_default_voice_name(config_.voice)) {
            args.push_back("--voice");
            args.push_back(config_.voice);
        }
        if (!config_.ref_text.empty()) {
            args.push_back("--ref-text");
            args.push_back(config_.ref_text);
        }
        if (!config_.instruct.empty()) {
            args.push_back("--instruct");
            args.push_back(config_.instruct);
        }

        run_process(args);
        if (!file_exists(output_path)) {
            throw std::runtime_error("CrispASR helper did not create Qwen3-TTS output: " + output_path);
        }
    }

    Qwen3TtsConfig config_;
    std::string helper_path_;
    std::string backend_;
};

Qwen3TtsSynthesizer::Qwen3TtsSynthesizer(Qwen3TtsConfig config)
    : impl_(new Impl(std::move(config))) {
}

Qwen3TtsSynthesizer::~Qwen3TtsSynthesizer() {
    delete impl_;
}

std::string Qwen3TtsSynthesizer::synthesize(const std::string & text, uint64_t chunk_index, bool is_final) {
    return impl_->synthesize(text, chunk_index, is_final);
}

std::vector<float> Qwen3TtsSynthesizer::synthesize_pcm(const std::string & text) {
    return impl_->synthesize_pcm(text);
}

} // namespace vox::tts
