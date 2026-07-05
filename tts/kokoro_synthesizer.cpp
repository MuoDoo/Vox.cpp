#include "kokoro_synthesizer.h"

#include <stdexcept>
#include <utility>

namespace vox::tts {
namespace {

[[noreturn]] void throw_unavailable() {
    throw std::runtime_error(
        "Kokoro TTS is not available in the in-process Vox build with latest official CrispASR. "
        "Use CosyVoice3 for now; Kokoro will need a CrispASR helper process to avoid ggml ABI conflicts.");
}

} // namespace

KokoroSynthesizer::KokoroSynthesizer(KokoroTtsConfig config)
    : impl_(nullptr) {
    (void) config;
    throw_unavailable();
}

KokoroSynthesizer::~KokoroSynthesizer() = default;

std::string KokoroSynthesizer::synthesize(const std::string & text, uint64_t chunk_index, bool is_final) {
    (void) text;
    (void) chunk_index;
    (void) is_final;
    throw_unavailable();
}

std::vector<float> KokoroSynthesizer::synthesize_pcm(const std::string & text) {
    (void) text;
    throw_unavailable();
}

} // namespace vox::tts
