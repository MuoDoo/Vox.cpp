# Vox.cpp

Local voice-to-voice experiments in C++. ASR can run through either the existing `whisper.cpp` path or a llama.cpp/libmtmd Qwen3-ASR path; translation uses `llama.cpp`; TTS can synthesize translated text with native CosyVoice3, Kokoro, or Qwen3-TTS GGUF runtimes.

## Current Target

`asr/` contains streaming ASR components that accept mono float32 PCM at 16 kHz. `StreamingQwenAsr` is the default ASR path and drives Qwen3-ASR GGUF models through llama.cpp `libmtmd`; `StreamingWhisper` keeps the existing Whisper fallback path. `translate/` is a llama.cpp translation component for GGUF translation models. `tts/` links CrispASR's CosyVoice3, Kokoro, and Qwen3-TTS GGUF runtimes in-process. `apps/vox.cpp` is the main program entry; it captures microphone audio, feeds ASR, optionally translates transcripts, and can synthesize translated text to wav files.

No network service is used at runtime. You need local model files under `models/`.

## Requirements

- CMake 3.20+
- A C++17 compiler
- SDL2

macOS:

```sh
brew install cmake sdl2
```

## Build

Initialize submodules first:

```sh
git submodule update --init --recursive
```

Then build:

```sh
cmake -S . -B build
cmake --build build --target vox -j
```

Build only the reusable libraries and tests without the SDL microphone app:

```sh
cmake -S . -B build-core -DVOX_BUILD_APPS=OFF
cmake --build build-core -j
```

Build the translation component:

```sh
cmake --build build --target vox_translate -j
```

For Apple Silicon GPU acceleration, configure with Metal enabled if your `whisper.cpp` revision does not enable it by default:

```sh
cmake -S . -B build -DGGML_METAL=ON
cmake --build build --target vox -j
```

## Model

The `vox` CLI can list, download, verify, repair, and remove known local models:

```sh
./build/bin/vox model list
./build/bin/vox model download qwen3-asr-1.7b
./build/bin/vox model download kokoro-tts
./build/bin/vox model download qwen3-tts
./build/bin/vox model verify qwen3-asr-1.7b
./build/bin/vox model repair qwen3-asr-1.7b
```

Model verification checks that expected files exist, are non-empty, and do not
have leftover partial downloads. Checksums are reported when metadata is
available; the current bundled manifests rely on file presence and size. Common
aliases such as `kokoro`, `cosyvoice`, and `qwen3-tts` resolve to their
canonical model entries.

### Whisper ASR

Download or place a local Whisper GGML model under `models/`. For multilingual recognition, use a non-`.en` model.

```sh
mkdir -p models
./external/whisper.cpp/models/download-ggml-model.sh base models
```

That creates `models/ggml-base.bin`.

For Chinese recognition, a larger multilingual model is usually better:

```sh
./external/whisper.cpp/models/download-ggml-model.sh small models
```

### Qwen3-ASR

For the llama.cpp ASR path, use the ggml-org Qwen3-ASR GGUF pair. The default Qwen path expects:

```text
models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf
models/asr/qwen3-asr-1.7b/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf
```

Download both files with:

```sh
scripts/download-qwen3-asr-gguf.sh
```

You can use the smaller 0.6B model for faster local experiments:

```sh
scripts/download-qwen3-asr-gguf.sh 0.6B Q8_0 models/asr/qwen3-asr-0.6b
```

Qwen3-ASR uses a separate multimodal projector GGUF. Keep the model and `mmproj` quantization matched.

### Translation

Translation models should also live under `models/`. The current `translate/` component is built around Tencent HY-MT1.5 GGUF via `llama.cpp`.

```sh
scripts/download-hymt-gguf.sh
```

That creates `models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf`.

Tencent's model card shows llama.cpp usage as:

```sh
llama-cli -hf tencent/HY-MT1.5-1.8B-GGUF:Q8_0 \
  -p "Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house." \
  -n 4096 --temp 0.7 --top-k 20 --top-p 0.6 --repeat-penalty 1.05 --no-warmup
```

The component builds the same translation prompt text and applies the GGUF chat template through llama.cpp. It does not hard-code HY chat tokens.

Tencent's model card recommends `top_k=20`, `top_p=0.6`, `temperature=0.7`, and `repeat_penalty=1.05`; these are the component defaults.
Check the Tencent HY Community License before distributing a product that includes this model.

### TTS

The TTS integration calls CrispASR C ABIs directly from the `external/CrispASR` submodule. It does not shell out to the `crispasr` executable; model loading, voice lookup, synthesis, and WAV writing failures are surfaced directly in-process.

CosyVoice3 remains the default TTS engine.

Download the minimum baked-voice CosyVoice3 GGUF set:

```sh
./build/bin/vox model download cosyvoice3-tts
```

That creates:

```text
models/tts/cosyvoice3/cosyvoice3-llm-q4_k.gguf
models/tts/cosyvoice3/cosyvoice3-flow-q8_0.gguf
models/tts/cosyvoice3/cosyvoice3-hift-f16.gguf
models/tts/cosyvoice3/cosyvoice3-voices.gguf
```

Pass the LLM GGUF with `--tts-model`. The runtime auto-discovers sibling flow, HiFT, and voices files when they are in the same directory. If they live elsewhere, pass `--tts-flow-model`, `--tts-hift-model`, and `--tts-voices-model`.

Kokoro-82M is available with `--tts-engine kokoro`:

```sh
./build/bin/vox model download kokoro-tts
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download-kokoro-tts-gguf.ps1
```

That creates:

```text
models/tts/kokoro/kokoro-82m-q8_0.gguf
models/tts/kokoro/kokoro-voice-af_heart.gguf
```

Pass the Kokoro model with `--tts-model`. The runtime auto-discovers `kokoro-voice-af_heart.gguf` in the same directory, or use `--tts-voice-model PATH`. Kokoro uses espeak-ng for phonemization; install espeak-ng or keep the `espeak-ng` executable on PATH. Use `--tts-language LANG` to override the espeak-ng voice, otherwise the ASR language is reused and `auto` becomes `en-us`.

Qwen3-TTS 0.6B is available with `--tts-engine qwen3-tts`. The recommended quick-test path is CustomVoice Q8_0 because it has built-in speakers and does not need a reference WAV:

```sh
./build/bin/vox model download qwen3-tts
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download-qwen3-tts-gguf.ps1
```

That creates:

```text
models/tts/qwen3-tts-0.6b-customvoice/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf
models/tts/qwen3-tts-0.6b-customvoice/qwen3-tts-tokenizer-12hz.gguf
```

Pass the talker GGUF with `--tts-model`. The runtime auto-discovers `qwen3-tts-tokenizer-12hz.gguf` in the same directory, or use `--tts-codec-model PATH`. CustomVoice speakers include `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, and `vivian`; use `dylan` or `eric` for Chinese output tests. The Base variant can also be downloaded with `./build/bin/vox model download qwen3-tts-0.6b-base`; it requires `--tts-voice-model` pointing to a baked voice GGUF or a reference WAV plus `--tts-ref-text`.

## Run

Default Qwen3-ASR model and auto language:

```sh
./build/bin/vox --final-only
```

Explicit Qwen3-ASR language:

```sh
./build/bin/vox --final-only \
  models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf \
  zh
```

Whisper remains available as a fallback by selecting it explicitly:

```sh
./build/bin/vox --asr-engine whisper models/ggml-base.bin en
```

Whisper Chinese example:

```sh
./build/bin/vox --asr-engine whisper models/ggml-small.bin zh
```

Qwen3-ASR with explicit model and projector:

```sh
./build/bin/vox \
  --asr-mmproj models/asr/qwen3-asr-0.6b/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  models/asr/qwen3-asr-0.6b/Qwen3-ASR-0.6B-Q8_0.gguf \
  en
```

For live Whisper ASR, prefer the CPU path first:

```sh
./build/bin/vox --asr-engine whisper --capture 2 --final-only \
  --no-gpu --no-flash-attn --gain 2 --rms-threshold 0.006 --min-token-p 0.35 \
  models/ggml-small.bin zh
```

On short streaming windows, the whisper.cpp Metal/GPU path can be slower or less stable than CPU, especially with small ASR models. If the app captures audio but produces no ASR output, keep `--no-gpu --no-flash-attn` for Whisper ASR. Translation models can still be tuned separately.

Select a capture device by index from the startup device list:

```sh
./build/bin/vox --capture 2 --final-only
```

For debugging live input, print microphone levels once per second:

```sh
./build/bin/vox --asr-engine whisper --capture 2 --debug-audio models/ggml-base.bin zh
```

If Whisper keeps returning no transcript despite visible microphone levels, relax its no-speech filter:

```sh
./build/bin/vox --asr-engine whisper --capture 2 --debug-audio --no-speech-thold 1.0 models/ggml-base.bin zh
```

If it produces hallucinated text during silence, add an RMS gate and token-probability filter:

```sh
./build/bin/vox --asr-engine whisper --capture 2 --debug-audio --whisper-debug \
  --no-gpu --no-flash-attn --gain 2 --rms-threshold 0.006 --min-token-p 0.35 \
  models/ggml-small.bin zh
```

To emit only the last corrected transcript after speech ends:

```sh
./build/bin/vox --asr-engine whisper --capture 2 --final-only \
  --no-gpu --no-flash-attn --gain 2 --rms-threshold 0.006 --min-token-p 0.35 \
  models/ggml-small.bin zh
```

The streaming window can be tuned with millisecond options:

```sh
./build/bin/vox --asr-engine whisper --step 3000 --length 10000 --keep 200 models/ggml-base.bin zh
```

Enable live translation by passing the translation model and target language after the ASR model and language. With Qwen3-ASR as the default ASR backend, use this command to transcribe speech and translate the result into English:

```sh
./build/bin/vox --final-only \
  models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf \
  auto \
  models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf \
  English
```

The `auto` argument keeps Qwen3-ASR language detection enabled. It is still required here because the current CLI uses positional arguments: `[asr_model] [language] [translation_model] [target_language]`.

The same translation path with Whisper ASR is:

```sh
./build/bin/vox --asr-engine whisper \
  models/ggml-base.bin \
  en \
  models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf \
  Chinese
```

The app translates ASR updates on a worker thread so llama.cpp inference does not block microphone capture. If translation falls behind, pending partial ASR updates are coalesced to the latest text; final results are always processed.

Enable TTS for translated final results by adding `--tts-model`. This writes one wav per synthesized result under `tts-output/`. CosyVoice3 is the default engine:

```sh
./build/bin/vox --final-only \
  --tts-model models/tts/cosyvoice3/cosyvoice3-llm-q4_k.gguf \
  models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf \
  auto \
  models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf \
  English
```

Kokoro-82M uses the same pipeline with `--tts-engine kokoro`:

```sh
./build/bin/vox --final-only \
  --tts-engine kokoro \
  --tts-no-gpu \
  --tts-model models/tts/kokoro/kokoro-82m-q8_0.gguf \
  models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf \
  auto \
  models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf \
  English
```

On macOS, keep `--tts-no-gpu` for Kokoro if Metal output sounds like high-frequency noise. ASR can still use GPU with this option.

Qwen3-TTS 0.6B CustomVoice can synthesize without a reference WAV. For English speech translated to Chinese, use a Chinese speaker such as `dylan`:

```sh
./build/bin/vox --final-only \
  --tts-engine qwen3-tts \
  --tts-model models/tts/qwen3-tts-0.6b-customvoice/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf \
  --tts-voice dylan \
  --tts-language Chinese \
  models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf \
  en \
  models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf \
  Chinese
```

For Chinese speech translated to English, pick an English speaker:

```sh
./build/bin/vox --final-only \
  --tts-engine qwen3-tts \
  --tts-model models/tts/qwen3-tts-0.6b-customvoice/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf \
  --tts-voice vivian \
  --tts-language English \
  models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf \
  zh \
  models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf \
  English
```

To play each generated wav after synthesis on macOS:

```sh
./build/bin/vox --final-only --tts-play \
  --tts-model models/tts/cosyvoice3/cosyvoice3-llm-q4_k.gguf \
  models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf \
  auto \
  models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf \
  English
```

By default, TTS only synthesizes final translations to avoid overlapping partial speech. Use `--tts-partials` for lower latency experiments.

TTS synthesis on CPU is dominated by the flow-matching stage. `--tts-flow-steps N` lowers the CFM Euler step count from the model default (10) to trade audio quality for speed; 5-6 steps is usually a good compromise:

```sh
./build/bin/vox --final-only --tts-play --tts-flow-steps 5 \
  --tts-model models/tts/cosyvoice3/cosyvoice3-llm-q4_k.gguf \
  models/asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf \
  auto \
  models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf \
  English
```

For Kokoro, `--tts-length-scale N` controls duration. Values above `1.0` speak slower; values below `1.0` speak faster.

The app intentionally has no CLI framework yet. The reusable ASR behavior lives in `vox::asr::StreamingWhisper` and `vox::asr::StreamingQwenAsr`; SDL microphone capture is an app-layer adapter. The reusable scheduling behavior lives in `vox::pipeline::AsyncTranscriptTranslator` and `vox::pipeline::AsyncTextToSpeech`.

## ASR Stream API

`vox::asr::StreamingWhisper` and `vox::asr::StreamingQwenAsr` are independent of microphones and SDL. Feed them mono float32 PCM at 16 kHz:

```cpp
vox::asr::StreamingWhisper recognizer(config);

for (const auto & transcript : recognizer.push_audio(samples)) {
    // Partial transcript for a processed window.
}

for (const auto & transcript : recognizer.flush()) {
    // Final transcript for the end of the stream.
}
```

Qwen3-ASR uses the same streaming shape, with a `StreamingQwenAsrConfig` that includes both `model_path` and `mmproj_path`.

## Test

The Whisper ASR test uses `external/whisper.cpp/samples/jfk.wav` as a local fixture and `models/ggml-base.bin` as the model.
The Qwen3-ASR config test does not load a model; it covers language option normalization for the llama.cpp path. The Qwen3-ASR smoke test uses `tests/fixtures/asr_en.wav` plus the default Qwen3-ASR 1.7B Q8_0 model and mmproj; if either model is missing, the test is skipped. The smoke test runs on CPU by default; set `VOX_TEST_QWEN_USE_GPU=1` to exercise the GPU path.
The HY-MT test loads `models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf`; if it is missing, the test is skipped.
The TTS WAV writer test does not load a model; it validates the local WAV output path shared by the TTS synthesizers.

```sh
ctest --test-dir build --output-on-failure
```

Run only the translation model test:

```sh
./build/bin/vox_translate_hymt_test
```

## Next Milestones

1. Improve streaming UX by stabilizing partial/final segments.
2. Improve translated partial-text stability.
3. Add direct audio-device playback for synthesized speech.
