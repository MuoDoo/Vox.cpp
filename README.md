# Vox.cpp

Local voice-to-voice experiments in C++. The first milestone is realtime local speech recognition with `whisper.cpp`; the next milestone is local text translation with `llama.cpp`.

## Current Target

`asr/` is a small realtime Whisper ASR component. `translate/` is a llama.cpp translation component for GGUF translation models. `apps/vox.cpp` is the main program entry; for now it starts ASR and prints transcripts.

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

Download or place a local GGML model under `models/`. For multilingual recognition, use a non-`.en` model.

```sh
mkdir -p models
./external/whisper.cpp/models/download-ggml-model.sh base models
```

That creates `models/ggml-base.bin`.

For Chinese recognition, a larger multilingual model is usually better:

```sh
./external/whisper.cpp/models/download-ggml-model.sh small models
```

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

## Run

Default model and auto language:

```sh
./build/bin/vox
```

Explicit model:

```sh
./build/bin/vox models/ggml-base.bin
```

Chinese example:

```sh
./build/bin/vox models/ggml-small.bin zh
```

The app intentionally has no CLI framework yet. The ASR behavior lives in `vox::asr::RealtimeWhisper`.

## Test

The ASR test uses `external/whisper.cpp/samples/jfk.wav` as a local fixture and `models/ggml-base.bin` as the model.
The HY-MT test loads `models/translate/HY-MT1.5-1.8B-Q4_K_M.gguf`; if it is missing, the test is skipped.

```sh
ctest --test-dir build --output-on-failure
```

Run only the translation model test:

```sh
./build/bin/vox_translate_hymt_test
```

## Next Milestones

1. Improve streaming UX by stabilizing partial/final segments.
2. Wire ASR chunks into the local llama.cpp translation component.
3. Add local TTS and audio playback for voice-to-voice translation.
