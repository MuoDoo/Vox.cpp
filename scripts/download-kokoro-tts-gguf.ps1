param(
    [string]$OutputDir = "models/tts/kokoro",
    [string]$ModelQuant = "q8_0",
    [string]$VoiceName = "af_heart",
    [switch]$Help
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Show-Usage {
    @"
Usage:
  powershell -ExecutionPolicy Bypass -File scripts/download-kokoro-tts-gguf.ps1 [output_dir] [model_quant] [voice_name]

Defaults:
  output_dir   models/tts/kokoro
  model_quant  q8_0
  voice_name   af_heart

Examples:
  powershell -ExecutionPolicy Bypass -File scripts/download-kokoro-tts-gguf.ps1
  powershell -ExecutionPolicy Bypass -File scripts/download-kokoro-tts-gguf.ps1 models/tts/kokoro q8_0 af_bella

Downloads the CrispASR Kokoro GGUF files needed for TTS:

  kokoro-82m-q8_0.gguf
  kokoro-voice-af_heart.gguf

Kokoro uses espeak-ng for phonemization. Install espeak-ng or place
espeak-ng.exe / espeak-ng.dll on PATH.
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Download-One {
    param(
        [Parameter(Mandatory = $true)][string]$RepoId,
        [Parameter(Mandatory = $true)][string]$Filename
    )

    $url = "https://huggingface.co/$RepoId/resolve/main/$Filename"
    $target = Join-Path $OutputDir $Filename
    $partial = "$target.part"

    if (Test-Path -LiteralPath $target) {
        Write-Host "File already exists: $target"
        return
    }

    Write-Host "Downloading $RepoId/$Filename"
    Write-Host "Target: $target"

    try {
        $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($null -ne $curl) {
            & curl.exe -L --fail --continue-at - --output $partial $url
            if ($LASTEXITCODE -ne 0) {
                throw "curl.exe exited with code $LASTEXITCODE"
            }
        } else {
            $ProgressPreference = "SilentlyContinue"
            Invoke-WebRequest -Uri $url -OutFile $partial -UseBasicParsing
        }

        Move-Item -LiteralPath $partial -Destination $target -Force
    } catch {
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        throw
    }
}

Download-One "cstr/kokoro-82m-GGUF" "kokoro-82m-$ModelQuant.gguf"
Download-One "cstr/kokoro-voices-GGUF" "kokoro-voice-$VoiceName.gguf"

Write-Host "Done:"
Write-Host "  $(Join-Path $OutputDir "kokoro-82m-$ModelQuant.gguf")"
Write-Host "  $(Join-Path $OutputDir "kokoro-voice-$VoiceName.gguf")"
