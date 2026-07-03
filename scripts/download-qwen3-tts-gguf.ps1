param(
    [string]$OutputDir = "models/tts/qwen3-tts-0.6b-customvoice",
    [string]$Variant = "customvoice",
    [string]$Quant = "q8_0"
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host @"
Usage:
  powershell -ExecutionPolicy Bypass -File scripts/download-qwen3-tts-gguf.ps1 [-OutputDir DIR] [-Variant customvoice|base] [-Quant q8_0]

Defaults:
  OutputDir  models/tts/qwen3-tts-0.6b-customvoice
  Variant    customvoice
  Quant      q8_0

Downloads the Qwen3-TTS 0.6B talker and required tokenizer/codec GGUF.
"@
}

if ($Variant -eq "-h" -or $Variant -eq "--help") {
    Show-Usage
    exit 0
}

switch ($Variant.ToLowerInvariant()) {
    { $_ -in @("customvoice", "custom-voice", "cv") } {
        $RepoId = "cstr/qwen3-tts-0.6b-customvoice-GGUF"
        $ModelFilename = "qwen3-tts-12hz-0.6b-customvoice-$Quant.gguf"
        break
    }
    "base" {
        $RepoId = "cstr/qwen3-tts-0.6b-base-GGUF"
        $ModelFilename = "qwen3-tts-12hz-0.6b-base-$Quant.gguf"
        break
    }
    default {
        Write-Error "Unknown variant: $Variant"
    }
}

$CodecRepoId = "cstr/qwen3-tts-tokenizer-12hz-GGUF"
$CodecFilename = "qwen3-tts-tokenizer-12hz.gguf"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Download-One([string]$Repo, [string]$Filename) {
    $Url = "https://huggingface.co/$Repo/resolve/main/$Filename"
    $Target = Join-Path $OutputDir $Filename
    $Partial = "$Target.part"

    if (Test-Path $Target) {
        Write-Host "File already exists: $Target"
        return
    }

    Write-Host "Downloading $Repo/$Filename"
    Write-Host "Target: $Target"

    try {
        Invoke-WebRequest -Uri $Url -OutFile $Partial -MaximumRedirection 10
        Move-Item -Force $Partial $Target
    } catch {
        if (Test-Path $Partial) {
            Remove-Item -Force $Partial
        }
        throw
    }
}

Download-One $RepoId $ModelFilename
Download-One $CodecRepoId $CodecFilename

Write-Host "Done:"
Write-Host "  $(Join-Path $OutputDir $ModelFilename)"
Write-Host "  $(Join-Path $OutputDir $CodecFilename)"
