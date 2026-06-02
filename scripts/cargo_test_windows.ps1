# Build/test Rust on Windows (GNU): winlibs MinGW + rustup subst for profile paths with spaces.
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Tools = Join-Path $Repo ".tools"
$WinlibsRoot = Join-Path $Tools "winlibs"
$MingwBin = Join-Path $WinlibsRoot "mingw64\bin"
$WinlibsZip = Join-Path $Tools "winlibs.zip"
$WinlibsUrl = "https://github.com/brechtsanders/winlibs_mingw/releases/download/14.2.0posix-19.1.1-12.0.0-ucrt-r2/winlibs-x86_64-posix-seh-gcc-14.2.0-mingw-w64ucrt-12.0.0-r2.zip"

function Ensure-Winlibs {
    if (Test-Path (Join-Path $MingwBin "gcc.exe")) {
        return
    }
    New-Item -ItemType Directory -Force -Path $Tools | Out-Null
    Write-Host "[mingw] downloading winlibs to $Tools"
    Invoke-WebRequest -Uri $WinlibsUrl -OutFile $WinlibsZip -UseBasicParsing
    Expand-Archive -Path $WinlibsZip -DestinationPath $WinlibsRoot -Force
    Remove-Item $WinlibsZip -Force
}

function Ensure-RustupSubst {
    $homeRustup = Join-Path $env:USERPROFILE ".rustup"
    if (-not (Test-Path $homeRustup)) {
        throw "rustup not installed: https://rustup.rs/"
    }
    if ($homeRustup -match " ") {
        if (-not (subst 2>&1 | Select-String "R:\\\\")) {
            Write-Host "[rustup] subst R: -> $homeRustup"
            subst R: $homeRustup
        }
        $env:RUSTUP_HOME = "R:\"
    }
}

Ensure-Winlibs
Ensure-RustupSubst
$sc = Join-Path $env:RUSTUP_HOME "toolchains\stable-x86_64-pc-windows-gnu\lib\rustlib\x86_64-pc-windows-gnu\bin\self-contained"
$env:PATH = "$MingwBin;$sc;$env:PATH"
$env:CARGO_TARGET_DIR = "R:\tsl-cargo-target"

Push-Location $Repo
rustup override set stable-x86_64-pc-windows-gnu | Out-Null
try {
    if ($args.Count -eq 0) {
        cargo test --workspace
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        cargo build --release -p tsl-cli
        exit $LASTEXITCODE
    }
    & cargo @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
