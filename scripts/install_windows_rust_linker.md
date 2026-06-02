# Windows: enable `cargo test` locally

**Quick path (recommended):** from repo root run:

```powershell
.\scripts\cargo_test_windows.ps1
```

This downloads winlibs MinGW into `.tools/`, maps rustup to `R:` when your profile path has spaces, and runs `cargo test --workspace`.

---

# Windows: enable `cargo test` manually

`cargo test --workspace` needs a linker. This machine had neither:

- **GNU:** `dlltool.exe` not found (install MinGW)
- **MSVC:** `link.exe` not found (install Visual Studio Build Tools)

Pick **one** path:

## Option A — MSVC (matches CI on `windows-latest`)

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with **Desktop development with C++**.
2. Open **x64 Native Tools Command Prompt** or restart the terminal after install.
3. From repo root:

```powershell
rustup default stable-x86_64-pc-windows-msvc
cd D:\TSL
cargo test --workspace
```

`rust-toolchain.toml` pins `channel = "stable"` (MSVC on Windows).

## Option B — GNU + MSYS2

1. Install [MSYS2](https://www.msys2.org/).
2. In MSYS2 UCRT64 or MINGW64 shell:

```bash
pacman -S mingw-w64-x86_64-toolchain
```

3. Add `C:\msys64\mingw64\bin` to `PATH`, then:

```powershell
rustup default stable-x86_64-pc-windows-gnu
cd D:\TSL
cargo test --workspace
```

Or run `.\scripts\verify_all.ps1` (prepends MSYS2 MinGW when present).

## Option C — WSL2 (recommended for GPU training + Rust)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cd /mnt/d/TSL
cargo test --workspace
bash scripts/train_local_wsl.sh
```
