# Scripts Archive

This directory contains historical scripts that have been superseded by the modular `src/` structure.

## Active Scripts (Referenced by TUI)

- `data/download_tsl51_v2.py` - TSL-51 dataset download (used by TUI)
- `data/download_expert_full.py` - Expert dataset download (used by TUI)
- `verify/` - Data verification scripts (used by TUI)

## Historical Scripts

These scripts are kept for reference but are no longer actively maintained:

- `archive/scripts_legacy/` - Legacy top-level scripts
- `archive/top_level_legacy/` - Legacy scripts moved from root
- `archive/data/` - Historical data download scripts
- `archive/debug/` - Debug scripts
- `archive/verify/` - Verification scripts (some still used)

## Migration Status

The active scripts in this directory should eventually be moved to proper locations:
- Download scripts → `src/data/download.py`
- Verify scripts → `src/data/verify.py`
