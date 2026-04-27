# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive project documentation (README.md)
- pyproject.toml with PEP 621 compliant build configuration
- MIT LICENSE file
- CHANGELOG.md following Keep a Changelog format
- Enhanced .gitignore with Python, IDE, and project-specific exclusions
- Documentation for archive directories (root_archive/, scripts/archive/)

### Changed
- Moved archive directories from .gitignore to keep them versioned (used by TUI)
- Updated dependency management from requirements.txt to pyproject.toml

### Technical Debt
- train_tsl51_v3.py is 2000+ lines and needs refactoring into modules
- Type hints missing in many modules (loader.py, trainer.py, etc.)
- Duplicate import/loading logic across inference scripts
- No CI/CD pipeline (GitHub Actions)
- Test coverage needs expansion
- No pre-commit hooks configured
- Windows UTF-8 fix and AMP fallback logic duplicated in multiple files

## [0.1.0] - 2024-04-25

### Added
- Initial TSL-51 Thai Sign Language recognition system
- Multiple model architectures (GRU, MLP, MOPGRU, HybridGRUTransformer, CTC)
- K-Fold Cross Validation with per-fold normalization
- Data augmentation (noise, scaling, horizontal flipping)
- GPU training with CUDA and automatic mixed precision (AMP)
- Real-time camera translation (camera_translate.py)
- Video prediction (predict_video.py)
- Inference on pre-extracted features (inference.py)
- Modular structure in src/ (data/, train/, inference/, models/)
- Support for multiple datasets (user_sign, expert, expert_full, combined, local)
- TUI (Text User Interface) for easy command execution
- Benchmarking script (benchmark_models.py)

### Datasets
- TSL-51 user_sign: 547 samples, 51 classes
- TSL-51 expert: 1,155 original samples
- TSL-51 expert_full: ~45,000 samples with augmentation
- Combined: 1,702 samples (user_sign + expert original)

### Features
- 162-dimensional feature extraction (63 left hand + 63 right hand + 36 pose)
- MediaPipe landmark integration
- Class weighting for imbalanced data
- Early stopping with patience
- Comprehensive results visualization
- Model auto-save functionality
