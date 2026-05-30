# Root Archive

This directory contains legacy scripts that are still referenced by the TUI (tools/config.py).

## Active Scripts (Referenced by TUI)

- `export_model.py` - Export trained model to portable format
- `export_onnx.py` - Export model to ONNX format
- `create_benchmark_report.py` - Create benchmark reports

## Historical Scripts

These scripts may be outdated but kept for reference:

- `analyze_sentence_data.py` - Sentence data analysis
- `benchmark_video.py` - Video benchmarking (superseded by benchmark_models.py)
- `download_expert_fixed.py` - Expert dataset download (fixed version)
- `download_expert_full.py` - Full expert dataset download
- `download_tsl51_v2.py` - TSL-51 dataset download (v2)
- `inference_standalone.py` - Standalone inference script
- `onnx_inference.py` - ONNX model inference
- `test_model_with_videos.py` - Model testing with videos
- `train_expert.py` - Expert training script
- `tsl.py` - Legacy TSL script
- `verify_fixes.py` - Verification script

## Migration Status

The active scripts in this directory should eventually be moved to proper locations:
- Export scripts → `src/inference/export.py`
- Benchmark report → `src/eval/report.py`
