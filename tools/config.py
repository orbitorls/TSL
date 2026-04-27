#!/usr/bin/env python3
"""
TSL Configuration Module
=======================
Defines all commands, presets, and training configurations.
This module is separated for easy maintenance and modification.

Architecture:
- COMMANDS: Dict[str, Dict] - All available CLI commands
- PRESETS: Dict[str, Dict] - Training configuration presets

Usage:
    from tools.config import COMMANDS, PRESETS, get_command, get_preset
"""
from typing import Dict, Optional, List, Any


# ============================================================================
# COMMAND DEFINITIONS - ALL FUNCTIONS WITH CONFIGURABLE PARAMS
# ============================================================================
# COMMAND DEFINITIONS - ALL FUNCTIONS WITH CONFIGURABLE PARAMS
# ============================================================================
COMMANDS: Dict[str, Dict[str, Any]] = {
    # TRAINING
    "train_gru": {
        "name": "Train GRU",
        "desc": "Train GRU model with K-Fold Cross Validation",
        "cmd": "python train_tsl51_v3.py",
        "params": [
            ("Model", "model", "gru", "select", [("GRU", "gru"), ("MLP", "mlp")]),
            ("Epochs", "epochs", "50", "input", None),
            ("Batch Size", "batch", "128", "input", None),
            ("Layers", "layers", "3", "input", None),
            ("Hidden Dim", "hidden", "256", "input", None),
            ("Learning Rate", "lr", "0.001", "input", None),
            ("Augmentation", "augment", "5", "input", None),
            ("K-Folds", "k_folds", "0", "input", None),
        ]
    },
    "train_mlp": {
        "name": "Train MLP",
        "desc": "Train MLP model (faster, simpler)",
        "cmd": "python train_tsl51_v3.py --model mlp",
        "params": [
            ("Epochs", "epochs", "50", "input", None),
            ("Batch Size", "batch", "128", "input", None),
            ("Layers", "layers", "3", "input", None),
            ("Hidden Dim", "hidden", "256", "input", None),
            ("Learning Rate", "lr", "0.001", "input", None),
            ("Augmentation", "augment", "5", "input", None),
        ]
    },
    "train_expert": {
        "name": "Train Expert",
        "desc": "Train Expert model variant",
        "cmd": "python train_tsl51_v3.py --dataset tsl51_expert",
        "params": [
            ("Epochs", "epochs", "30", "input", None),
            ("Batch Size", "batch", "64", "input", None),
        ]
    },
    
    # INFERENCE
    "inference": {
        "name": "Inference",
        "desc": "Run inference on landmark data (.npz)",
        "cmd": "python inference.py --model models/tsl51_gru_best.pt --input data.npz --top-k 3",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
            ("Input File", "input", "data.npz", "input", None),
            ("Top K", "top_k", "3", "input", None),
        ]
    },
    "predict_video": {
        "name": "Predict Video",
        "desc": "Predict from video file using MediaPipe",
        "cmd": "python predict_video.py --model models/tsl51_gru_best.pt --input video.mp4",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
            ("Video Path", "input", "video.mp4", "input", None),
        ]
    },
    "camera": {
        "name": "Camera Translation",
        "desc": "Real-time camera translation",
        "cmd": "python camera_translate.py --model models/tsl51_gru_best.pt",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
        ]
    },
    "translate": {
        "name": "Translate JSON",
        "desc": "Translate JSON landmark files",
        "cmd": "python translate.py --model models/tsl51_gru_best.pt --input landmarks.json",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
            ("Input JSON", "input", "landmarks.json", "input", None),
        ]
    },
    
    # DATA
    "download": {
        "name": "Download TSL-51",
        "desc": "Download TSL-51 dataset from HuggingFace",
        "cmd": "python scripts/archive/data/download_tsl51_v2.py --samples 1200",
        "params": [
            ("Max Samples", "max_samples", "", "input", None),
        ]
    },
    "download_expert": {
        "name": "Download Expert",
        "desc": "Download Expert dataset",
        "cmd": "python scripts/archive/data/download_expert_full.py",
        "params": [
            ("Max Samples", "max_samples", "", "input", None),
        ]
    },
    "clear_cache": {
        "name": "Clear Cache",
        "desc": "Clear cached datasets",
        "cmd": 'python -c "import shutil; shutil.rmtree(\'.cache\', ignore_errors=True); print(\'Cache cleared\')"',
        "params": []
    },
    
    # EXPORT
    "export_model": {
        "name": "Export Model",
        "desc": "Export trained model to portable format",
        "cmd": "python root_archive/export_model.py --input models/tsl51_gru_best.pt",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
        ]
    },
    "export_onnx": {
        "name": "Export ONNX",
        "desc": "Export model to ONNX format",
        "cmd": "python root_archive/export_onnx.py",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
        ]
    },
    
    # ANALYSIS
    "benchmark": {
        "name": "Benchmark",
        "desc": "Run video benchmark (WER/BLEU/ROUGE)",
        "cmd": "python benchmark_models.py --dataset tsl51_user_sign --samples 50",
        "params": [
            ("Model Path", "model", "models/tsl51_gru_best.pt", "input", None),
            ("Samples", "samples", "50", "input", None),
        ]
    },
    "reports": {
        "name": "Reports",
        "desc": "Create benchmark reports",
        "cmd": "python root_archive/create_benchmark_report.py",
        "params": []
    },
    "check_data": {
        "name": "Check Data",
        "desc": "Run data verification scripts",
        "cmd": "python -c \"import os; [os.system(f'python scripts/archive/verify/{f}') for f in os.listdir('scripts/archive/verify') if f.endswith('.py')]\"",
        "params": []
    },
    
    # WEBSITE
    "website_dev": {
        "name": "Website Dev",
        "desc": "Start TSL website (localhost)",
        "cmd": "cd tsl-website && npm run dev",
        "params": []
    },
    "website_deploy": {
        "name": "Website Deploy",
        "desc": "Deploy to Vercel",
        "cmd": "npx vercel --dir tsl-website",
        "params": []
    },
}


# ============================================================================
# TRAINING PRESETS - Pre-defined configurations for quick access
# ============================================================================
PRESETS = {
    "quick": {
        "model": "gru", 
        "epochs": "5", 
        "batch": "64", 
        "layers": "2", 
        "hidden": "128", 
        "augment": "0", 
        "k_folds": "5"
    },
    "default": {
        "model": "gru", 
        "epochs": "50", 
        "batch": "128", 
        "layers": "3", 
        "hidden": "256", 
        "augment": "5", 
        "k_folds": "5"
    },
    "full_cv": {
        "model": "gru", 
        "epochs": "100", 
        "batch": "128", 
        "layers": "3", 
        "hidden": "256", 
        "augment": "10", 
        "k_folds": "5"
    },
    "mlp": {
        "model": "mlp", 
        "epochs": "50", 
        "batch": "128", 
        "layers": "3", 
        "hidden": "256", 
        "augment": "5", 
        "k_folds": "5"
    },
}

# Alias for compatibility
TRAINING_PRESETS = PRESETS


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_command(cmd_key: str) -> Optional[Dict[str, Any]]:
    """Get command by key."""
    return COMMANDS.get(cmd_key)


def get_preset(name: str) -> Dict[str, str]:
    """Get preset by name with fallback to default."""
    return PRESETS.get(name, PRESETS["default"])


def list_commands() -> List[str]:
    """List all available command keys."""
    return list(COMMANDS.keys())


def list_presets() -> List[str]:
    """List all available preset keys."""
    return list(PRESETS.keys())
