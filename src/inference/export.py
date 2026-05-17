"""ONNX export utilities for TSL models."""

import sys
from pathlib import Path

import torch

from src.core.models import MODEL_REGISTRY as MODEL_CLASSES


def export_to_onnx(
    model_path: str,
    output_path: str | None = None,
    model_type: str = "gru",
    input_dim: int = 162,
    num_classes: int = 51,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.3,
    opset_version: int = 17,
) -> str:
    """
    Export PyTorch model to ONNX format.

    Args:
        model_path: Path to PyTorch model checkpoint (.pt)
        output_path: Output path for ONNX model (default: same as input with .onnx)
        model_type: Model architecture ('gru', 'mlp', 'mopgru', 'hybrid')
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Get model class
    model_class = MODEL_CLASSES.get(model_type, MODEL_CLASSES["gru"])

    # Create model
    model = model_class(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Determine output path
    if output_path is None:
        output_path = str(Path(model_path).with_suffix(".onnx"))

    # Create dummy input
    dummy_input = torch.randn(1, input_dim)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,  # type: ignore[arg-type]
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Model exported to: {output_path}")

    # Verify export
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
    except ImportError:
        print("Warning: onnx not installed, skipping verification")

    return output_path


def quantize_onnx_model(
    onnx_path: str,
    output_path: str | None = None,
    quantization_mode: str = "int8",
) -> str:
    """
    Quantize ONNX model for edge deployment.

    Args:
        onnx_path: Path to ONNX model
        output_path: Output path for quantized model
        quantization_mode: Quantization mode ('int8', 'uint8')

    Returns:
        Path to quantized ONNX model
    """
    try:
        from onnxruntime.quantization import quantize_dynamic
    except ImportError:
        print("Error: onnx and onnxruntime required for quantization")
        return onnx_path

    # Determine output path
    if output_path is None:
        output_path = str(Path(onnx_path).with_suffix(f".quant.{quantization_mode}.onnx"))

    # Quantize
    quantize_dynamic(
        onnx_path,
        output_path,
        weight_type=quantization_mode,
    )

    print(f"Quantized model saved to: {output_path}")
    return output_path


def main():
    """CLI for ONNX export."""
    import argparse

    parser = argparse.ArgumentParser(description="Export TSL model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--output", type=str, help="Output ONNX path")
    parser.add_argument("--type", type=str, default="gru", choices=MODEL_CLASSES.keys())
    parser.add_argument("--input-dim", type=int, default=162)
    parser.add_argument("--num-classes", type=int, default=51)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true", help="Quantize after export")
    parser.add_argument("--quant-mode", type=str, default="int8", choices=["int8", "uint8"])

    args = parser.parse_args()

    # Export
    onnx_path = export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        model_type=args.type,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        opset_version=args.opset,
    )

    # Quantize if requested
    if args.quantize:
        quantize_onnx_model(onnx_path, quantization_mode=args.quant_mode)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
