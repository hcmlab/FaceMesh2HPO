import gzip
import os

import onnx
import onnxoptimizer
from onnx_ir.passes.common import DeduplicateInitializersPass
from onnx_ir import load, save


def optimize_and_shrink_onnx(input_path: str, output_path: str):
    # Load the model
    model = onnx.load(input_path)

    # Get list of optimization passes
    passes = onnxoptimizer.get_available_passes()
    print("Available passes:", passes)

    # Typical passes for size reduction
    passes_to_run = [
        "eliminate_deadend",        # remove unreachable nodes
        "eliminate_identity",       # remove Identity nodes
        "fuse_add_bias_into_conv",  # fuse conv + add(bias)
        "fuse_bn_into_conv",        # fuse BatchNorm into Conv
        "eliminate_duplicate_initializer",  # deduplicate repeated weights
    ]

    # Apply the optimization passes
    optimized_model = onnxoptimizer.optimize(model, passes_to_run)

    # Save the optimized model
    onnx.save(optimized_model, output_path)
    print_file_size_reduction(input_path, output_path)


def deduplicate_onnx_ir_style(input_path: str, output_path: str):
    model = load(input_path)

    # Run deduplication pass
    pass_obj = DeduplicateInitializersPass(size_limit=1024)
    result = pass_obj(model)
    if result.modified:
        print("Deduplicated initializers (removed duplicates).")
    else:
        print("No duplicate initializers found.")
    save(result.model, output_path)
    print_file_size_reduction(input_path, output_path)


def compress_onnx_to_gzip(onnx_path: str, gz_path: str, compresslevel: int = 9):
    """
    Compress an ONNX file to .gz format.

    Args:
        onnx_path: path to the ONNX file (e.g., "model_optimized.onnx")
        gz_path:   path to the compressed output (e.g., "model_optimized.onnx.gz")
        compresslevel: gzip compression level (0–9, 9 is max)
    """
    with open(onnx_path, "rb") as f_in:
        with gzip.open(gz_path, "wb", compresslevel=compresslevel) as f_out:
            f_out.write(f_in.read())
    print_file_size_reduction(onnx_path, gz_path)


def print_file_size_reduction(input_path: str, output_path: str):
    # Optional: print size info
    size_onnx = os.path.getsize(input_path)
    size_gz = os.path.getsize(output_path)
    print(f"Input size:  {size_onnx / 1024 / 1024:.2f} MB")
    print(f"Output size:  {size_gz / 1024 / 1024:.2f} MB")
    print(f"Reduction:  {100 * (1 - size_gz / size_onnx):.1f}%")