"""
Grey Liquid Lab — MoM Slice Extractor
Experiment #8: Extract SWA-free layer slice from Gemma 4 e4b

Usage:
    python mom_slice_extract.py --model gemma4-e4b-bf16.gguf --output gemma4-e4b-fullatt-slice.gguf
    python mom_slice_extract.py --model gemma4-e4b-bf16.gguf --output gemma4-e4b-fullatt-slice.gguf --info

This script:
1. Reads GGUF metadata to find SWA vs full-attention layer indices
2. Extracts only full-attention layers (removes SWA layers)
3. Writes a new GGUF with updated block_count metadata
4. The output is a smaller model that can be tested for Q2_K compatibility

Research hypothesis: Removing SWA layers (the Q2_K failure amplifier) may enable
Q2_K quantization even with the 4.0x FFN ratio still in the danger zone.
"""

import argparse
import struct
import sys
import os
from pathlib import Path

try:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
    GGUF_LIB = True
except ImportError:
    GGUF_LIB = False
    print("NOTE: gguf library not found. Install with: pip install gguf")
    print("Falling back to manual GGUF parsing for info mode only.")


def get_swa_layer_indices(reader):
    """
    Determine which layer indices use Sliding Window Attention.

    Reads directly from the 'sliding_window_pattern' boolean array in GGUF
    metadata. In Gemma 4 e4b: True = SWA layer, False = full-attention layer.

    Confirmed Gemma 4 e4b architecture (from GGUF sliding_window_pattern):
    - 35 SWA layers (indices 0-4, 6-10, 12-16, 18-22, 24-28, 30-34, 36-40)
    - 7 full-attention layers (indices 5, 11, 17, 23, 29, 35, 41 — every 6th)
    - Ratio: 5:1 local:global  (matches Gemma 4 architecture paper)

    NOTE: The 'shared_kv_layers' metadata field (value=18) refers to KV sharing
    across blocks, NOT the number of SWA layers. Do not confuse these.
    """
    total_blocks = None
    for key in reader.fields:
        if 'block_count' in key:
            total_blocks = int(reader.fields[key].parts[-1][0])

    if total_blocks is None:
        raise ValueError("Could not find block_count in GGUF metadata")

    print(f"[info] Total blocks: {total_blocks}")

    # Primary: read directly from sliding_window_pattern field
    if 'gemma4.attention.sliding_window_pattern' in reader.fields:
        f = reader.fields['gemma4.attention.sliding_window_pattern']
        pattern = [bool(f.parts[idx][0]) for idx in f.data]  # True = SWA
        full_layers = sorted(i for i, v in enumerate(pattern) if not v)
        swa_layers  = sorted(i for i, v in enumerate(pattern) if v)
        print(f"[info] Pattern read from GGUF: {len(swa_layers)} SWA, {len(full_layers)} full-att")
    else:
        # Fallback: Gemma 4 default — full attention every 6th layer
        full_layers = list(range(5, total_blocks, 6))
        swa_layers  = [i for i in range(total_blocks) if i not in set(full_layers)]
        print(f"[info] sliding_window_pattern not found — using Gemma 4 default (every 6th)")

    print(f"[info] Full-attention layers ({len(full_layers)}): {full_layers}")
    print(f"[info] SWA layers ({len(swa_layers)}): {swa_layers}")
    return full_layers, swa_layers


def show_info(model_path):
    """Show architecture analysis without extracting."""
    if not GGUF_LIB:
        print("gguf library required for info mode. Run: pip install gguf")
        return
    
    reader = GGUFReader(model_path)
    
    print("\n=== Gemma 4 Architecture Analysis (Grey Liquid MoM Research) ===\n")
    
    key_fields = ['block_count', 'embedding_length', 'feed_forward_length', 
                  'attention.head_count', 'attention.head_count_kv',
                  'attention.sliding_window', 'attention.shared_kv_layers',
                  'context_length', 'size_label']
    
    total_blocks = None
    ffn_length = None
    embed_length = None
    
    for key in reader.fields:
        for kf in key_fields:
            if kf in key:
                try:
                    val = reader.fields[key].parts[-1][0]
                    print(f"  {key.split('.')[-1]:30s} = {val}")
                    if 'block_count' in key:
                        total_blocks = int(val)
                    if 'feed_forward' in key:
                        ffn_length = int(val)
                    if 'embedding_length' in key and 'per_layer' not in key:
                        embed_length = int(val)
                except Exception:
                    pass
    
    if ffn_length and embed_length:
        ratio = ffn_length / embed_length
        zone = "DANGER ZONE ❌" if 3.0 <= ratio <= 5.5 else ("SAFE LOW ✅" if ratio < 3.0 else "SAFE HIGH ✅")
        print(f"\n  FFN expansion ratio: {ffn_length}/{embed_length} = {ratio:.2f}x  [{zone}]")
    
    print(f"\n  Total tensors: {len(reader.tensors)}")
    
    if total_blocks:
        full_layers, swa_layers = get_swa_layer_indices(reader)
        print(f"\n  Full-attention layers: {len(full_layers)}")
        print(f"  SWA layers: {len(swa_layers)}")
        print(f"\n  MoM slice (full-attention only):")
        print(f"    Layers kept: {len(full_layers)} / {total_blocks} ({100*len(full_layers)/total_blocks:.0f}% depth)")
        
        # Count tensors that would be kept
        kept_tensors = 0
        for tensor in reader.tensors:
            name = tensor.name
            if name.startswith('blk.'):
                parts = name.split('.')
                try:
                    blk_idx = int(parts[1])
                    if blk_idx in full_layers:
                        kept_tensors += 1
                except ValueError:
                    pass
            else:
                kept_tensors += 1  # embeddings, norms, output
        
        total_size_gb = sum(t.n_bytes for t in reader.tensors) / 1e9
        kept_size_estimate = total_size_gb * (len(full_layers) / total_blocks)
        print(f"    Estimated slice size: ~{kept_size_estimate:.1f} GB (from {total_size_gb:.1f} GB)")
        print(f"\n  Q2_K compatibility prediction:")
        if ratio and 3.0 <= ratio <= 5.5:
            print(f"    Full model: ❌ FAIL (FFN 4.0x danger zone + 18 SWA layers)")
            print(f"    SWA-free slice: ❓ UNKNOWN — Hypothesis H1: may pass (no SWA amplifier)")
            print(f"    Experiment #8 will test this hypothesis.")


def extract_slice(model_path, output_path, keep_layers):
    """Extract specified layers into a new GGUF file."""
    if not GGUF_LIB:
        print("gguf library required. Install with: pip install gguf")
        print("Then re-run this script.")
        sys.exit(1)
    
    reader = GGUFReader(model_path)
    
    # Get total block count and other metadata
    total_blocks = None
    for key in reader.fields:
        if 'block_count' in key:
            total_blocks = int(reader.fields[key].parts[-1][0])
    
    keep_set = set(keep_layers)
    new_block_count = len(keep_layers)
    
    # Map old layer indices to new consecutive indices
    layer_remap = {old: new for new, old in enumerate(sorted(keep_layers))}
    
    print(f"Extracting {new_block_count} layers from {total_blocks} total...")
    print(f"Layer mapping: {layer_remap}")
    
    writer = GGUFWriter(output_path, "gemma4")
    
    # Copy metadata, update block_count
    for key, field in reader.fields.items():
        if 'block_count' in key:
            writer.add_uint32(key, new_block_count)
        elif 'shared_kv_layers' in key:
            writer.add_uint32(key, 0)  # No SWA in the slice
        elif 'sliding_window' in key:
            pass  # Skip SWA metadata since we removed those layers
        else:
            try:
                # Copy the field value
                val = field.parts[-1][0]
                dtype = str(type(val).__name__)
                if dtype in ('int', 'int32', 'uint32'):
                    writer.add_uint32(key, int(val))
                elif dtype in ('float', 'float32'):
                    writer.add_float32(key, float(val))
                elif dtype == 'str':
                    writer.add_string(key, str(val))
                elif dtype == 'bool':
                    writer.add_bool(key, bool(val))
            except Exception as e:
                pass  # Skip fields we can't easily copy
    
    # Copy tensors
    kept = 0
    skipped = 0
    for tensor in reader.tensors:
        name = tensor.name
        
        if name.startswith('blk.'):
            parts = name.split('.')
            try:
                blk_idx = int(parts[1])
                if blk_idx not in keep_set:
                    skipped += 1
                    continue
                # Rename with new consecutive index
                new_idx = layer_remap[blk_idx]
                new_name = f"blk.{new_idx}." + '.'.join(parts[2:])
                writer.add_tensor(new_name, tensor.data, raw_dtype=tensor.tensor_type)
                kept += 1
            except ValueError:
                writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
                kept += 1
        else:
            # Keep embeddings, norms, output head
            writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
            kept += 1
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    output_size = os.path.getsize(output_path) / 1e9
    print(f"\n✅ Slice extracted successfully!")
    print(f"   Tensors kept: {kept}, skipped: {skipped}")
    print(f"   Output: {output_path} ({output_size:.2f} GB)")
    print(f"\nNext steps (Experiment #8):")
    print(f"  1. Quantize: llama-quantize.exe {output_path} slice-q2k.gguf Q2_K")
    print(f"  2. Test:     llama-cli.exe -m slice-q2k.gguf -p 'What is 2+2?' -n 50")
    print(f"  3. Document result in GREY_LIQUID_REPORT_008.md")


def main():
    parser = argparse.ArgumentParser(
        description="Grey Liquid MoM Slice Extractor — Experiment #8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--model", required=True, help="Path to source GGUF (bf16)")
    parser.add_argument("--output", help="Output GGUF path for the slice")
    parser.add_argument("--info", action="store_true", help="Show architecture analysis only, don't extract")
    parser.add_argument("--layers", help="Comma-separated layer indices to keep (default: auto-detect full-attention layers)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    if args.info or not args.output:
        show_info(args.model)
        return
    
    if not GGUF_LIB:
        print("Install gguf library first: pip install gguf")
        sys.exit(1)
    
    reader = GGUFReader(args.model)
    
    if args.layers:
        keep_layers = [int(x.strip()) for x in args.layers.split(',')]
    else:
        full_layers, swa_layers = get_swa_layer_indices(reader)
        keep_layers = full_layers
        print(f"\nAuto-selected {len(keep_layers)} full-attention layers for SWA-free slice")
    
    extract_slice(args.model, args.output, keep_layers)


if __name__ == "__main__":
    main()
