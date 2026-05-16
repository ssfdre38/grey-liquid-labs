"""
Grey Liquid Lab -- MoM Slice Extractor
Experiments #8 / #8b: Extract layer slices from Gemma 4 e4b for Q2_K testing

Usage:
    # Show architecture info
    python mom_slice_extract.py --model gemma4-e4b-bf16.gguf --info

    # Extract 7 full-attention layers (Experiment #8 approach -- FAILED, shape mismatch)
    python mom_slice_extract.py --model gemma4-e4b-bf16.gguf --output gemma4-fullatt.gguf

    # Extract 35 SWA-only layers (Experiment #8b -- pending)
    python mom_slice_extract.py --model gemma4-e4b-bf16.gguf --output gemma4-swa-only.gguf --swa-only

    # Extract custom layer list
    python mom_slice_extract.py --model gemma4-e4b-bf16.gguf --output custom.gguf --layers 0,1,2,3,4

Confirmed Gemma 4 e4b architecture (from sliding_window_pattern field, Exp #8):
  Full-attention layers: 7  (indices 5,11,17,23,29,35,41 -- every 6th)
  SWA layers:           35  (all others -- 5:1 local:global ratio)
  Full-att Q/K shape:   [2560, 4096]  (8 heads x 512 head_dim)
  SWA Q/K shape:        [2560, 2048]  (8 heads x 256 head_dim_swa)
  shared_kv_layers=18 -> measures KV sharing, NOT SWA count
"""

import argparse
import sys
import os
import numpy as np

try:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
    GGUF_LIB = True
except ImportError:
    GGUF_LIB = False
    print("NOTE: gguf library not found. Install with: pip install gguf")


def get_swa_layer_indices(reader):
    """
    Read sliding_window_pattern directly from GGUF metadata.
    Returns (full_attention_layers, swa_layers) as sorted index lists.

    Confirmed Gemma 4 e4b: 35 SWA + 7 full-att (5:1 local:global ratio).
    NOTE: shared_kv_layers=18 is KV sharing, not SWA count -- do not confuse.
    """
    total_blocks = None
    for key in reader.fields:
        if 'block_count' in key:
            total_blocks = int(reader.fields[key].parts[-1][0])

    if total_blocks is None:
        raise ValueError("Could not find block_count in GGUF metadata")

    if 'gemma4.attention.sliding_window_pattern' in reader.fields:
        f = reader.fields['gemma4.attention.sliding_window_pattern']
        pattern = [bool(f.parts[idx][0]) for idx in f.data]  # True = SWA
        full_layers = sorted(i for i, v in enumerate(pattern) if not v)
        swa_layers  = sorted(i for i, v in enumerate(pattern) if v)
    else:
        # Fallback: Gemma 4 default -- full attention every 6th layer
        full_layers = list(range(5, total_blocks, 6))
        swa_layers  = [i for i in range(total_blocks) if i not in set(full_layers)]
        print("[info] sliding_window_pattern not found -- using Gemma 4 default (every 6th)")

    return full_layers, swa_layers


def copy_field(writer, key, field):
    """Copy a single GGUF metadata field to writer with correct type handling."""
    try:
        field_type = field.types[0]

        if field_type == GGUFValueType.ARRAY:
            elem_type = field.types[1]
            if elem_type == GGUFValueType.STRING:
                values = [bytes(field.parts[idx]).decode('utf-8') for idx in field.data]
            elif elem_type == GGUFValueType.FLOAT32:
                values = [float(field.parts[idx][0]) for idx in field.data]
            elif elem_type in (GGUFValueType.INT32, GGUFValueType.UINT32):
                values = [int(field.parts[idx][0]) for idx in field.data]
            elif elem_type == GGUFValueType.BOOL:
                values = [bool(field.parts[idx][0]) for idx in field.data]
            else:
                return  # Skip unknown array element types
            writer.add_array(key, values)

        elif field_type == GGUFValueType.UINT32:
            writer.add_uint32(key, int(field.parts[-1][0]))
        elif field_type == GGUFValueType.INT32:
            writer.add_int32(key, int(field.parts[-1][0]))
        elif field_type == GGUFValueType.FLOAT32:
            writer.add_float32(key, float(field.parts[-1][0]))
        elif field_type == GGUFValueType.STRING:
            val = bytes(field.parts[-1]).decode('utf-8')
            writer.add_string(key, val)
        elif field_type == GGUFValueType.BOOL:
            writer.add_bool(key, bool(field.parts[-1][0]))

    except Exception:
        pass  # Skip fields that cannot be copied


def show_info(model_path):
    """Print architecture analysis without extracting."""
    if not GGUF_LIB:
        print("gguf library required. Run: pip install gguf")
        return

    reader = GGUFReader(model_path)
    full_layers, swa_layers = get_swa_layer_indices(reader)
    total_blocks = len(full_layers) + len(swa_layers)

    ffn_length = embed_length = None
    key_len = key_len_swa = None

    print("\n=== Gemma 4 Architecture Analysis (Grey Liquid MoM Research) ===\n")
    for key, field in reader.fields.items():
        if any(k in key for k in ['block_count', 'embedding_length', 'feed_forward_length',
                                   'head_count', 'sliding_window', 'shared_kv_layers',
                                   'context_length', 'size_label', 'key_length', 'value_length',
                                   'rope.freq_base', 'dimension_count']):
            try:
                if field.types[0] == GGUFValueType.ARRAY:
                    if 'sliding_window_pattern' in key:
                        f = field
                        n_swa = sum(bool(f.parts[i][0]) for i in f.data)
                        n_full = len(f.data) - n_swa
                        print(f"  {'sliding_window_pattern':30s} = [{n_full} full-att, {n_swa} SWA] (array len {len(f.data)})")
                else:
                    val = field.parts[-1][0]
                    short = key.split('.')[-1]
                    print(f"  {short:30s} = {val}")
                    if 'feed_forward_length' in key:
                        ffn_length = int(val)
                    if 'embedding_length' in key and 'per_layer' not in key and 'swa' not in key:
                        embed_length = int(val)
                    if key == 'gemma4.attention.key_length':
                        key_len = int(val)
                    if 'key_length_swa' in key:
                        key_len_swa = int(val)
            except Exception:
                pass

    if ffn_length and embed_length:
        ratio = ffn_length / embed_length
        zone = "DANGER ZONE (FAIL)" if 3.0 <= ratio <= 5.5 else ("SAFE LOW" if ratio < 3.0 else "SAFE HIGH")
        print(f"\n  FFN ratio: {ffn_length}/{embed_length} = {ratio:.2f}x  [{zone}]")

    if key_len and key_len_swa:
        print(f"\n  Layer distribution:")
        print(f"    Full-attention: {len(full_layers)} layers  {full_layers}")
        print(f"    SWA:            {len(swa_layers)} layers  (5:1 local:global)")
        print(f"    Q/K shape full: [2560, {8*key_len}]  (8 heads x {key_len})")
        print(f"    Q/K shape SWA:  [2560, {8*key_len_swa}]  (8 heads x {key_len_swa})")

    print(f"\n  Total tensors: {len(reader.tensors)}")
    total_size = sum(t.n_bytes for t in reader.tensors) / 1e9
    full_att_frac = len(full_layers) / total_blocks
    swa_frac = len(swa_layers) / total_blocks
    print(f"\n  Slice size estimates (from {total_size:.1f} GB total):")
    print(f"    Full-att only ({len(full_layers)} layers): ~{total_size*full_att_frac:.2f} GB")
    print(f"    SWA only ({len(swa_layers)} layers):       ~{total_size*swa_frac:.2f} GB")
    print(f"\n  Experiment status:")
    print(f"    Exp #8  (full-att slice + de-SWA patch): FAILED -- shape mismatch")
    print(f"    Exp #8b (SWA-only slice Q2_K):           PENDING -- use --swa-only")


def extract_slice(model_path, output_path, keep_layers, swa_only=False):
    """
    Extract specified layers into a new GGUF file.

    swa_only=True: metadata configured for an all-SWA model (Exp #8b).
    swa_only=False: metadata configured for a full-attention-only model.
    """
    if not GGUF_LIB:
        print("gguf library required. Install with: pip install gguf")
        sys.exit(1)

    reader = GGUFReader(model_path)
    full_layers, swa_layers = get_swa_layer_indices(reader)
    total_blocks = len(full_layers) + len(swa_layers)

    keep_set = set(keep_layers)
    new_block_count = len(keep_layers)
    layer_remap = {old: new for new, old in enumerate(sorted(keep_layers))}

    # Read SWA-specific values to promote for SWA-only mode
    kl_swa = vl_swa = dc_swa = None
    try:
        kl_swa = int(reader.fields['gemma4.attention.key_length_swa'].parts[-1][0])
        vl_swa = int(reader.fields['gemma4.attention.value_length_swa'].parts[-1][0])
        dc_swa = int(reader.fields['gemma4.rope.dimension_count_swa'].parts[-1][0])
    except Exception:
        pass

    # Read per-layer embedding dimension (Gemma 4: 256)
    per_layer_emb_dim = 256
    try:
        per_layer_emb_dim = int(reader.fields['gemma4.embedding_length_per_layer_input'].parts[-1][0])
    except Exception:
        pass

    mode_label = "SWA-only" if swa_only else "full-attention-only"
    print(f"\nExtracting {new_block_count}-layer {mode_label} slice from {total_blocks}-layer model...")
    print(f"Output: {output_path}")

    SKIP_INTERNAL = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}

    writer = GGUFWriter(output_path, "gemma4")

    for key, field in reader.fields.items():
        if key in SKIP_INTERNAL:
            continue

        if 'block_count' in key:
            writer.add_uint32(key, new_block_count)
            continue

        if 'sliding_window_pattern' in key:
            # All kept layers are the same type
            writer.add_array(key, [swa_only] * new_block_count)
            continue

        if 'shared_kv_layers' in key:
            writer.add_uint32(key, 0)
            continue

        if swa_only:
            # Promote SWA dims as the primary dims since all layers are SWA
            if key == 'gemma4.attention.key_length' and kl_swa is not None:
                writer.add_uint32(key, kl_swa)
                continue
            if key == 'gemma4.attention.value_length' and vl_swa is not None:
                writer.add_uint32(key, vl_swa)
                continue
            if key == 'gemma4.rope.dimension_count' and dc_swa is not None:
                writer.add_uint32(key, dc_swa)
                continue

        copy_field(writer, key, field)

    # Copy tensors
    kept = skipped = 0
    total_t = len(reader.tensors)
    print(f"Copying tensors (this takes several minutes for a 14 GB file)...")
    for i, tensor in enumerate(reader.tensors):
        if i % 100 == 0:
            print(f"  [{i}/{total_t}] {tensor.name}")
        name = tensor.name

        if name.startswith('blk.'):
            parts = name.split('.')
            try:
                blk_idx = int(parts[1])
                if blk_idx not in keep_set:
                    skipped += 1
                    continue
                new_idx = layer_remap[blk_idx]
                new_name = f"blk.{new_idx}." + '.'.join(parts[2:])
                writer.add_tensor(new_name, tensor.data, raw_dtype=tensor.tensor_type)
                kept += 1
            except ValueError:
                writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
                kept += 1
        elif name == 'per_layer_token_embd.weight':
            # Shape: [total_layers * per_layer_emb_dim, vocab_size] — slice axis 0
            slices = []
            for old_idx in sorted(keep_layers):
                start = old_idx * per_layer_emb_dim
                slices.append(tensor.data[start:start + per_layer_emb_dim])
            writer.add_tensor(name, np.concatenate(slices, axis=0), raw_dtype=tensor.tensor_type)
            kept += 1
        elif name == 'per_layer_model_proj.weight':
            # Shape: [embed_size, total_layers * per_layer_emb_dim] — slice axis 1
            slices = []
            for old_idx in sorted(keep_layers):
                start = old_idx * per_layer_emb_dim
                slices.append(tensor.data[:, start:start + per_layer_emb_dim])
            writer.add_tensor(name, np.concatenate(slices, axis=1), raw_dtype=tensor.tensor_type)
            kept += 1
        else:
            writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
            kept += 1

    print("Writing GGUF output file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_size = os.path.getsize(output_path) / 1e9
    print(f"\nDone!")
    print(f"  Tensors kept: {kept}, skipped: {skipped}")
    print(f"  Output: {output_path} ({output_size:.2f} GB)")
    if swa_only:
        print(f"\nNext (Experiment #8b):")
        print(f"  Quantize: llama-quantize.exe {output_path} swa-q2k.gguf Q2_K")
        print(f"  Test:     llama-cli.exe -m swa-q2k.gguf -p \"What is 2+2?\" -n 50")
    else:
        print(f"\nNext (full-att slice):")
        print(f"  Quantize: llama-quantize.exe {output_path} fullatt-q2k.gguf Q2_K")
        print(f"  Test:     llama-cli.exe -m fullatt-q2k.gguf -p \"What is 2+2?\" -n 50")


def main():
    parser = argparse.ArgumentParser(
        description="Grey Liquid MoM Slice Extractor -- Experiments #8 / #8b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--model", required=True, help="Path to source GGUF (bf16)")
    parser.add_argument("--output", help="Output GGUF path for the slice")
    parser.add_argument("--info", action="store_true", help="Show architecture analysis only")
    parser.add_argument("--swa-only", action="store_true",
                        help="Extract SWA layers only (Experiment #8b). Default: full-attention only.")
    parser.add_argument("--layers", help="Comma-separated layer indices to keep (overrides auto-detect)")

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
        swa_only = args.swa_only
    else:
        full_layers, swa_layers = get_swa_layer_indices(reader)
        if args.swa_only:
            keep_layers = swa_layers
            swa_only = True
            print(f"Auto-selected {len(keep_layers)} SWA layers for SWA-only slice (Experiment #8b)")
        else:
            keep_layers = full_layers
            swa_only = False
            print(f"Auto-selected {len(keep_layers)} full-attention layers for full-att-only slice")

    extract_slice(args.model, args.output, keep_layers, swa_only=swa_only)


if __name__ == "__main__":
    main()