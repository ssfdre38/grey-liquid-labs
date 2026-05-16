"""
Grey Liquid Lab — De-SWA Patcher
Experiment #8: Remove Sliding Window Attention from Gemma 4 e4b bf16

This script:
1. Copies the source GGUF (14GB) to a new destination
2. Patches ONLY the SWA metadata fields in-place (39 bytes changed in 14GB)
   - gemma4.attention.shared_kv_layers: 18 → 0
   - gemma4.attention.sliding_window_pattern: 35 True → 0 (False)

This forces llama.cpp to treat ALL 42 layers as full-attention.
The weight tensors are untouched — this is purely a metadata patch.

Research hypothesis (H1): Removing SWA metadata allows Q2_K to work on Gemma 4.
If coherent output is produced at Q2_K, sub-3-bit barrier on Gemma 4 is broken.

Usage:
    python deswa_patch.py
"""

import struct
import shutil
import numpy as np
from pathlib import Path

try:
    from gguf import GGUFReader
except ImportError:
    print("Error: gguf library not found. Run: pip install gguf")
    raise SystemExit(1)

SOURCE = Path(r"C:\Users\admin\gemma4-turbo-family\gemma4-e4b-bf16.gguf")
DEST   = Path(r"C:\Users\admin\gemma4-turbo-family\gemma4-e4b-bf16-deswa.gguf")


def collect_patch_ops(source: Path):
    """Read source GGUF and return list of (byte_offset, new_bytes) patch operations."""
    print(f"[deswa] Reading source GGUF metadata: {source.name}")
    r = GGUFReader(str(source))
    base = r.data.ctypes.data

    patches = []

    # 1. Patch gemma4.attention.shared_kv_layers: uint32 18 → 0
    f = r.fields['gemma4.attention.shared_kv_layers']
    offset = f.parts[f.data[0]].ctypes.data - base
    current = int(r.data[offset:offset + 4].view(np.uint32)[0])
    print(f"[deswa] shared_kv_layers @ byte {offset}: {current} → 0")
    patches.append((offset, struct.pack('<I', 0)))

    # 2. Patch gemma4.attention.sliding_window_pattern: True → False for all SWA layers
    f = r.fields['gemma4.attention.sliding_window_pattern']
    swa_count = 0
    full_count = 0
    for data_idx, part_idx in enumerate(f.data):
        offset = f.parts[part_idx].ctypes.data - base
        current_val = int(r.data[offset])
        if current_val == 1:  # True = SWA layer
            patches.append((offset, b'\x00'))
            swa_count += 1
        else:
            full_count += 1

    print(f"[deswa] sliding_window_pattern: {swa_count} SWA layers → False, {full_count} full-att layers unchanged")
    print(f"[deswa] Total patches: {len(patches)} byte-level writes")
    return patches


def apply_patches(dest: Path, patches: list):
    """Apply patch operations to destination file."""
    print(f"[deswa] Applying {len(patches)} patches to: {dest.name}")
    with open(dest, 'r+b') as f:
        for offset, data in patches:
            f.seek(offset)
            f.write(data)
    print("[deswa] All patches applied.")


def verify_patches(dest: Path):
    """Verify the patches were applied correctly."""
    print("[deswa] Verifying patches...")
    r = GGUFReader(str(dest))

    # Check shared_kv_layers
    f = r.fields['gemma4.attention.shared_kv_layers']
    val = int(f.parts[f.data[0]][0])
    ok_kv = val == 0
    print(f"  shared_kv_layers = {val} {'✅' if ok_kv else '❌ FAIL (expected 0)'}")

    # Check sliding_window_pattern: all should be False
    f = r.fields['gemma4.attention.sliding_window_pattern']
    pattern = [int(f.parts[idx][0]) for idx in f.data]
    all_false = all(v == 0 for v in pattern)
    swa_remaining = sum(pattern)
    print(f"  sliding_window_pattern: {len(pattern)} layers, {swa_remaining} still SWA {'✅ all clear' if all_false else '❌ FAIL'}")

    return ok_kv and all_false


def main():
    print("=" * 60)
    print("Grey Liquid Lab — De-SWA Patcher (Experiment #8)")
    print("=" * 60)

    if not SOURCE.exists():
        print(f"Error: Source not found: {SOURCE}")
        raise SystemExit(1)

    source_size_gb = SOURCE.stat().st_size / 1e9
    print(f"\nSource: {SOURCE.name} ({source_size_gb:.2f} GB)")
    print(f"Dest:   {DEST.name}")

    # Step 1: Collect patch operations from source
    patches = collect_patch_ops(SOURCE)

    # Step 2: Copy source to destination
    if DEST.exists():
        print(f"\n[deswa] Destination already exists — skipping copy, applying patches directly")
    else:
        print(f"\n[deswa] Copying {source_size_gb:.2f} GB file... (this takes a few minutes)")
        shutil.copy2(SOURCE, DEST)
        copied_gb = DEST.stat().st_size / 1e9
        print(f"[deswa] Copy complete: {copied_gb:.2f} GB")

    # Step 3: Apply patches in-place to destination
    apply_patches(DEST, patches)

    # Step 4: Verify
    if verify_patches(DEST):
        print("\n✅ De-SWA patch successful!")
        print(f"\nDe-SWA'd model: {DEST}")
        print("\nNext steps (Experiment #8):")
        print(f"  cd C:\\Users\\admin\\gemma4-turbo-family")
        print(f"  llama-cpp\\llama-quantize.exe gemma4-e4b-bf16-deswa.gguf gemma4-e4b-q2k-deswa.gguf Q2_K")
        print(f"  llama-cpp\\llama-cli.exe -m gemma4-e4b-q2k-deswa.gguf -p \"What is 2+2?\" -n 50 --no-warmup")
    else:
        print("\n❌ Verification failed — check output above")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
