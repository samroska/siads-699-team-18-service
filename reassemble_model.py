#!/usr/bin/env python3
import os
import glob
import sys

def reassemble_model(base_filename="PAD-UFES-20.zip"):
    """Reassemble split model files into original zip file."""
    
 
    split_pattern = f"{base_filename}.part*"
    split_files = sorted(glob.glob(split_pattern))
    
    if not split_files:
        print(f"No split files found matching pattern: {split_pattern}")
        return False
    
    print(f"Found {len(split_files)} split files:")
    for f in split_files:
        size = os.path.getsize(f) / (1024 * 1024)   
        print(f"  {f} ({size:.1f} MB)")
    
  
    output_file = base_filename
    print(f"\nReassembling into: {output_file}")

    """
    This script is now deprecated. Model loading is handled automatically from PAD-UFES-20.keras.zip.
    No manual reassembly of split files is required.
    """
