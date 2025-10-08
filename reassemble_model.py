#!/usr/bin/env python3
"""
Script to manually reassemble split model files.
This is useful if you want to create the full zip file manually.
"""

import os
import glob
import sys

def reassemble_model(base_filename="PAD-UFES-20.zip"):
    """Reassemble split model files into original zip file."""
    
    # Find all split files
    split_pattern = f"{base_filename}.part*"
    split_files = sorted(glob.glob(split_pattern))
    
    if not split_files:
        print(f"No split files found matching pattern: {split_pattern}")
        return False
    
    print(f"Found {len(split_files)} split files:")
    for f in split_files:
        size = os.path.getsize(f) / (1024 * 1024)  # Size in MB
        print(f"  {f} ({size:.1f} MB)")
    
    # Reassemble files
    output_file = base_filename
    print(f"\nReassembling into: {output_file}")
    
    try:
        with open(output_file, 'wb') as outfile:
            for split_file in split_files:
                print(f"Reading: {split_file}")
                with open(split_file, 'rb') as infile:
                    outfile.write(infile.read())
        
        # Verify the output file
        output_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nSuccess! Created {output_file} ({output_size:.1f} MB)")
        
        # Verify it's a valid zip file
        import zipfile
        try:
            with zipfile.ZipFile(output_file, 'r') as zf:
                file_list = zf.namelist()
                print(f"Zip file contains {len(file_list)} files:")
                for f in file_list:
                    print(f"  {f}")
            return True
        except zipfile.BadZipFile:
            print("Warning: The reassembled file is not a valid zip file!")
            return False
            
    except Exception as e:
        print(f"Error reassembling files: {e}")
        return False

if __name__ == "__main__":
    base_file = "PAD-UFES-20.zip"
    if len(sys.argv) > 1:
        base_file = sys.argv[1]
    
    success = reassemble_model(base_file)
    sys.exit(0 if success else 1)