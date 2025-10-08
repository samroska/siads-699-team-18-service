# Model File Setup

## Large Model File Handling

Due to GitHub's file size limitations, the ML model is split into smaller parts:

- `PAD-UFES-20.zip.partaa` (50 MB)
- `PAD-UFES-20.zip.partab` (11 MB)

### Automatic Assembly
The application automatically reassembles these files when loading the model. No manual intervention is required.

### Manual Assembly (Optional)
If you need to manually create the full zip file:

```bash
python reassemble_model.py
```

This will create `PAD-UFES-20.zip` from the split parts.

### For Developers
To split a new model file:

```bash
# Split into 50MB chunks
split -b 50m your-model.zip your-model.zip.part

# List the created parts
ls -lh your-model.zip.part*
```

## Files
- `PAD-UFES-20.zip.partaa` - First part of the model (50MB)
- `PAD-UFES-20.zip.partab` - Second part of the model (~11MB)
- `reassemble_model.py` - Script to manually reassemble split files
- The full `PAD-UFES-20.zip` should not be committed to git