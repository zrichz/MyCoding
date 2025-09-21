# Quick Start Guide - Enhanced TI CHANGER

## Your Problem is SOLVED! 🎉

The script now handles textual inversion files with different internal structures, including the ones that were giving you `KeyError: 'string_to_param'` errors.

## What Changed

✅ **Flexible file loading** - Works with various TI file formats  
✅ **Automatic format detection** - No manual configuration needed  
✅ **Better error messages** - Clear feedback when files can't be processed  
✅ **All features preserved** - Every option still works as expected  

## How to Use

### Option 1: Run with Windows Launcher
```bash
run_TI_CHANGER.bat
```

### Option 2: Run with Cross-Platform Launcher  
```bash
python run_TI_CHANGER.py
```

### Option 3: Run Directly
```bash
c:\MyPythonCoding\MyCoding\image_processors\.venv\Scripts\python.exe TI_CHANGER_MULTIPLE_2024_10_22.py
```

## Testing Your Files

1. **Run the script** using any method above
2. **Select your problematic .pt file** when prompted
3. **Watch the analysis output** - it will tell you if the file is compatible
4. **Choose any option** (1-6) - they all work with flexible loading now

## Expected Output for Compatible Files

```
Analyzing selected file...
✅ TI loaded successfully from your_file.pt
📍 Embedding tensor found at: [path description]  
📊 Final data structure keys: [list of keys]
📊 Tensor shape: [dimensions]
📏 Dimensions: [number]
🔢 Number of vectors: [count]
```

## If a File Still Won't Load

The script will now try multiple loading strategies:

1. **Standard format** (string_to_param -> *)
2. **Alternative formats** (emb_params, embeddings, etc.)  
3. **Raw tensor detection** (direct tensor files)
4. **Deep search** (tensors anywhere in the file structure)

If none work, you'll get clear feedback about why the file couldn't be processed.

## All 6 Options Work

- **Option 1**: Smooth/blur vectors (Gaussian kernel)
- **Option 2**: Condense to single mean vector  
- **Option 3**: Decimate with zeros
- **Option 4**: Divide by scalar
- **Option 5**: Roll/shift vectors
- **Option 6**: Extract individual vectors to separate files ⭐

## Option 6 is Perfect for Your Use Case

Since you wanted to "load a .pt file, show the number of vectors present, and allow saving of each of the eight vectors as an individual .pt file" - **Option 6 does exactly this!**

It will:
1. ✅ Show you the number of vectors in your file
2. ✅ Extract each vector to a separate .pt file  
3. ✅ Save them with descriptive names like `filename_vector_01.pt`, `filename_vector_02.pt`, etc.
4. ✅ Work with any compatible TI file format

## Need Help?

Check these documentation files:
- `ENHANCEMENT_SUMMARY.md` - Complete overview of changes
- `FLEXIBLE_LOADING_IMPLEMENTATION.md` - Technical details
- `README_Vector_Extraction.md` - Option 6 detailed guide

The script is now much more robust and should handle your TI files without any issues! 🚀
