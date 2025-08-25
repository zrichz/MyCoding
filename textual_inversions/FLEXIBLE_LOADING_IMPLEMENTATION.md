# Flexible TI File Loading - Implementation Summary

## Problem Addressed
The original script failed with `KeyError: 'string_to_param'` when loading certain textual inversion (.pt) files that use different internal key structures.

## Solution Implemented
Added comprehensive flexible loading functionality that can handle various TI file formats:

### New Functions Added

#### 1. `analyze_pt_file(filepath)`
- **Purpose**: Quickly analyze if a .pt file contains textual inversion data
- **Features**: 
  - Detects multiple TI file patterns
  - Provides informative feedback about file structure
  - Returns boolean for compatibility check

#### 2. `find_embedding_tensor(data)`
- **Purpose**: Search for embedding tensors using multiple patterns
- **Patterns Supported**:
  - `['string_to_param', '*']` (standard format)
  - `['emb_params', '*']` (alternative format)
  - `['embeddings', '*']` (common alternative)
  - `['string_to_param']` (direct tensor)
  - `['embeddings']` (direct tensor)
  - `['*']` (wildcard key)

#### 3. `search_all_tensors(data)`
- **Purpose**: Recursively search for 2D tensors in any part of the file
- **Features**:
  - Deep traversal of nested dictionaries
  - Automatic tensor validation (shape, dtype)
  - Finds tensors regardless of key names

#### 4. `load_ti_file_flexible(filepath)`
- **Purpose**: Main loading function with automatic format detection and normalization
- **Features**:
  - Handles raw tensor files
  - Handles dictionary files with various structures
  - Creates standardized output format
  - Preserves original data while ensuring compatibility
  - Adds missing metadata (name, step, tokens)

### File Format Support

#### Standard Format
```python
{
    'string_to_param': {'*': tensor},
    'string_to_token': {'*': token_id},
    'name': 'embedding_name',
    'step': training_step
}
```

#### Alternative Formats Supported
- Files with `emb_params` instead of `string_to_param`
- Files with `embeddings` key
- Files with direct tensors (no nested dictionaries)
- Raw tensor files without dictionary structure
- Files with non-standard key names

### Integration with Main Script

#### Before (Rigid Loading)
```python
data = torch.load(filepath, map_location='cpu')
tensor = data['string_to_param']['*']  # Would fail on alternative formats
```

#### After (Flexible Loading)
```python
flexible_result = load_ti_file_flexible(filepath)
if flexible_result:
    data, tensor, tensor_path = flexible_result
    # data is now guaranteed to have standard structure
    # tensor is the actual embedding tensor
    # tensor_path describes where the tensor was found
```

### Benefits

1. **Compatibility**: Works with TI files from different tools and training methods
2. **Robustness**: Graceful handling of unexpected file structures
3. **Transparency**: Clear reporting of what was found and how it was processed
4. **Standardization**: Converts all formats to a consistent internal structure
5. **Backward Compatibility**: Still works with standard format files

### Error Handling

- Clear error messages when files cannot be processed
- Informative feedback about file structure
- Graceful fallback attempts with multiple loading strategies
- Educational output to help users understand file compatibility

### Usage

The flexible loading is now automatically used in the main script:
1. File selection (unchanged)
2. Automatic format analysis and compatibility check
3. Flexible loading with structure normalization
4. Standard processing continues with guaranteed format

This implementation ensures that the script works with a much wider variety of textual inversion files while maintaining all existing functionality.
