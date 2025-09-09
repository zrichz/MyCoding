#!/usr/bin/env python3
"""
DEMONSTRATION: TI_CHANGER_MULTIPLE Menu System

This script shows the new menu structure for comparing TI files.
The user now has TWO ways to access the file comparison feature:

METHOD 1: Direct access from main menu
========================================
When you run TI_CHANGER_MULTIPLE_2024_10_22.py, you'll see:

    MAIN MENU - CHOOSE OPERATION TYPE
    ================================
    1. Process a single TI file (load, analyze, transform)
    2. Compare two TI files (vector-by-vector analysis)  <-- DIRECT ACCESS
    3. Exit

Choose option 2 to compare two files immediately without loading a single file first.

METHOD 2: After loading a single file
=====================================
If you choose option 1 (Process a single TI file), after loading a file you'll see:

    TI CHANGER - MULTI VECTOR FILE OPERATIONS
    =========================================
    1. Apply smoothing to all vectors
    2. Create single mean vector (condensed)
    ...
    14. Average specified vectors and combine with remaining
    15. Compare two TI files (vector-by-vector analysis)  <-- ALSO AVAILABLE HERE

Choose option 15 to compare the loaded file with another file.

FEATURES OF THE COMPARISON TOOL:
===============================
✅ Loads and validates two .pt files
✅ Checks compatibility and trims if needed
✅ Creates 36x24 heatmap visualizations for each vector
✅ Shows difference plots between corresponding vectors
✅ Displays similarity metrics (MSE, MAE, correlation, etc.)
✅ Allows saving average and difference as new .pt files
✅ Proper suffix naming for saved files

USAGE EXAMPLE:
=============
1. Run: python3 TI_CHANGER_MULTIPLE_2024_10_22.py
2. Choose option 2 from main menu
3. Select first .pt file when prompted
4. Select second .pt file when prompted
5. View visualizations and metrics
6. Choose whether to save average/difference files

This provides a much better user experience with clear access paths!
"""

print(__doc__)
