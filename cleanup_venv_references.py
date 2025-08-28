#!/usr/bin/env python3
"""
Virtual Environment Cleanup Script
Removes all references to spurious venv_myDL1 and updates to venv_mycoding
"""

import json
import os
import re
from pathlib import Path

def clean_jupyter_notebook(notebook_path):
    """Clean a Jupyter notebook to remove venv_myDL1 references"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Update kernel specification
        if 'metadata' in notebook:
            if 'kernelspec' in notebook['metadata']:
                kernelspec = notebook['metadata']['kernelspec']
                if kernelspec.get('display_name') == 'venv_myDL1':
                    kernelspec['display_name'] = 'venv_mycoding'
                    kernelspec['name'] = 'venv_mycoding'
                    print(f"Updated kernel in: {notebook_path}")
        
        # Clear cells with venv_myDL1 error outputs
        if 'cells' in notebook:
            for cell in notebook['cells']:
                if 'outputs' in cell:
                    for output in cell['outputs']:
                        if 'text' in output:
                            # Check if text contains venv_myDL1 paths
                            if isinstance(output['text'], list):
                                output['text'] = [line for line in output['text'] 
                                                if 'venv_myDL1' not in line]
                            elif isinstance(output['text'], str):
                                if 'venv_myDL1' in output['text']:
                                    output['text'] = ""
                        
                        # Clear traceback outputs containing venv_myDL1
                        if 'traceback' in output:
                            output['traceback'] = [line for line in output['traceback']
                                                 if 'venv_myDL1' not in line]
        
        # Write back the cleaned notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
            
        return True
        
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")
        return False

def main():
    """Main cleanup function"""
    # Get the MyCoding directory
    base_dir = Path(__file__).parent
    
    # List of notebooks to clean
    notebooks_to_clean = [
        "archived/Dr45_SD_anim_script_MUCH FASTER.ipynb",
        "machine_learning/DUDL_CNN_MNIST_SANDBOX.ipynb", 
        "machine_learning/GAN_FCN_CIFAR10_smaller.ipynb",
        "machine_learning/WIP_CNN_Feature_Maps_directory_cuda.ipynb",
        "machine_learning/WIP GAN_FCN_RGB_imagesFromDir.ipynb",
        "machine_learning/WORKS_CNN_Feature_Maps_CIFAR10_cuda.ipynb",
        "machine_learning/WORKS_CNN_Feature_Maps_directory_cuda.ipynb",
        "machine_learning/WORKS_GAN_CNN_RGB_imagesFromDir.ipynb"
    ]
    
    print("Starting Virtual Environment Cleanup...")
    print("=" * 50)
    
    cleaned_count = 0
    for notebook in notebooks_to_clean:
        notebook_path = base_dir / notebook
        if notebook_path.exists():
            if clean_jupyter_notebook(notebook_path):
                cleaned_count += 1
                print(f"✅ Cleaned: {notebook}")
            else:
                print(f"❌ Failed: {notebook}")
        else:
            print(f"⚠️  Not found: {notebook}")
    
    print("=" * 50)
    print(f"Cleanup completed. {cleaned_count} notebooks processed.")
    print("\nManual steps still needed:")
    print("1. Open each notebook in Jupyter/VS Code")
    print("2. Select the 'venv_mycoding' kernel")
    print("3. Save the notebook to apply kernel changes")
    print("\nAll references to venv_myDL1 should now be cleaned!")

if __name__ == "__main__":
    main()
