#!/usr/bin/env python3
"""
Python Requirements Scanner for MyCoding Repository
Scans all Python files and extracts import statements to generate a requirements list
Excludes .venv and other cache directories
"""

import os
import re
import ast
from pathlib import Path
from collections import defaultdict

def get_stdlib_modules():
    """Return set of Python standard library modules"""
    return {
        'os', 'sys', 'json', 'math', 'random', 'time', 'datetime', 'collections',
        'itertools', 'functools', 'operator', 'pathlib', 'glob', 'shutil', 'subprocess',
        'threading', 'multiprocessing', 'queue', 'socket', 'urllib', 'http', 'email',
        'html', 'xml', 'csv', 'sqlite3', 'logging', 'unittest', 'argparse', 'configparser',
        'tempfile', 'gzip', 'zipfile', 'tarfile', 'pickle', 'copy', 'hashlib', 'hmac',
        'uuid', 'base64', 'binascii', 'struct', 'array', 'weakref', 'gc', 'inspect',
        'types', 'enum', 'dataclasses', 'typing', 'contextlib', 'atexit', 'signal',
        'platform', 'getpass', 'io', 're', 'string', 'textwrap', 'unicodedata',
        'locale', 'calendar', 'heapq', 'bisect', 'statistics', 'decimal', 'fractions',
        'warnings', 'traceback', 'pdb', 'profile', 'pstats', 'timeit', 'cProfile',
        'tkinter', 'turtle', 'cmd', 'readline', 'rlcompleter', 'pprint', 'ssl', 'select',
        'asyncio', 'concurrent', 'ctypes', 'abc', 'builtins', 'codecs'
    }

def get_package_mappings():
    """Return mapping of import names to pip package names"""
    return {
        'cv2': 'opencv-python',
        'PIL': 'Pillow', 
        'skimage': 'scikit-image',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'bs4': 'beautifulsoup4',
        'feedparser': 'feedparser',
        'astropy': 'astropy',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'transformers': 'transformers',
        'requests': 'requests',
        'flask': 'Flask',
        'django': 'Django',
        'fastapi': 'fastapi'
    }

def should_skip_directory(dir_path):
    """Check if directory should be skipped"""
    skip_dirs = {'.venv', '__pycache__', '.git', '.pytest_cache', 'node_modules', 
                'venv', 'env', '.tox', 'build', 'dist'}
    return dir_path.name in skip_dirs

def extract_imports_from_file(file_path):
    """Extract import statements from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Try AST parsing first
        try:
            tree = ast.parse(content)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            return imports
            
        except SyntaxError:
            # Fallback to regex
            imports = set()
            import_pattern = r'^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            
            for line in content.split('\n'):
                if line.strip().startswith('#'):
                    continue
                match = re.match(import_pattern, line)
                if match:
                    imports.add(match.group(1))
            
            return imports
            
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return set()

def scan_directory(root_dir):
    """Scan directory for Python files and collect imports"""
    root_path = Path(root_dir)
    all_imports = defaultdict(list)
    file_count = 0
    
    print(f"Scanning directory: {root_path}")
    print("Excluding: .venv, __pycache__, .git, and other cache directories")
    print("=" * 60)
    
    for root, dirs, files in os.walk(root_path):
        # Filter out directories to skip
        dirs[:] = [d for d in dirs if not should_skip_directory(Path(d))]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(root_path)
                
                imports = extract_imports_from_file(file_path)
                file_count += 1
                
                if imports:
                    print(f"✓ {relative_path}: {len(imports)} imports")
                    for imp in imports:
                        all_imports[imp].append(str(relative_path))
                else:
                    print(f"- {relative_path}: no imports")
    
    print(f"\nScanned {file_count} Python files")
    return dict(all_imports)

def generate_requirements(all_imports):
    """Generate requirements list from imports"""
    stdlib_modules = get_stdlib_modules()
    package_mappings = get_package_mappings()
    
    # Separate third-party from standard library
    third_party = {}
    stdlib_used = {}
    
    for module, files in all_imports.items():
        if module in stdlib_modules:
            stdlib_used[module] = files
        else:
            third_party[module] = files
    
    # Map to pip package names
    requirements = set()
    unmapped = set()
    
    for module in third_party.keys():
        if module in package_mappings:
            requirements.add(package_mappings[module])
        else:
            requirements.add(module)
            unmapped.add(module)
    
    return sorted(requirements), sorted(unmapped), third_party, stdlib_used

def main():
    script_dir = Path(__file__).parent
    
    print("Python Requirements Scanner for MyCoding Repository")
    print("=" * 60)
    
    # Scan for imports
    all_imports = scan_directory(script_dir)
    
    # Generate requirements
    requirements, unmapped, third_party, stdlib_used = generate_requirements(all_imports)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total unique imports: {len(all_imports)}")
    print(f"Standard library modules: {len(stdlib_used)}")
    print(f"Third-party modules: {len(third_party)}")
    print(f"Pip packages needed: {len(requirements)}")
    
    if unmapped:
        print(f"Packages to verify manually: {len(unmapped)}")
    
    # Print requirements
    print("\n" + "=" * 60)
    print("REQUIREMENTS.TXT CONTENT")
    print("=" * 60)
    
    for req in requirements:
        print(req)
    
    if unmapped:
        print("\n# Verify these package names manually:")
        for pkg in unmapped:
            print(f"# {pkg}")
    
    # Print detailed breakdown
    print("\n" + "=" * 60)
    print("THIRD-PARTY MODULES DETAIL")
    print("=" * 60)
    
    for module, files in sorted(third_party.items()):
        print(f"\n{module}: (used in {len(files)} files)")
        for file_path in sorted(files)[:3]:  # Show first 3 files
            print(f"  - {file_path}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")
    
    # Save requirements.txt
    requirements_file = script_dir / "requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write("# Auto-generated requirements for MyCoding repository\n")
        f.write("# Generated by analyze_requirements.py\n")
        f.write("# Excludes .venv and cache directories\n\n")
        for req in requirements:
            f.write(f"{req}\n")
        
        if unmapped:
            f.write("\n# Verify these package names manually:\n")
            for pkg in unmapped:
                f.write(f"# {pkg}\n")
    
    print(f"\n✓ Requirements saved to: {requirements_file}")
    print("\nTo install all requirements:")
    print("  source .venv/bin/activate")
    print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
