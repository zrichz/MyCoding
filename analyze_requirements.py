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

class RequirementsScanner:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.imports = defaultdict(set)
        self.failed_files = []
        
        # Directories to skip during scanning
        self.skip_dirs = {'.venv', '__pycache__', '.git', '.pytest_cache', 'node_modules', 
                         'venv', 'env', '.tox', 'build', 'dist', '.egg-info'}
        
        # Common standard library modules (Python 3.8+) - these don't need pip install
        self.stdlib_modules = {
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
            'tkinter', 'turtle', 'cmd', 'readline', 'rlcompleter', 'pprint', 'ssl', 'select'
        }
        
        # Package name mappings (import name -> pip package name)
        self.package_mappings = self._get_package_mappings()
        
    def _get_package_mappings(self):
        """Return package name mappings"""
        return {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'skimage': 'scikit-image', 
            'sklearn': 'scikit-learn',
            'yaml': 'PyYAML',
            'requests': 'requests',
            'bs4': 'beautifulsoup4',
            'matplotlib': 'matplotlib',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'scipy': 'scipy',
            'torch': 'torch',
            'torchvision': 'torchvision',
            'transformers': 'transformers',
            'tensorflow': 'tensorflow',
            'keras': 'keras',
            'flask': 'Flask',
            'django': 'Django',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'pydantic': 'pydantic',
            'sqlalchemy': 'SQLAlchemy',
            'psycopg2': 'psycopg2-binary',
            'pymongo': 'pymongo',
            'redis': 'redis',
            'celery': 'celery',
            'pytest': 'pytest',
            'jupyter': 'jupyter',
            'ipython': 'ipython',
            'notebook': 'notebook',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'bokeh': 'bokeh',
            'streamlit': 'streamlit',
            'dash': 'dash',
            'gradio': 'gradio',
            'huggingface_hub': 'huggingface-hub',
            'datasets': 'datasets',
            'accelerate': 'accelerate',
            'wandb': 'wandb',
            'tensorboard': 'tensorboard',
            'tqdm': 'tqdm',
            'rich': 'rich',
            'click': 'click',
            'typer': 'typer',
            'colorama': 'colorama',
            'python_dotenv': 'python-dotenv',
            'feedparser': 'feedparser',
            'astropy': 'astropy'
        }
        
    def should_skip_directory(self, dir_path):
        """Check if directory should be skipped"""
        return dir_path.name in self.skip_dirs
    
    def extract_imports_from_file(self, file_path):
        """Extract import statements from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to extract imports
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
            
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return set()
    
    def scan_directory(self):
        """Scan all Python files in the directory"""
        python_files = list(self.root_dir.rglob("*.py"))
        
        print(f"Scanning {len(python_files)} Python files...")
        
        for file_path in python_files:
            relative_path = file_path.relative_to(self.root_dir)
            imports = self.extract_imports_from_file(file_path)
            
            for module in imports:
                self.imports[module].add(str(relative_path))
    
    def categorize_modules(self):
        """Categorize modules into standard library and third-party"""
        third_party = {}
        stdlib_used = {}
        
        for module, files in self.imports.items():
            if module in self.stdlib_modules:
                stdlib_used[module] = files
            else:
                third_party[module] = files
        
        return third_party, stdlib_used
    
    def get_known_package_mappings(self):
        """Map import names to pip package names"""
        mappings = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'skimage': 'scikit-image',
            'sklearn': 'scikit-learn',
            'serial': 'pyserial',
            'requests': 'requests',
            'bs4': 'beautifulsoup4',
            'yaml': 'PyYAML',
            'jwt': 'PyJWT',
            'dateutil': 'python-dateutil',
            'psutil': 'psutil',
            'win32api': 'pywin32',
            'win32gui': 'pywin32',
            'win32con': 'pywin32',
            'pywintypes': 'pywin32'
        }
        return mappings
    
    def generate_requirements(self):
        """Generate requirements.txt content"""
        third_party, stdlib_used = self.categorize_modules()
        package_mappings = self.get_known_package_mappings()
        
        requirements = []
        unmapped_modules = []
        
        for module in sorted(third_party.keys()):
            if module in package_mappings:
                requirements.append(package_mappings[module])
            else:
                # Try the module name as-is
                requirements.append(module)
                unmapped_modules.append(module)
        
        return requirements, unmapped_modules, third_party, stdlib_used
    
    def print_report(self):
        """Print a comprehensive report"""
        requirements, unmapped, third_party, stdlib = self.generate_requirements()
        
        print("\n" + "="*80)
        print("PYTHON REQUIREMENTS ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nDirectory analyzed: {self.root_dir}")
        print(f"Python files found: {sum(len(files) for files in self.imports.values())}")
        
        print(f"\n📦 THIRD-PARTY PACKAGES REQUIRED ({len(requirements)}):")
        print("-" * 50)
        for i, pkg in enumerate(requirements, 1):
            print(f"{i:2d}. {pkg}")
        
        if unmapped:
            print(f"\n⚠️  MODULES THAT MIGHT NEED VERIFICATION ({len(unmapped)}):")
            print("-" * 50)
            for module in unmapped:
                files = list(third_party[module])[:3]  # Show first 3 files
                files_str = ", ".join(files)
                if len(third_party[module]) > 3:
                    files_str += f" (+{len(third_party[module])-3} more)"
                print(f"   {module:20} → Used in: {files_str}")
        
        print(f"\n🐍 STANDARD LIBRARY MODULES USED ({len(stdlib)}):")
        print("-" * 50)
        for module in sorted(stdlib.keys()):
            print(f"   {module}")
        
        print(f"\n📝 REQUIREMENTS.TXT CONTENT:")
        print("-" * 50)
        for pkg in requirements:
            print(pkg)
        
        print(f"\n💾 INSTALLATION COMMAND:")
        print("-" * 50)
        if requirements:
            print("pip install " + " ".join(requirements))
        else:
            print("No third-party packages required!")
        
        return requirements

def main():
    # Get the MyCoding directory (where this script is located)
    script_dir = Path(__file__).parent
    
    print("Python Requirements Scanner for MyCoding Repository")
    print("=" * 60)
    print(f"Scanning directory: {script_dir}")
    print("Excluding: .venv, __pycache__, .git, and other cache directories")
    print()
    
    scanner = RequirementsScanner(script_dir)
    scanner.scan_directory()
    requirements = scanner.print_summary()
    
    # Save requirements.txt
    requirements_file = script_dir / "requirements.txt"
    with open(requirements_file, 'w') as f:
        f.write("# Auto-generated requirements for MyCoding repository\n")
        f.write("# Generated by analyze_requirements.py\n\n")
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"\n✓ Requirements saved to: {requirements_file}")
    print("\nTo install all requirements:")
    print("  source .venv/bin/activate")
    print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
