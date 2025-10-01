#!/usr/bin/env python3
"""
IBM EGA Font Checker - Verifies if Ac437_IBM_EGA_8x8.ttf is installed
"""

import os
import sys
from pathlib import Path

def check_font_locations():
    """Check common font installation locations for IBM EGA font"""
    
    font_filename = "Ac437_IBM_EGA_8x8.ttf"
    
    # Common Windows font locations
    locations_to_check = [
        "C:/Windows/Fonts/",
        "C:/Windows/System32/Fonts/",
        os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts/"),  # User fonts
        "C:/Program Files/Common Files/Microsoft Shared/Fonts/",
        # Alternative paths
        "C:/WINDOWS/Fonts/",
        "C:/WINNT/Fonts/",
    ]
    
    print("🔍 Checking for IBM EGA Font (Ac437_IBM_EGA_8x8.ttf)...")
    print("=" * 60)
    
    found_locations = []
    
    for location in locations_to_check:
        font_path = os.path.join(location, font_filename)
        exists = os.path.exists(font_path)
        
        status = "✅ FOUND" if exists else "❌ Not found"
        print(f"{status}: {font_path}")
        
        if exists:
            found_locations.append(font_path)
            # Get file info
            try:
                stat = os.stat(font_path)
                size_kb = stat.st_size / 1024
                print(f"    📏 Size: {size_kb:.1f} KB")
            except:
                pass
    
    print("\n" + "=" * 60)
    
    if found_locations:
        print(f"🎉 IBM EGA Font found in {len(found_locations)} location(s):")
        for loc in found_locations:
            print(f"   • {loc}")
        
        # Test loading with PIL
        print("\n🧪 Testing font loading with PIL...")
        try:
            from PIL import ImageFont
            test_font = ImageFont.truetype(found_locations[0], 16)
            print("✅ Font loads successfully with PIL!")
        except ImportError:
            print("⚠️  PIL (Pillow) not available for testing")
        except Exception as e:
            print(f"❌ Error loading font with PIL: {e}")
    else:
        print("❌ IBM EGA Font NOT FOUND in any standard location!")
        print("\n📥 To install the font:")
        print("1. Download Ac437_IBM_EGA_8x8.ttf")
        print("2. Right-click the font file → 'Install' or 'Install for all users'")
        print("3. Or manually copy to C:/Windows/Fonts/")
        
        # Check if we can list fonts directory
        print("\n📂 Contents of C:/Windows/Fonts/ (first 10 .ttf files):")
        try:
            fonts_dir = "C:/Windows/Fonts/"
            if os.path.exists(fonts_dir):
                ttf_files = [f for f in os.listdir(fonts_dir) if f.lower().endswith('.ttf')]
                for i, font_file in enumerate(sorted(ttf_files)[:10]):
                    print(f"   • {font_file}")
                if len(ttf_files) > 10:
                    print(f"   ... and {len(ttf_files) - 10} more .ttf files")
            else:
                print("   ❌ Cannot access fonts directory")
        except Exception as e:
            print(f"   ❌ Error listing fonts: {e}")

def check_system_fonts():
    """Try to list system fonts using PIL if available"""
    try:
        from PIL import ImageFont
        print("\n🔍 Attempting to find EGA-related fonts in system...")
        
        # Common variations of the font name
        ega_variants = [
            "Ac437_IBM_EGA_8x8.ttf",
            "IBM_EGA_8x8.ttf", 
            "EGA8x8.ttf",
            "IBM EGA",
            "EGA"
        ]
        
        for variant in ega_variants:
            try:
                font = ImageFont.truetype(variant, 12)
                print(f"✅ Found system font: {variant}")
            except:
                pass
                
    except ImportError:
        print("⚠️  PIL not available for system font detection")

if __name__ == "__main__":
    print("IBM EGA Font Checker")
    print("=" * 60)
    
    check_font_locations()
    check_system_fonts()
    
    print("\n" + "=" * 60)
    print("💡 If font is not found, you can download it from:")
    print("   • https://int10h.org/oldschool-pc-fonts/")
    print("   • Search for 'IBM EGA 8x8' or 'Ac437_IBM_EGA_8x8'")
    print("\n🛠️  Installation: Right-click font file → Install for all users")
    
    input("\nPress Enter to exit...")