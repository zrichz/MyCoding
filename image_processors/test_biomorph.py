#!/usr/bin/env python3
"""
Simple biomorph test to debug the algorithm
"""
def test_biomorph():
    # Test with original BASIC parameters
    width, height = 100, 100
    const_real, const_imag = 0.5, 0.0
    max_iter = 100
    escape_radius = 10.0
    
    # Original BASIC coordinate system
    ymax = 2.5
    ymin = -ymax
    aspect_ratio = 4.0/3.0  # Original was 429/321
    xmax = ymax * aspect_ratio
    xmin = -xmax
    
    print(f"Coordinate bounds: x=({xmin:.2f}, {xmax:.2f}), y=({ymin:.2f}, {ymax:.2f})")
    
    result = [[0 for _ in range(width)] for _ in range(height)]
    black_count = 0
    white_count = 0
    
    for i in range(height):
        for j in range(width):
            # Map pixel to complex plane (exact BASIC translation)
            x0 = xmin + (xmax - xmin) * j / (width - 1)
            y0 = -ymin - (ymax - ymin) * i / (height - 1)
            
            x, y = x0, y0
            
            # Iterate exactly as in BASIC
            for n in range(1, max_iter + 1):
                # z^3 + c
                xx = x * (x * x - 3 * y * y) + const_real
                yy = y * (3 * x * x - y * y) + const_imag
                x, y = xx, yy
                
                # Original BASIC escape condition
                if abs(x) > escape_radius or abs(y) > escape_radius or x*x + y*y > escape_radius*escape_radius:
                    break
            
            # Original BASIC coloring logic
            if abs(x) < escape_radius and abs(y) < escape_radius:
                result[i][j] = 0  # Black - the biomorph
                black_count += 1
            else:
                result[i][j] = 255  # White - background
                white_count += 1
    
    print(f"Result: {black_count} black pixels, {white_count} white pixels")
    
    if black_count > 0:
        print("SUCCESS: Biomorph generated!")
        print("Sample black pixel locations:")
        found = 0
        for i in range(height):
            for j in range(width):
                if result[i][j] == 0 and found < 5:
                    x0 = xmin + (xmax - xmin) * j / (width - 1)
                    y0 = -ymin - (ymax - ymin) * i / (height - 1)
                    print(f"  Pixel ({i},{j}) -> coords ({x0:.3f},{y0:.3f})")
                    found += 1
    else:
        print("FAILED: All pixels are white!")
        
    return result

if __name__ == "__main__":
    test_biomorph()
