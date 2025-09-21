import argparse
import os
from gaussian_clahe import gaussian_clahe_color
from PIL import Image

def process_image_file(input_path, output_path, tile_size, kernel_size, stride, max_size):
    img = Image.open(input_path).convert('RGB')
    if max_size:
        img.thumbnail((max_size, max_size))
    result_img = gaussian_clahe_color(img, tile_size, kernel_size, stride)
    result_img.save(output_path)
    print(f"✔ Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="CLAHE-style contrast enhancement with Gaussian-weighted tiles.")
    parser.add_argument("input", help="Input image file or folder")
    parser.add_argument("output", help="Output image file or folder")
    parser.add_argument("--tile", type=int, default=64, help="Tile size (default: 64)")
    parser.add_argument("--kernel", type=int, default=21, help="Gaussian kernel size (default: 21)")
    parser.add_argument("--stride", type=int, help="Stride between tiles (default: tile_size // 2)")
    parser.add_argument("--max_size", type=int, choices=[360, 512, 1024, 2048], 
                        help="Maximum width/height for processing (default: no resizing)")

    args = parser.parse_args()

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for fname in os.listdir(args.input):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                input_path = os.path.join(args.input, fname)
                output_path = os.path.join(args.output, f"enhanced_{fname}")
                process_image_file(input_path, output_path, args.tile, args.kernel, args.stride, args.max_size)
    else:
        process_image_file(args.input, args.output, args.tile, args.kernel, args.stride, args.max_size)

if __name__ == "__main__":
    main()
