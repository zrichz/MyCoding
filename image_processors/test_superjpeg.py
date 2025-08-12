# Test the SuperJPEG encoder with a simple test
import numpy as np
from superjpeg_encoder import SuperJPEGEncoder
import json

print('Testing SuperJPEG encoder...')

# Create a simple test image (gradient)
test_image = np.zeros((64, 64, 3), dtype=np.uint8)
for i in range(64):
    for j in range(64):
        test_image[i, j, 0] = int(255 * i / 63)  # Red gradient
        test_image[i, j, 1] = int(255 * j / 63)  # Green gradient  
        test_image[i, j, 2] = 128  # Blue constant

print(f'Created test image: {test_image.shape}')

# Test different block sizes
for block_size in [4, 8, 16]:
    print(f'Testing block size {block_size}x{block_size}...')
    
    encoder = SuperJPEGEncoder(block_size=block_size, quality=50)
    
    # Encode
    encoded_data = encoder.encode_image(test_image)
    print(f'  Encoded successfully')
    
    # Decode
    decoded_image = encoder.decode_image(encoded_data)
    print(f'  Decoded successfully: {decoded_image.shape}')
    
    # Check dimensions match
    if decoded_image.shape == test_image.shape:
        print(f'  ✓ Dimensions match')
    else:
        print(f'  ✗ Dimension mismatch: {decoded_image.shape} vs {test_image.shape}')
    
    # Save test file
    filename = f'test_superjpeg_{block_size}x{block_size}.json'
    with open(filename, 'w') as f:
        json.dump(encoded_data, f, indent=2)
    print(f'  Saved {filename}')

print('✅ All tests completed successfully!')
