from superjpeg_encoder import SuperJPEGEncoder
import numpy as np

print('Testing improved quantization scaling...')
print()

# Test different block sizes
for block_size in [8, 16, 32, 64]:
    encoder = SuperJPEGEncoder(block_size=block_size, quality=50)
    quant_table = encoder.quantization_table
    
    print(f'Block size {block_size}x{block_size}:')
    print(f'  DC coefficient (0,0): {quant_table[0,0]}')
    print(f'  Low freq (1,1): {quant_table[1,1]}')
    if block_size >= 8:
        print(f'  Mid freq (4,4): {quant_table[4,4]}')
    if block_size >= 16:
        print(f'  High freq (8,8): {quant_table[8,8]}')
    if block_size >= 32:
        print(f'  Very high freq (16,16): {quant_table[16,16]}')
    
    # Show corner values
    corner = quant_table[-1, -1]
    print(f'  Highest freq ({block_size-1},{block_size-1}): {corner}')
    print()

print('✅ Quantization scaling test completed!')
