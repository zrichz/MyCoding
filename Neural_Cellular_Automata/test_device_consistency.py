"""
Device Consistency Test for NCA
==============================
This script tests if all tensors are on the correct device during training.
"""

import torch
import sys
import os

# Add the current directory to path to import NCA_baseline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NCA_baseline import NeuralCellularAutomata, NCATrainer

def test_device_consistency():
    print("🔧 Testing device consistency for NCA...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        print("✅ CUDA available, using GPU")
        device = torch.device('cuda')
    
    try:
        # Create model
        print("Creating model...")
        model = NeuralCellularAutomata(channel_n=16, fire_rate=0.5)
        model.to(device)
        
        # Create dummy target image
        print("Creating target image...")
        target_image = torch.rand(1, 4, 64, 64).to(device)
        
        # Create trainer
        print("Creating trainer...")
        trainer = NCATrainer(model, target_image, device)
        
        # Test seed creation
        print("Testing seed creation...")
        seed = trainer.create_seed(64, random_init=True)
        print(f"✓ Seed device: {seed.device}")
        
        # Test model forward pass
        print("Testing model forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(seed, steps=5)
            print(f"✓ Output device: {output.device}")
        
        # Test training step
        print("Testing training step...")
        model.train()
        loss, training_output = trainer.train_step(steps_range=(5, 10))
        print(f"✓ Training output device: {training_output.device}")
        print(f"✓ Training loss: {loss:.6f}")
        
        print("\n🎉 All device consistency tests passed!")
        print("Your NCA should now work correctly with CUDA.")
        
    except Exception as e:
        print(f"\n❌ Device consistency test failed: {str(e)}")
        print("This indicates there's still a device mismatch issue.")
        return False
    
    return True

if __name__ == "__main__":
    test_device_consistency()
