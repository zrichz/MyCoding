import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# Define the custom dataset class
class EmojiDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Create the dataset
emoji_dataset = EmojiDataset(image_dir=emoji_dir, transform=transform)

# Create a DataLoader
dataloader = DataLoader(emoji_dataset, batch_size=32, shuffle=True)

# Example usage
for i, images in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(images.shape)  # Should print torch.Size([32, 3, 64, 64])
    # Perform further processing on the images