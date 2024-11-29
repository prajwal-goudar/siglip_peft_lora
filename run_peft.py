# Authors: 
#   Name: Prajwal Goudar
#   Student ID: 202304088
#   Email: x2023dvk@stfx.ca

#   Name: Rahul Rudragoudar
#   Student ID: 202304297
#   Email: x2023dvl@stfx.ca

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForImageClassification
from peft import get_peft_model, LoraConfig
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
import argparse


# Argument parser
parser = argparse.ArgumentParser(description='Run PEFT model training on Google Siglip Model which is currently used in df-analyze to generate embeddings')
parser.add_argument('--parquet_file', type=str, required=True, help='Path to the parquet file, the paquet file should be made up of images in byte format named as image, label i.e. the target variable should be named as label and other extra features (metadata) that you want to include')
parser.add_argument('--extra_features', type=str, nargs='+', default=[], help='Extra features other than label and image. Please note: Image should be converted into bytes format while storiong in parquet file.')
parser.add_argument('--test_size', type=float, default=0.2, help='Test size for train-test split, should be ranging between 0.1-0.6.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs. Default is 10.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training. Default is 16. Adjust this if you encounter out-of-memory errors.')
args = parser.parse_args()

# Load the dataset
df = pd.read_parquet(args.parquet_file)

# Define image transformations

# The images are transformed to 384 by 384 because the siglip 384 model was trained on images of these size. If you want to get rid of this transformation method you can comment out resize method.
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# This class will help us to create image objects with their features variables. This also performs transformation as required by tensor packages.
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, extra_features=[]):
        self.dataframe = dataframe
        self.transform = transform
        self.extra_features = extra_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_bytes = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['label']
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        extra_data = {feature: self.dataframe.iloc[idx][feature] for feature in self.extra_features}
        if self.transform:
            img = self.transform(img)
        return img, label, extra_data

# Split the dataset into train and test
train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)

# Create datasets
train_dataset = ImageDataset(train_df, transform=transform, extra_features=args.extra_features)
test_dataset = ImageDataset(test_df, transform=transform, extra_features=args.extra_features)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Load pre-trained model
model_name = 'google/siglip-so400m-patch14-384'
model = AutoModelForImageClassification.from_pretrained(model_name)

# Apply LoRA
# Please note: If you find better configuration please feel free to use them here.
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q_proj', 'v_proj'],
    inference_mode=False
)

# This uses hugging face peft package to prepare peft model.
model = get_peft_model(model, config)

# This line cleans up any cache which was built during the execution of any previous runs from torch. This can be omitted if your machine has enough amount of memory.
torch.cuda.empty_cache()

# Selects the device type i.e. cpu or gpu and transfers the model to 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

print("Model loaded successfully")

# These are optimiser and loss functions
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# This helps to analyse the training process in real time using TensorBoard package
writer = SummaryWriter(log_dir='./logs')

# Training loop
num_epochs = args.num_epochs
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch + 1}/{num_epochs}')
    
    # Training Phase
    model.train()
    total_train_loss = 0
    for batch_idx, (images, labels, extra_data) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
        # Log training loss to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
    
    avg_train_loss = total_train_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

# Test the model after training
model.eval()
test_loss = 0
with torch.no_grad():
    for images, labels, extra_data in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')

# Close the writer
writer.close()

# Save the model
model_save_path = 'trained_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')