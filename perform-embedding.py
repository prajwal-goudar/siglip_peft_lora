import argparse
import torch
from transformers import AutoModelForImageClassification
from peft import PeftModel, LoraConfig, get_peft_model
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Perform embedding with a fine-tuned model.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
parser.add_argument('--parquet_file', type=str, required=True, help='Path to the parquet file containing images and labels.')
parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader.')
parser.add_argument('--extra_features', type=str, nargs='+', default=[], help='Extra features other than label and image.')
args = parser.parse_args()

# Load the base model
model_name = 'google/siglip-so400m-patch14-384'
base_model = AutoModelForImageClassification.from_pretrained(model_name)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q_proj', 'v_proj'],
    inference_mode=True
)

# Apply the LoRA configuration to the base model
model = get_peft_model(base_model, config)

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_bytes = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['label']
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Load the fine-tuned weights
model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.eval()

df = pd.read_parquet(args.parquet_file)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Split the dataset into train and test
train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)

# Create datasets
train_dataset = ImageDataset(train_df, transform=transform)
test_dataset = ImageDataset(test_df, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

embeddings = []
labels = []

with torch.no_grad():
    for images, lbls in tqdm(test_loader):
        images = images.to(device)
        outputs = model(pixel_values=images, output_hidden_states=True)
        cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
        labels.extend(lbls.numpy())

# Concatenate all embeddings
embeddings = np.concatenate(embeddings, axis=0)

df_embeddings = pd.DataFrame(embeddings)
df_embeddings['label'] = labels
# Add extra features
for feature in args.extra_features:
    df_embeddings[feature] = test_df[feature].reset_index(drop=True)
df_embeddings.to_parquet('image_embeddings.parquet')