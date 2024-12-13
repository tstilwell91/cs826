import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
import numpy as np
import pandas as pd
import random
from openslide import OpenSlide
import torch.nn.functional as F

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration parameters
DATA_DIR = "/home/tstil004/phd/multi-omics/slides/"  # Directory containing the TCGA dataset
CASE_MAPPING_FILE = '/home/tstil004/phd/multi-omics/file_case_mapping.csv'  # Path to the case mapping file
BATCH_SIZE = 8  # Number of WSIs processed in parallel during feature extraction. Increase to utilize GPU memory better if available.
NUM_WORKERS = 8  # Number of worker processes used by the DataLoader to load data in parallel. Increase to speed up data loading.
EPOCHS = 1  # Number of times the entire dataset is passed through the model. More epochs ensure better coverage of the dataset.
NUM_TILES = 8  # Number of random tiles sampled from each WSI per epoch. More tiles provide better representation but require more memory.
FEATURE_DIM = 512  # Dimensionality of the feature vector extracted from each tile by the ResNet18 model.

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the case mapping file
case_mapping_df = pd.read_csv(CASE_MAPPING_FILE)

# Define a custom Dataset to handle WSIs
class WSIDataset(Dataset):
    def __init__(self, wsi_paths, transform=None, num_tiles=NUM_TILES):
        self.wsi_paths = wsi_paths
        self.transform = transform
        self.num_tiles = num_tiles
        print(f"Initialized WSIDataset with {len(self.wsi_paths)} WSIs.")

    def __len__(self):
        return len(self.wsi_paths)  # Dataset length is based on the number of WSIs

    def __getitem__(self, idx):
        wsi_path = self.wsi_paths[idx]
        slide = OpenSlide(wsi_path)
        width, height = slide.dimensions
        tiles = []
        for _ in range(self.num_tiles):
            # Calculate random coordinates to extract a tile
            tile_size = min(width, height) // 10  # Define tile size
            x = random.randint(0, max(0, width - tile_size))
            y = random.randint(0, max(0, height - tile_size))

            # Extract a tile from the WSI
            tile = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
            if self.transform:
                tile = self.transform(tile)
            tiles.append(tile)

        tiles = torch.stack(tiles)  # Shape: (NUM_TILES, 3, 224, 224)
        return wsi_path, tiles

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load WSI paths
print("Loading WSI paths...")
wsi_paths = []
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".svs") or file.endswith(".tif"):
            wsi_paths.append(os.path.join(root, file))
if len(wsi_paths) == 0:
    raise ValueError(f"No WSI files found in {DATA_DIR}. Please check the directory and ensure it contains subdirectories with files with supported extensions.")
print(f"Found {len(wsi_paths)} WSI paths across subdirectories.")
dataset = WSIDataset(wsi_paths, transform=transform)

total_dataset_size = len(dataset)  # Total number of WSIs
num_batches_per_epoch = (total_dataset_size + BATCH_SIZE - 1) // BATCH_SIZE  # Calculate the number of batches per epoch

print(f"Number of tiles per WSI: {NUM_TILES}")
print(f"Total number of WSIs: {total_dataset_size}")
print(f"Batch size: {BATCH_SIZE} (Defined by the number of WSIs processed in parallel during feature extraction)")
print(f"Number of batches per epoch: {num_batches_per_epoch} (Calculated as total number of WSIs {total_dataset_size} divided by batch size {BATCH_SIZE}, rounded up)")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# Load pretrained ResNet18 model and modify it for feature extraction
print("Loading pretrained ResNet18 model...")
resnet18 = models.resnet18(pretrained=True)
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])  # Remove the final classification layer
resnet18 = resnet18.to(device)
resnet18.eval()
print("Model loaded and ready for feature extraction.")

# Attention MIL module
class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMIL, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_tiles, feature_dim)
        attn_weights = self.attention(x)  # Shape: (batch_size, num_tiles, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # Apply softmax across tiles
        weighted_features = torch.sum(attn_weights * x, dim=1)  # Weighted sum of tile features
        return weighted_features

# Feature extraction loop with Attention MIL
def extract_features_with_attention(dataloader, model, attention_model, device, epochs):
    all_features = []
    with torch.no_grad():
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}...")
            for batch_idx, (wsi_paths, batch) in enumerate(dataloader):
                print(f"Processing batch {batch_idx + 1} of epoch {epoch + 1}... (using device: {device})")
                batch = batch.to(device, non_blocking=True)  # Shape: (BATCH_SIZE, NUM_TILES, 3, 224, 224)
                print(f"Batch moved to {device}. Performing forward pass...")
                batch_size, num_tiles, _, _, _ = batch.shape
                batch = batch.view(batch_size * num_tiles, 3, 224, 224)  # Reshape to process all tiles
                features = model(batch)  # Extract features from ResNet18
                features = features.view(batch_size, num_tiles, -1)  # Shape: (BATCH_SIZE, NUM_TILES, FEATURE_DIM)
                aggregated_features = attention_model(features)  # Aggregate tile features using Attention MIL

                # Extract case_id from wsi_paths using the mapping file
                for i, wsi_path in enumerate(wsi_paths):
                    wsi_file_name = os.path.basename(wsi_path)
                    case_id = case_mapping_df.loc[case_mapping_df['file_name'] == wsi_file_name, 'case_id'].values
                    if len(case_id) == 0:
                        print(f"Warning: No matching case_id found for WSI {wsi_file_name}. Skipping.")
                        continue
                    case_id = case_id[0]
                    aggregated_feature = aggregated_features[i].cpu().numpy()
                    feature_row = [case_id] + aggregated_feature.tolist()
                    all_features.append(feature_row)
                print(f"Batch {batch_idx + 1} of epoch {epoch + 1} processed.")

    return all_features

# Initialize Attention MIL model
attention_mil = AttentionMIL(input_dim=FEATURE_DIM, hidden_dim=256).to(device)
attention_mil.eval()

# Extract features with Attention MIL
print("Starting feature extraction with Attention MIL...")
features = extract_features_with_attention(dataloader, resnet18, attention_mil, device, EPOCHS)

# Save features as .csv
print("Saving features as .csv...")
columns = ['case_id'] + [f'feature_{i}' for i in range(FEATURE_DIM)]
features_df = pd.DataFrame(features, columns=columns)
features_df.to_csv('extracted_wsi_features.csv', index=False)

print("Feature extraction complete. Saved to 'extracted_wsi_features.csv'")
