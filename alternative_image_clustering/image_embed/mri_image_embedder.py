import os
import glob
import joblib
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class MRIImageEmbedder:
    """
    Extracts CLIP embeddings for MRI slice images.
    Works on CPU or GPU.

    Produces a dictionary: { "A1K2P5_slice_003.jpg": numpy array([512 dims]), ... }
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        print(f"Loading CLIP model: {model_name}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        print(f"Model loaded on: {self.device.upper()}")

    def _load_images(self, img_dir: str):
        """
        Recursively load JPG/PNG images from a folder.
        """
        print(f"Searching for images inside: {img_dir}")

        img_paths = sorted(
            glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
        )

        print(f"Found {len(img_paths)} images.")
        return img_paths

    def generate_embeddings(self, img_dir: str, save_path: str):
        """
        Extract CLIP embeddings for each image and save to a .pbz2 file.
        """
        img_paths = self._load_images(img_dir)

        if len(img_paths) == 0:
            raise ValueError("ERROR: No images found in the directory!")

        embeddings = {}

        print("\nExtracting embeddings...\n")

        for p in tqdm(img_paths):
            image = Image.open(p).convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)

            # Normalize features
            features = features / features.norm(p=2, dim=-1, keepdim=True)

            embeddings[os.path.basename(p)] = features.cpu().detach().numpy()[0]

        print(f"\nSaving embeddings to: {save_path}")
        joblib.dump(embeddings, save_path)

        print("✓ Embeddings saved successfully!")
        return embeddings
