import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.samples = []

        # Loop over each patient folder inside root_dir
        for patient in os.listdir(root_dir):
            patient_dir = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            img_path, mask_path = None, None

            # Accept both .nii and .nii.gz, matching the M&Ms naming convention
            for f in os.listdir(patient_dir):
                full = os.path.join(patient_dir, f)

                if f.endswith("_sa.nii") or f.endswith("_sa.nii.gz"):
                    img_path = full
                if f.endswith("_sa_gt.nii") or f.endswith("_sa_gt.nii.gz"):
                    mask_path = full

            if img_path and mask_path:
                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load NIfTI files with nibabel
        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.int64)

        # Convert to PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask, img_path
