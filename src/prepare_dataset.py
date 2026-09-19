"""
File: src/prepare_dataset.py
Purpose: This script is the single entry point for preparing the image dataset for the DIP branch.
Why we need it: We need a reliable, optimized way to transition from the raw 10-class Kaggle 
                dataset to a clean, standardized format suitable for PyTorch CNN training. 
                It handles data retrieval, clean directory mapping, and image resizing in one go.
"""

import os
import shutil
import cv2
import kagglehub

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Mapping from the messy Kaggle folder names to clean, standardized names
CLASS_MAPPING = {
    "1. Eczema 1677": "eczema",
    "2. Melanoma 15.75k": "melanoma",
    "3. Atopic Dermatitis - 1.25k": "atopic_dermatitis",
    "4. Basal Cell Carcinoma (BCC) 3323": "basal_cell_carcinoma",
    "5. Melanocytic Nevi (NV) - 7970": "melanocytic_nevi",
    "6. Benign Keratosis-like Lesions (BKL) 2624": "benign_keratosis",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "psoriasis_lichen_planus",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": "seborrheic_keratoses",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "fungal_infections",
    "10. Warts Molluscum and other Viral Infections - 2103": "viral_infections"
}

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
IMG_SIZE = 224

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def get_kaggle_dataset_path() -> str:
    """
    Downloads or locates the 'ismailpromus' dataset via the Kaggle API.
    Why we need it: We need the absolute path to the raw data files without forcing 
    the user to manually download and extract 6GB of images.
    """
    print("Locating dataset via kagglehub...")
    # This automatically downloads if missing, or just returns the path if cached
    path = kagglehub.dataset_download("ismailpromus/skin-diseases-image-dataset")
    # The actual images are inside the 'IMG_CLASSES' subdirectory
    return os.path.join(path, "IMG_CLASSES")


def setup_raw_symlinks(kaggle_path: str):
    """
    Creates symbolic links in 'data/raw' pointing to the cached Kaggle folders, renaming them cleanly.
    Why we need it: Copying 6GB of raw data is highly inefficient (Quantity < Quality/Performance). 
    Symlinking gives us cleanly named folders in 'data/raw' with zero extra disk space or I/O cost.
    """
    print("\nSetting up clean symlinks in data/raw/ ...")
    os.makedirs(RAW_DIR, exist_ok=True)
    
    for original_name, clean_name in CLASS_MAPPING.items():
        src_path = os.path.join(kaggle_path, original_name)
        dest_path = os.path.join(RAW_DIR, clean_name)
        
        # If the symlink or folder already exists, remove it so we can start fresh
        if os.path.exists(dest_path) or os.path.islink(dest_path):
            if os.path.islink(dest_path):
                os.unlink(dest_path)
            else:
                shutil.rmtree(dest_path)
                
        # Create the symbolic link
        os.symlink(src_path, dest_path)
        print(f"  Linked: {clean_name} -> {original_name}")


def preprocess_images():
    """
    Reads images from the clean 'data/raw/' symlinks, resizes them to 224x224, and saves to 'data/processed/'.
    Why we need it: PyTorch CNNs require a uniform input size (224x224). We also catch and skip 
    corrupted images here so the training loop never crashes.
    """
    print("\nStarting image preprocessing (Resizing to 224x224)...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    classes = os.listdir(RAW_DIR)
    
    for class_name in classes:
        class_input_dir = os.path.join(RAW_DIR, class_name)
        class_output_dir = os.path.join(PROCESSED_DIR, class_name)
        
        os.makedirs(class_output_dir, exist_ok=True)
        image_files = os.listdir(class_input_dir)
        
        print(f"  Processing {class_name} ({len(image_files)} images)...")
        
        for filename in image_files:
            input_path = os.path.join(class_input_dir, filename)
            output_path = os.path.join(class_output_dir, filename)
            
            # Read image as BGR
            image = cv2.imread(input_path)
            
            if image is None:
                print(f"    [!] Skipping unreadable file: {filename}")
                continue
                
            # Resize and save (maintaining BGR format)
            image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(output_path, image_resized)
            
    print("\nPreprocessing complete! All data is ready in data/processed/.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    kaggle_data_path = get_kaggle_dataset_path()
    setup_raw_symlinks(kaggle_data_path)
    preprocess_images()
