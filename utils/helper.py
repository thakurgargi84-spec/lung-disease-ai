import cv2
import os

# Folder paths
left_folder = "dataset/segmentation/masks/left mask"
right_folder = "dataset/segmentation/masks/right mask"

# Output folder
output_folder = "dataset/segmentation/combined_masks"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Get all filenames
left_files = os.listdir(left_folder)

# Loop through all mask images
for filename in left_files:

    # Create full paths
    left_path = os.path.join(left_folder, filename)
    right_path = os.path.join(right_folder, filename)

    # Read masks in grayscale
    left_mask = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_mask = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    # Combine masks
    combined_mask = cv2.bitwise_or(left_mask, right_mask)

    # Save combined mask
    save_path = os.path.join(output_folder, filename)

    cv2.imwrite(save_path, combined_mask)

print("✅ Combined masks created successfully!")