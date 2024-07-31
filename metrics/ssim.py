import os
import csv
import cv2
from skimage.metrics import structural_similarity as compare_ssim

def calculate_ssim(image1_path, image2_path):
    imageA = cv2.imread(image1_path)
    imageB = cv2.imread(image2_path)
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = compare_ssim(grayA, grayB, full=True)
    return score

# Folder containing images
folder_path = "./result"
folder_path2 = "./test_input"

# Output CSV file
output_file = "ssim_results.csv"

# List to store results
results = []

# Get sorted list of filenames in both folders
files1 = sorted(os.listdir(folder_path))
files2 = sorted(os.listdir(folder_path2))

# Iterate over each pair of images in sorted order
for filename1, filename2 in zip(files1, files2):
    image1_path = os.path.join(folder_path, filename1)
    image2_path = os.path.join(folder_path2, filename2)
    # Calculate SSIM
    ssim_score = calculate_ssim(image1_path, image2_path)
    # Append result to list
    results.append((filename1, filename2, ssim_score))

# Write results to CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image1", "Image2", "SSIM"])
    writer.writerows(results)

print("SSIM results saved to", output_file)
