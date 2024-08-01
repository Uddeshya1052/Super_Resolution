Custom Training Dataset Folder
In this folder, we store the custom training dataset, which is essential for training the Super-Resolution Generative Adversarial Network (SRGAN) model. This dataset is specifically curated to enhance the performance of the SRGAN by providing high-quality, high-resolution images as well as their corresponding low-resolution counterparts. The dataset is organized into two main subfolders:

1. HR_Image (High-Resolution Images) Folder
The HR_Image folder contains the ground truth images, which are high-resolution (HR) images. These images represent the desired output that the SRGAN model aims to generate after processing low-resolution images. Here are the key features of the images in this folder:

High Quality: These images are of high resolution, with fine details and sharpness. They serve as the benchmark for the model to learn from.

Ground Truth: They are used as the ground truth during the training process, guiding the model to generate images that closely resemble these high-resolution standards.

Diverse Content: The folder should ideally contain a diverse range of images to ensure that the model learns various features and generalizes well to different types of images.

Format and Size: Images are usually stored in widely-used formats like JPEG or PNG, and they should maintain consistent dimensions suitable for training.

2. LR_Image (Low-Resolution Images) Folder
The LR_Image folder houses the low-resolution (LR) images, which are the degraded versions of the high-resolution images. These images are inputs for the SRGAN model during training. Here’s what to know about these images:

Reduced Quality: These images have lower resolution and are less detailed than their high-resolution counterparts. The reduction in quality can be due to various factors such as downsampling, blurring, or compression artifacts.

Model Input: The SRGAN takes these low-resolution images as input and attempts to reconstruct them into high-resolution versions similar to those in the HR_Image folder.

Consistency with HR Images: Each low-resolution image in this folder should correspond directly to a high-resolution image in the HR_Image folder, allowing the model to learn the mapping from LR to HR effectively.

Varied Degradation: The images might have varying degrees of degradation to simulate different real-world scenarios where super-resolution is necessary
