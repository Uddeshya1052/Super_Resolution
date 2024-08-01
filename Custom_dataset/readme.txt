📂 Custom Training Dataset Folder
Welcome to the Custom Training Dataset Folder, the cornerstone of our Super-Resolution Generative Adversarial Network (SRGAN) project. This folder is meticulously organized to ensure that our SRGAN model is trained with the best possible data, empowering it to transform low-resolution images into stunning, high-resolution masterpieces.

Our dataset is thoughtfully curated and divided into two key subfolders:

1. 🖼️ HR_Image (High-Resolution Images) Folder
The HR_Image folder is the heart of our dataset, containing the high-resolution (HR) images that serve as the ground truth for our SRGAN model. These images represent the gold standard that the model aspires to achieve when processing low-resolution inputs.

Key Features:

       ✨ High Quality:
        These images are captured in exquisite detail, featuring fine textures and sharpness that set the benchmark for the model to learn from.
        As the gold standard, they guide the model in generating outputs that meet high-resolution excellence.

       🔍 Ground Truth:
        Used as the reference point during training, these images are the target outputs that the model aims to emulate.

       🌈 Diverse Content:
        The folder boasts a rich variety of images, encompassing different scenes, subjects, and styles to ensure the model can generalize across a wide array of image types.

       📏 Format and Size:
        Images are typically in popular formats like JPEG or PNG and maintain consistent dimensions to optimize the training process.

2. 🖼️ LR_Image (Low-Resolution Images) Folder
The LR_Image folder contains the low-resolution (LR) images that are the input for the SRGAN model during training. These images are intentionally degraded to challenge the model to reconstruct them into their high-resolution counterparts.

Key Features:

       🔄 Reduced Quality:
       These images possess lower resolution and detail, mirroring real-world challenges such as downsampling, blurring, or compression artifacts.
       This degradation provides a challenging task for the SRGAN to overcome, refining its capability to enhance image quality.

       🧩 Model Input:
       Serving as the starting point, these images are what the SRGAN processes to transform them into high-resolution images.

      🔗 Consistency with HR Images:
      Each low-resolution image is paired with a high-resolution counterpart in the HR_Image folder, ensuring a direct mapping for the model to learn from.
      This pairing allows the model to effectively learn the transformation from low-resolution to high-resolution.

     🔄 Varied Degradation:
      The dataset features images with varying levels of degradation, simulating different real-world scenarios where super-resolution is required.
      This variation aids in enhancing the model’s robustness and adaptability
