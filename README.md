#Master thesis: 
##Deep Learning-Powered Image Super-resolution implementation on embedded device. 
WHAT?  - Image super-resolution is the process of enhancing a high-resolution version of an image from a lower-resolution counterpart.

WHY? - Due to the limitations of camera hardware, camera pose, limited bandwidth, varying illumination conditions, and occlusions, the quality of the surveillance feed is significantly degraded at times, thereby compromising monitoring of behaviour, activities, and other sporadic information in the scene.

USE CASE: In recent years, there has been notable advancement in image super-resolution, primarily driven by deep learning techniques. This category of image processing, vital in computer vision and related fields, finds applications in diverse areas such as medical imaging, anomaly detection, edge detection, semantic image segmentation, digit recognition, and scene recognition. 

CHALLENGES: Despite these significant strides, a persistent challenge remains how to effectively recover finer texture details, and determining the optimal upscaling factor for the image? 

AIM: This thesis aims to address this issue by delving into the intricacies of upscaling, employing a thorough evaluation utilizing both subjective (perceptual loss) and objective metrics (MSE). The findings from this research endeavour aspire to not only contribute to the ongoing exploration of image super-resolution but also to serve as a catalyst for the deep learning community. Specifically, it encourages the integration of Image Super-Resolution (ISR) as a pre-processing step in various vision tasks, particularly when dealing with low-resolution input images.
Tasks:
1)	Literature review to understand the current state of research, pinpointing key challenges in the field.
2)	Use and enhance the existing state-of-the-art method, followed by training to attain optimal performance on an available dataset. {Set5, Set14 or BSD100}
3)	Compare with existing methods and determine an optimal upscaling factor for the image resolution.
4)	Evaluate the algorithm's quality and efficacy using a diverse set of metrics, encompassing both subjective and objective measures **.
5)	Make the model applicable (inference time and accuracy) on an embedded device (e.g. Nvidia Jetson).
6)	Integrate it in the (embedded) anomaly detection pipeline (pre-processing), then execute the complete system on the device to compare both the inference time and accuracy metrics to the original pipeline.

** Subjective measures include mean squared reconstruction error between ground truth and HR image. However, the ability of MSE to capture perceptually relevant differences such as high texture details is very limited as they are defined on pixel wise image differences. Hence perceptual loss is more significant. Perceptual loss = content loss + adversarial loss
