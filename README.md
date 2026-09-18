# SMPAN
SMPAN: Statistics-Guided Multi-Scale Pyramid Attention Network for Remote Sensing Scene Classification

All images were resized to 256×256 and normalized using the ImageNet mean [0.485,0.456,0.406] and standard deviation [0.229,0.224,0.225]. 
RandAugment and Mixup were applied using a dataset-specific hyperparameter (N,M,α)，The triplets were (2,8,0.25) for UCM, (2,10,1.0) for AID, and (3,11,1.2)
for NWPU.

## Datasets

The experiments were conducted on the following publicly available datasets:

- [UC Merced Land Use Dataset (UCM)](https://vision.ucmerced.edu/datasets/)
- [Aerial Image Dataset (AID)](https://captain-whu.github.io/AID/)
- [NWPU-RESISC45](https://gcheng-nwpu.github.io/#Datasets)

Please download the datasets from their official project pages and organize
each dataset with one subdirectory per scene category.


