# colorCycleGAN-PyTorch
This is the code (in PyTorch) for our paper [Single Image Colorization via Modified CycleGAN](https://ieeexplore.ieee.org/document/8803677)，accepted in *ICIP 2019*, which allows using unpaired images for training and reasonably predict corresponding color distribute of grayscale image in RGB color space.

Note: The pkl-weight in the dir ```/checkpoints``` corrupted during the upload. I’m sorry I didn’t check it in time after uploading. I will update if I have time.
## Prerequisites
Linux

Python 3

CPU or NVIDIA GPU + CUDA CuDNN

## Datasets

The color domain data in the paper is randomly selected from the [PASCAL VOC](https://pjreddie.com/projects/pascal-voc-dataset-mirror/), and grayscaled color domain data to gray domain data.
You can build your own dataset by setting up the following directory structure:

    ├── datasets                  
    |   ├── src_data         # gray
    |   |   ├── train
    |   |   ├── test 
    |   ├── tgt_data         # color
    |   |   ├── train  
    |   |   ├── test 

## Running 
- Training
```
python colorization.py
```
- Testing
```
python test.py
```
## Reference
If you find the code useful, please cite our paper:
```
@INPROCEEDINGS{8803677,
  author={Xiao, Yuxuan and Jiang, Aiwen and Liu, Changhong and Wang, Mingwen},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)}, 
  title={Single Image Colorization Via Modified Cyclegan}, 
  year={2019},
  volume={},
  number={},
  pages={3247-3251},
  doi={10.1109/ICIP.2019.8803677}}
  ```
