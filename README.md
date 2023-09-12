## LRFNet

Code and result about LRFNet (BMVC 2023)

Label-guided Real-time Fusion Network for RGB-T Semantic Segmentation

![model](https://github.com/KevinBHLin/LRFNet/blob/main/picture/model.png)
## Requirements

Python 3.9.13 Pytorch 1.12.0, Cuda 11.3, TensorboardX 2.0, opencv-python

## Dataset
The datasets can be found in [dataset](https://pan.baidu.com/s/1FVar8L6ihvQfx7xNcNumKw) 提取码: 6wne
## Result
Pretrained models and Predict maps can be found in [checkpoint](https://pan.baidu.com/s/15HrIL4fyxIafFkQQ5B6hPQ) 提取码: bear

## Testing
Download the checkpoints and save in ./checkpoint/

<span class="highlight">python run_demo_MFNet.py --data_dir [your MFdataset data_dir]</span>

<mark>python run_demo_PST.py --data_dir [your PST900 data_dir]<mark>
## Acknowledgement
The implement of this project is based on the code of ‘RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes (IEEE RAL)’ proposed by Yuxiang Sun et all.

## Contact
Further discussion: linzr9@mail2.sysu.edu.cn or linbaihong111@126.com.


## Citation

@INPROCEEDINGS{LPFNet_BMVC_2024,  
	author={Lin, Z. and Lin, B. and Guo, Y.},  
	booktitle={British Machine Vision Conference (BMVC)},   
	title={Label-guided Real-time Fusion Network for RGB-T Semantic Segmentation},   
	year={2023}
}


 
 
