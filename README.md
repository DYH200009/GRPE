# GRPE: High-fidelity 3D Gaussian Reconstruction for Plant Entities
## 📖 Abstract
Plant models hold significant importance for constructing virtual worlds. Currently, there is a lack of algorithms capable of achieving high-fidelity reconstruction of plant surfaces.
In this paper, we propose a unified architecture to reconstruct high-fidelity 3D surface models and render realistic plant views, which enhances geometric accuracy during Gaussian densification and mesh extraction from 2D images.
The algorithm initially employs large vision models for semantic segmentation to extract plant objects from 2D RGB images, generating sparse mappings and camera poses. Subsequently, these images and point clouds are processed to produce Gaussian ellipsoids and 3D textured models, with the algorithm detecting smooth regions during densification.
To ensure precise alignment of the Gaussians with object surfaces, the algorithm incorporates a robust 3D Gaussian splatting method that includes an outlier removal algorithm. Compared to traditional techniques, this approach yields models that are more accurate and exhibit less noise.
Experimental results demonstrate that our method outperforms existing plant modeling approaches, surpassing surpassing existing methods in terms of PSNR, LPIPS, and SSIM metrics. The high-precision annotated plant dataset and system code are available at \url{https://github.com/DYH200009/GRPE}.
<div align="center">
<img width="30%" alt="image" src="img/fig.gif">
<img width="30%" alt="image" src="img/grape.gif">
<img width="30%" alt="image" src="img/tomato.gif">
</div>
<p align="center"><strong>GRPE (Ours)'s mesh results on some classes of the GRPE dataset.</strong></p>





## 🔧 Setup of GRPE
### 1. Clone the repo.
```
# https
git clone https://github.com/DYH200009/GRPE
# or ssh
git clone git@github.com:DYH200009/GRPE.git
```

### 2. Environment setup.
If you have an environment used for 3dgs, use it. 
If not, please refer to the environment configuration of [3DGS environment](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#local-setup) 
The ``environment.yml`` file for 3DGS has been provided in our repo.

Additionally, you also need to install the 
``open3d`` ``scipy`` ``matplotlib`` ``pillow``
libraries.

```
# If you have already installed the 3dgs environment,
# please activate the environment and execute the following command :
conda activate gaussian_splatting
pip install open3d scipy matplotlib pillow
cd submodules/diff-gaussian-rasterization/
pip install -e .
cd ../simple-knn/
pip install -e .
cd  ../../
```
In addition, you need to install SuGaR's running environment separately 
If you have an environment used for SuGaR, use it. 
If not, please refer to the environment configuration of [SuGaR environment](https://github.com/Anttwo/SuGaR?tab=readme-ov-file#installation) 

### 3. Download the demo dataset
- Create a new ``data`` folder
- Download the file ([GRAPE-Dataset](https://drive.google.com/file/d/153DR5sdkT8pJUXNnMED4pkLfhWcas4MW/view?usp=sharing)).
- Unzip it to ``data`` folder.

### 4. Run the codes 
In order to run our code, 
```
# python train.py -s [path_to_dataset] -m [path_to_output] --eval
# cd MeshExtractor
# python train.py -s [path_to_dataset] -r <"density" or "sdf"> -c [path_to_output]
```
Run demo:
```
python train.py -s data/GRAPE/grape -m outputCKT/grape 
cd MeshExtractor
python train.py -s data/GRAPE/grape -r density -c outputCKT/grape 
```



## ⭕️ Acknowledgment
This project is based on [GeoGussian](https://github.com/yanyan-li/GeoGaussian) and [SuGaR](https://github.com/Anttwo/SuGaR) 
Thanks for the contribution of the open source community.





