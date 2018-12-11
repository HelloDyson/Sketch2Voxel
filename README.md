# Pixel2Voxel (pytorch)
Final project repo of 6.869 Computer Vision MIT

Original 3D-R2N2 with Theano: https://github.com/chrischoy/3D-R2N2

Original CycleGAN/Pix2pix page: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# Installation

This code tests on Ubuntu 18.04 with CUDA 9.0 but it should work on Windows/ Mac /Linux os

	git clone this repository
	cd Pytorch-3dr2n2
	pip install -r pix2pix/requirements.txt
	
install PyTorch in https://pytorch.org/get-started/locally/

# Dataset
Download the following dataset of 3D-R2N2 and save in data

ShapeNet rendered images http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz

ShapeNet voxelized models http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

# Download CycleGan/pix2pix dataset
	cd pix2pix
	bash ./scripts/download_pix2pix_model.sh edge2shoes_pretrained
	bash ./datasets/download_pix2pix_dataset.sh edge2shoes

# Test 3D reconstruction
	open "3D-R2N2 3D Reconstruction.ipynb" and run all
	
# Process data into pytorch dataloader format
	open "Data preparing.ipynb" and run all

# Train on PyTorch dataloader and model
	open "Train on Pytorch.ipynb" and run all
