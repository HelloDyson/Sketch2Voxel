# Pytorch-3dr2n2
Final project repo of 6.869 Computer Vision MIT

Original 3D-R2N2 with Theano: https://github.com/chrischoy/3D-R2N2

Original CycleGAN/Pix2pix page: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# Installation

	git clone this repository
	cd Pytorch-3dr2n2
	pip install -r requirements.txt
install PyTorch in https://pytorch.org/get-started/locally/

# Dataset
Download the following dataset of 3D-R2N2 and save in data/
ShapeNet rendered images http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
ShapeNet voxelized models http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

# Download CycleGan/pix2pix dataset via
	bash .pix2pix/scripts/download_pix2pix_model.sh edge2shoes_pretrained
	bash .pix2pix/datasets/download_pix2pix_dataset.sh edge2shoes

# Process data into pytorch dataloader format
	open "Train on Pytorch.ipynb" and run all

# Train on PyTorch dataloader and model
	open "Train on Pytorch.ipynb" and run all

# Test 3D reconstruction
	open "3D-R2N2 3D Reconstruction.ipynb" and run all
