# 2D Image Relighting with Image-to-Image Translation
Computational Photography, Spring 2020, EPFL

## Context

Changing the direction of the light source in a photo is not a trivial task. This task can be even more complicated if we want to change the direction of the light source from any direction to a specific one.

Here we provide our attempt to solve this problem using GANs.
Specifically [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  trained with the [VIDIT](https://github.com/majedelhelou/VIDIT) dataset that contains images of the same scene with different types of light temperature and 8 different light source positions (N, NE, E, SE, S, SW, W, NW).

The results are 8 neural networks trained to be able to change the direction of the light source from any direction to one of the 8 previously mentioned. Additionally, we provide, as a tool, a simple CNN trained to identify the direction of the light in an image.

## Getting Started
### Prerequisites

 - Linux or MacOs
 - Python 3
 - CPU or NVIDIA GPU + cuDNN

### Installation
First, clone this repository an the submodules:
```
git clone --recursive https://github.com/emarazz/cp_2020.git
cd cp_2020
```
 To pull all changes in the repo and the submodules:
```
git pull --recurse-submodules
```

 - **pix2pix: [Project](https://phillipi.github.io/pix2pix/) | [Repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | [Paper](https://arxiv.org/pdf/1611.07004.pdf)**

		Now, install the requirements of the submodule:

	```
	cd pytorch-CycleGAN-and-pix2pix
	conda env create -f environment.yml
	```

 - **VIDIT: [Repository](https://github.com/majedelhelou/VIDIT) |  [Paper](https://arxiv.org/pdf/2005.05460.pdf)**
	 Download the dataset from the project's [repository](https://github.com/majedelhelou/VIDIT).

Finally, go back and manually install the additional packages:
```
cd ..
conda install keras opencv  scikit-image
```

## Using the full pretrained solution

The full solution can be run using the `run.py` script. Make sure to add the `pix2pix` directory to the PYTHONPATH:

```
export PYTHONPATH="${PYTHONPATH}:pytorch-CycleGAN-and-pix2pix"
```

The `run.py` script converts a directory of images from any direction to a single one, given either as a command line argument:

```
python run.py --direction W --input images_directory --output output_directory
```

or determined automatically from another image using a CNN classifier:

```
python run.py --direction_image direction_image.png --input images_directory --output output_directory
```

All the processed images will be placed in outpu\_directory. Check `python run.py --help` for more options.


## Train new relighting model

The training of the relighting models is based on the pix2pix framework. We provide a custom data preparation script.

 - Prepare the data. As required by the pix2pix framework, the input and target images should be side by side. To merge them, run the following command:
	```
	python merger.py  --RIGHT_DIRECTION NW --TARGET ./4500_allDirToNW
	```
	 In this case, the prepared data is for: all directions to `NW` and the output data
	 is under the folder `./4500_allDirToNW` which contains a `/train` and
	 `/test` folders.

	 For more information run: `python merger.py --help`
- Train the model. Go to the submodule folder `pytorch-CycleGAN-and-pix2pix` and run:
	```
	python train.py --dataroot ../allDirToNW --name 4500_allDirToNW_pix2pix --model pix2pix --direction AtoB --n_epochs 1000 --gpu_ids 0
	```
	For more options, please see the [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
	 Best results were obtained with batch size equals 1 (which is the default value),


## Test
To test the model. Be sure to be in `./pytorch-CycleGAN-and-pix2pix` and run:

```
python test.py --dataroot ../4500allDirToNW --name 4500_allDirToNW_pix2pix d--model pix2pix --direction AtoB --gpu_ids 0
```

## Train new classification model

The Jupyter Notebook `LightDirectionClassifier.ipynb` includes the data preparation, training and evaluation of the light direction classifier.
