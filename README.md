# 2D Image Relighting with Image-to-Image Translation
Computational Photography, Spring 2020, EPFL

## Context

Recent advancements in machine learning have made possible to tackle novel and
complex problems in many fields, including computational photography. With the
advent of Generative Adversarial Networks (GANs), a finer level of control in
manipulating various features of an image has become possible. One example of
such fine manipulation is changing the position of the light source in a scene.
This is fundamentally an ill-posed problem, since it requires understanding the
scene geometry to generate proper lighting effects. Despite these challenging
conditions, visually consistent results can nowadays be produced using modern
neural networks

## Getting Started
### Prerequisites

 - Linux or MacOs
 - Python 3
 - CPU or NVIDIA GPU + cuDNN

### Installation
1. To clone the git repository and the necessary submodules run the following
commands:
```
git clone --recursive https://github.com/paul92/cp_2020.git
cd cp_2020
```
```
git pull --recurse-submodules
```

 - **pix2pix: [Project](https://phillipi.github.io/pix2pix/) | [Repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | [Paper](https://arxiv.org/pdf/1611.07004.pdf)**

2. In order to install the dependencies of the project in a safe manner, first
create a virtual environment using conda:

	```
	cd pytorch-CycleGAN-and-pix2pix
	conda env create -f environment.yml
	```

 - **VIDIT: [Repository](https://github.com/majedelhelou/VIDIT) |  [Paper](https://arxiv.org/pdf/2005.05460.pdf)**

3. The VIDIT dataset can be found and downloaded from [this repository](https://github.com/majedelhelou/VIDIT).

4. Finally, install the following additional dependencies:
```
conda install keras opencv scikit-image
```

## Using the full pretrained solution

Firstly, make sure to add the `pix2pix` directory project (which was cloned
with the submodules) to the PYTHONPATH:

```
export PYTHONPATH="${PYTHONPATH}:pytorch-CycleGAN-and-pix2pix"
```

Download the pretrained models from
[here](https://drive.google.com/drive/folders/1Rz8aJ0aqsg1HZe5Sltx9LUylQsfmwpki?usp=sharing)
and save them in the root directory of the repository, in the `checkpoints`
directory.

The full solution can be run using the `run.py` script.

The `run.py` script changes the lighting direction in all the images from
the given directory. The taget lighting direction can be passed as a command
line agument (from one of the eight possible directions - N, NE, S, SE, W, NW,
SW, E):

```
python run.py --direction W --input images_directory --output output_directory
```

or determined automatically from another image using a CNN classifier:

```
python run.py --direction_image direction_image.png --input images_directory --output output_directory
```

All the processed images will be placed in the directory indicated by the `output\_directory` flag. Check `python run.py --help` for more options.


## Train new relighting model

The training of the relighting models is based on the pix2pix framework. We provide a custom data preparation script.

 - The input and target images should be side by side. To merge them, run the following command:
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
