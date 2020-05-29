import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util.util import tensor2im, save_image
import torch
import argparse


if __name__ == '__main__':

    opt = argparse.Namespace(aspect_ratio=1.0,
                             batch_size=1,
                             checkpoints_dir='./checkpoints',
                             crop_size=256,
                             dataroot='singleimagetest/',
                             dataset_mode='single',
                             direction='AtoB',
                             display_winsize=256,
                             epoch='latest',
                             eval=False,
                             gpu_ids=[0],
                             init_gain=0.02,
                             init_type='normal',
                             input_nc=3,
                             isTrain=False,
                             load_iter=0,
                             load_size=256,
                             max_dataset_size=1000,
                             model='pix2pix',
                             n_layers_D=3,
                             name='4500_allDirToE_pix2pix',
                             ndf=64,
                             netD='basic',
                             netG='unet_256',
                             ngf=64,
                             no_dropout=False,
                             norm='batch',
                             ntest=1000,
                             num_test=50,
                             output_nc=3,
                             phase='test',
                             preprocess='resize_and_crop',
                             results_dir='./results/',
                             suffix='',
                             num_threads=0,   # test code only supports num_threads = 1
                             serial_batches=True,  # disable data shuffling; comment this line if results on randomly chosen images are needed.
                             no_flip=True,    # no flip; comment this line if results on flipped images are needed.
                             display_id=-1,   # no visdom display; the test code saves the results to a HTML file.
                             verbose=False)


    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        print(type(data))
        print(data)
        image = data['A']
        with torch.no_grad():
            output = model.netG(image)

        output = tensor2im(output)
        save_image(output, 'output.png')

