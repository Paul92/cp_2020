import os
import sys
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util.util import tensor2im, save_image
import torch
import argparse
import keras
import cv2
import argparse
from skimage.transform import resize
import numpy as np

directions = {0: 'N', 1: 'NE', 2: 'E', 3: 'SE', 4: 'S', 5: 'SW', 6: 'W', 7: 'NW'}

religthing_model_names = {'N': '4500_allDirToN_pix2pix',
                          'NE': '4500_allDirToNE_pix2pix',
                          'E': '4500_allDirToE_pix2pix',
                          'SE': '4500_allDirToSE_pix2pix',
                          'S': '4500_allDirToS_pix2pix',
                          'SW': '4500_allDirToSW_pix2pix',
                          'W': '4500_allDirToW_pix2pix',
                          'NW': '4500_allDirToNW_pix2pix'}

parser = argparse.ArgumentParser(description='Light direction classifier')

parser.add_argument('--input', type=str, help='Path to input image')
parser.add_argument('--model', type=str, default='./models/small_cnn.h5', help='Path to light classification model')

parser.add_argument('--direction', type=str, default=None, help='Direction to do relighting. Can be N, NE, E, SE, S, SW, W, NW.')
parser.add_argument('--direction_image', type=str, default=None, help='Path to the image used to determine target direction')


def get_direction(args):
    if args.direction == None and args.direction_image == None:
        print('Please choose --direction or --direction_image')
        sys.exit(1)
    elif args.direction_image == None:
        if args.direction not in directions.values():
            print('The chosen direction can be N, NE, E, SE, S, SW, W, NW.')
            sys.exit(1)
        return args.direction
    else:
        return classify_direction(args.direction_image, args.model)


def classify_direction(input_image, model_path):
    model = keras.models.load_model(args.model)
    image = np.array([resize(cv2.imread(args.input, cv2.IMREAD_COLOR), (256, 256)).astype('float32') / 255])
    prediction = model.predict(image)

    direction = np.where(prediction[0] == np.max(prediction[0]))[0][0]
    return directions[direction]


if __name__ == '__main__':

    args = parser.parse_args()
    direction = get_direction(args)
    relighting_model_name = religthing_model_names[direction]


    print('Using direction', direction)
    print('Using relighting model', relighting_model_name)


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
                             name=relighting_model_name,
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

