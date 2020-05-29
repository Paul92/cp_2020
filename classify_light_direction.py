import keras
import cv2
import argparse
from skimage.transform import resize
import numpy as np

parser = argparse.ArgumentParser(description='Light direction classifier')

parser.add_argument('--input', type = str, help = 'Path to input image')
parser.add_argument('--model', type = str, default = './models/small_cnn.h5', help = 'Path to model')

opt = parser.parse_args()

model = keras.models.load_model(opt.model)

image = np.array([resize(cv2.imread(opt.input, cv2.IMREAD_COLOR), (256, 256)).astype('float32') / 255])

prediction = model.predict(image)


models = {0: '4500_allDirToN_pix2pix',
          1: '4500_allDirToNE_pix2pix',
          2: '4500_allDirToE_pix2pix',
          3: '4500_allDirToSE_pix2pix',
          4: '4500_allDirToS_pix2pix',
          5: '4500_allDirToSW_pix2pix',
          6: '4500_allDirToW_pix2pix',
          7: '4500_allDirToNW_pix2pix'}

direction = np.where(prediction[0] == np.max(prediction[0]))[0][0]
model = models[direction]

print(model)

