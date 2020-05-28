import os
from PIL import Image

ROOT = "imagesOnly"
TARGET = "pix2pixWMultipleDirections_to_SW"

#LEFT_DIRECTION = "E"
RIGHT_DIRECTION = "SW"

TEMPERATURE = '4500'

def getImages(scene, direction):
    Dir = os.path.join(scene, direction)
    imagesPaths = [os.path.join(Dir, f) for f in os.listdir(Dir)]
    imagesPaths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][5:]))
    images = [Image.open(x) for x in imagesPaths]
    return images


if __name__ == '__main__':

    mainScenes = [os.path.join(ROOT, f) for f in os.listdir(ROOT)]

#  For temperatures
#    scenes = []
#    for scene in mainScenes:
#        scenes += [os.path.join(scene, f) for f in os.listdir(scene)]
#
    scenes = [os.path.join(scene, TEMPERATURE) for scene in mainScenes]

    if not os.path.exists(TARGET):
        os.makedirs(TARGET)

    for scene in scenes:
        for LEFT_DIRECTION in ['E',  'N',  'NE',  'NW',  'S',  'SE',  'SW',  'W']:

            scenetarget = os.path.join(TARGET, scene, LEFT_DIRECTION)
            os.makedirs(scenetarget)
    
            leftImages = getImages(scene, LEFT_DIRECTION)
            rightImages = getImages(scene, RIGHT_DIRECTION)

    
            imageSize = leftImages[0].size
    
            finalSize = (imageSize[0] * 2, imageSize[1])
    
            newIm = Image.new('RGB', finalSize)
            for i in range(len(leftImages)):
                filename = '_'.join(scene.split('/')[-2:]) + '_' + LEFT_DIRECTION + '_' + RIGHT_DIRECTION + '_' + str(i) + '.png'
                print(filename)
                newIm.paste(leftImages[i], (0, 0))
                newIm.paste(rightImages[i], (imageSize[0], 0))
                newIm.save(os.path.join(scenetarget, filename))
    
