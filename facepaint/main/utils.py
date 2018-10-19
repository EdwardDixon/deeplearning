import numpy as np
from PIL import Image


def load_pixels(path_to_image_file, image_width, image_height):
    im = Image.open(path_to_image_file).resize((image_width, image_height), Image.BICUBIC)
    pixels = np.array(im)
    return pixels


# Create suitable training matrix
def map_imagematrix_to_tuples(im):

    image_height = im.shape[0]
    image_width = im.shape[1]

    # One row per pixel
    X = np.zeros((image_width * image_height, 2))

    # Fill in y values
    X[:,1] = np.repeat(range(0, image_height), image_width, 0)

    # Fill in x values
    X[:,0] = np.tile(range(0, image_width), image_height)


    # Normalize X
    X = X - X.mean()
    X = X / X.var()

    # Prepare target values
    Y = np.reshape(im, (image_width * image_height, 3))

    return (X, Y)

