import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.callbacks import Callback

IM_SIZE = 256
IM_CHANNELS = 3

def load_pixels(path_to_image_file, image_width, image_height):
    im = Image.open(path_to_image_file).resize((image_width, image_height), Image.BICUBIC)
    pixels = np.array(im)
    return pixels


def save_pixels(path_to_image_file, image_array):
    im_out = Image.fromarray(image_array, 'RGB' )
    im_out.save(path_to_image_file)


# Takes care of clipping, casting to int8, etc.
def save_ndarray(path_to_outfile, x, width = IM_SIZE, height = IM_SIZE, channels = IM_CHANNELS):

    # TODO Figure out the mistake in building input matrices that makes this necessary!
    x = np.rot90(x, k=0)

    out_arr = np.clip(x, 0, 255)
    out_arr = np.reshape(out_arr, (width, height, channels), 1)

    save_pixels(path_to_outfile, out_arr.astype(np.int8))


# Create suitable training matrix
def map_imagematrix_to_tuples(im):

    image_height = im.shape[0]
    image_width = im.shape[1]

    # One row per pixel
    X = np.zeros((image_width * image_height, 2))

    # Fill in y values
    X[:,0] = np.repeat(range(0, image_height), image_width, 0)

    # Fill in x values
    X[:,1] = np.tile(range(0, image_width), image_height)

    # Prepare target values
    Y = np.reshape(im, (image_width * image_height, 3))

    return (X, Y)




image_matrix = load_pixels("head.jpg", IM_SIZE, IM_SIZE)
#save_pixels("out.jpg", test_im)

X,Y = map_imagematrix_to_tuples(image_matrix)

model = Sequential()

# Inputs: x and y
model.add(Dense(20, input_dim=2, activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))
model.add(Dense(20,  activation="relu"))

# Outputs: r,g,b
model.add(Dense(3))

model.compile(loss='mean_squared_error',
               optimizer='adagrad')

# Let's see the how the output changes as the model trains
class training_monitor(Callback):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        cur_img = model.predict(X)
        save_ndarray("image_epoch_" + str(self.epoch) + ".jpg", cur_img)
        self.epoch = self.epoch + 1

image_progress_monitor = training_monitor()
model.fit(X, Y, nb_epoch = 1000, batch_size = 100, callbacks=[image_progress_monitor], shuffle=True)

learnt_image = model.predict(X)
save_ndarray("final_image.jpg", learnt_image)
