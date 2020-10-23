import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import argparse
from tensorflow.keras.preprocessing import image
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', dest='image',
                    help='Image path to recognize')

args = parser.parse_args()
if args.image == None:
    parser.print_help()
    exit()
    
model = tf.keras.models.load_model('saved_model/model1_88')

img = image.load_img(args.image, target_size=(48, 48))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

if np.argmax(pred) == 0:
    print("Anger (Ofke)")
elif np.argmax(pred) == 1:
    print("Contempt (Kuçumseme)")
elif np.argmax(pred) == 2:
    print("Disgust (İgrenme)")
elif np.argmax(pred) == 3:
    print("Fear (Korku)")
elif np.argmax(pred) == 4:
    print("Happy (Mutlu)")
elif np.argmax(pred) == 5:
    print("Sadness (Keder)")
elif np.argmax(pred) == 6:
    print("Surprise (Saskin)")
