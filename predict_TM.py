from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

directory = 'target'
os.mkdir(directory + "/_doc_imgs")
os.mkdir(directory + "/_other_imgs")
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        # print(f)
        image = Image.open(f)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        y_classes = prediction.argmax(axis=-1)
        if(y_classes[0] == 0):
            print(f+ " - Document")
        else:
            print(f+ " - Non Document")




