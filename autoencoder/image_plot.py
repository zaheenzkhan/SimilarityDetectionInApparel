import os
from glob import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_paths = glob(os.path.join("./data", '*.jpg'))
image_shape=(1,258*258)

for image_file in image_paths:
    # Re-size to image_shape
    # image = scipy.misc.imresize(scipy.misc.imread(image_file),image_shape)
    gray = cv2.imread(image_file,0)
    print(gray)
    img = np.array(gray)
    print(img.shape)
    stacked_img = np.stack((img,) * 3, axis=-1)
    print(stacked_img.shape)
    # rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # print(rgb.shape)
    # gray = cv2.resize(image, image_shape)

# You may need to convert the color.
im_pil=Image.fromarray(stacked_img)
if im_pil.mode != 'RGB':
    im_pil = im_pil.convert('RGB')

im_pil.save("img1.png", "PNG")

f, a = plt.subplots(1, 1, figsize=(258, 258))
a.imshow(img)

fig = plt.figure()
fig.savefig('full_figure.png')
