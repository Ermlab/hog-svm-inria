import os
pos_img_dir = 'INRIAPerson/train_64x128_H96/pos/'
neg_img_dir = 'INRIAPerson/train_64x128_H96/neg/'

pos_img_files = os.listdir(pos_img_dir)
neg_img_files = os.listdir(neg_img_dir)

from skimage import data, color, exposure
from skimage.feature import hog

pos_filepath = pos_img_dir + pos_img_files[0]
pos_img = data.imread(pos_filepath,as_grey=True)
neg_filepath = neg_img_dir + neg_img_files[0]
neg_img = data.imread(neg_filepath,as_grey=True)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 4))

PERSON_WIDTH = 64
PERSON_HEIGHT = 128
leftop = [16,16]
rightbottom =  [16+PERSON_WIDTH,16+PERSON_HEIGHT]

pos_roi = pos_img[leftop[1]:rightbottom[1],leftop[0]:rightbottom[0]]
fd, pos_hog_image = hog(pos_roi, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2), visualise=True)
neg_roi = neg_img[leftop[1]:rightbottom[1],leftop[0]:rightbottom[0]]
fd, neg_hog_image = hog(neg_roi, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2), visualise=True)
pos_hog_image = exposure.rescale_intensity(pos_hog_image, in_range=(0, 0.1))
neg_hog_image = exposure.rescale_intensity(neg_hog_image, in_range=(0, 0.1))


plt.subplot(141).set_axis_off()
plt.imshow(pos_roi, cmap=plt.cm.gray)
plt.title('Positive image 0')
plt.subplot(142).set_axis_off()
plt.imshow(pos_hog_image, cmap=plt.cm.gray)
plt.title('Postive HOG')

plt.subplot(143).set_axis_off()
plt.imshow(neg_roi, cmap=plt.cm.gray)
plt.title('Negative image 0')
plt.subplot(144).set_axis_off()
plt.imshow(neg_hog_image, cmap=plt.cm.gray)
plt.title('Negative HOG')
plt.show()
