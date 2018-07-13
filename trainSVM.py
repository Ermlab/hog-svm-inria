import os
from skimage import data, color, exposure
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(16, 4))

PERSON_WIDTH = 64
PERSON_HEIGHT = 128
leftop = [16, 16]
rightbottom = [16 + PERSON_WIDTH, 16 + PERSON_HEIGHT]

pos_img_dir = 'INRIAPerson/train_64x128_H96/pos/'
neg_img_dir = 'INRIAPerson/train_64x128_H96/neg/'
pos_img_files = os.listdir(pos_img_dir)
neg_img_files = os.listdir(neg_img_dir)

X = []
y = []
print('start loading ' + str(len(pos_img_files)) + ' positive files')
for pos_img_file in pos_img_files:
    pos_filepath = pos_img_dir + pos_img_file
    pos_img = data.imread(pos_filepath, as_grey=True)
    pos_roi = pos_img[leftop[1]:rightbottom[1], leftop[0]:rightbottom[0]]
    fd = hog(pos_roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)
    X.append(fd)
    y.append(1)
print('start loading ' + str(len(neg_img_files)) + ' negative files')
for neg_img_file in neg_img_files:
    neg_filepath = neg_img_dir + neg_img_file
    neg_img = data.imread(neg_filepath, as_grey=True)
    neg_roi = neg_img[leftop[1]:rightbottom[1], leftop[0]:rightbottom[0]]
    fd = hog(neg_roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)
    X.append(fd)
    y.append(0)

## covert list into numpy array
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

from sklearn import svm
from sklearn.externals import joblib

print('start learning SVM.')
lin_clf = svm.LinearSVC()
lin_clf.fit(X, y)
#clf = svm.SVC()
#clf.fit(X, y)
print('finish learning SVM.')
print(lin_clf.fit(X,y))
print(lin_clf.score(X,y))

joblib.dump(lin_clf, 'person_detector.pkl', compress=9)
