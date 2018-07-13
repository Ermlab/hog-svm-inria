import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    image = Image.open('INRIAPerson/Test/pos/crop001501.png')
    # Convert to numpy array, OpenCV works with numpy arrays
    image_arr = np.asarray(image)

    # Initialize the HOG descriptor
    hog = cv2.HOGDescriptor()
    # Set an SVM detector for the given hog descriptor
    # In this case we use the integrated in OpenCV people detector
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detect the pedestrians in the input image
    # detections - numpy array of possible detections in the format:
    #   (starting point x; starting point y; width; height of detection)
    # weights - numpy array of weights (confidences) for each detection
    detections, weights = hog.detectMultiScale(image_arr)

    # Convert to a normal Python list
    detections_rectangles = detections.tolist()

    # Start drawing rectangles for each possible detection
    draw = ImageDraw.Draw(image)
    for x, y, w, h in detections_rectangles:
        draw.rectangle(
            [x, y, x + w, y + h], outline=(255, 0, 0))

    image.show()

if __name__ == '__main__':
    main()