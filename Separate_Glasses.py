import cv2
import numpy as np

def separate_eyeglasses(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a mask initialized with zeros
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define the region of interest (ROI) coordinates
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces and mark the ROI in the mask
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # Apply GrabCut algorithm to refine the mask
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, image.shape[1] - 1, image.shape[0] - 1)  # Set a rectangle covering the entire image
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Create a mask where all likely foreground and possible foreground regions are marked
    mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to extract the eyeglasses
    eyeglasses = image * mask_2[:, :, np.newaxis]

    # Display the original image, mask, and extracted eyeglasses
    cv2.imshow('Original Image', image)
    cv2.imshow('Mask', mask * 50)
    cv2.imshow('Eyeglasses', eyeglasses)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the path to the image
image_path = 'image.jpg'

# Call the function to separate eyeglasses from the face in the image
separate_eyeglasses(image_path)
