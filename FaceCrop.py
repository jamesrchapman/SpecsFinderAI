import cv2

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Crop out the faces
for (x, y, w, h) in faces:
    # Create a rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Crop the face region
    face_roi = image[y:y+h, x:x+w]
    
    # Display the cropped face
    cv2.imshow('Face', face_roi)
    cv2.waitKey(0)

# Display the original image with rectangles around the faces
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
