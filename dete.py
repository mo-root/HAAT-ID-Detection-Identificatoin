import cv2

face_cascade = cv2.CascadeClassifier('face_detector.xml')
img = cv2.imread('testID.png')

faces = face_cascade.detectMultiScale(img, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
# Export the result
cv2.imshow("face_detected.png", img)

cropped = img[y: y + h, x: x + w]

# cv2.imshow('image', cropped)
cv2.waitKey(0)

print('Successfully saved')
