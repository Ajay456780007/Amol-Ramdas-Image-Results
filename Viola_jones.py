import cv2
from matplotlib import pyplot as plt


def Viola_jones(image):
    # Load Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # No face detected, return RGB image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    else:
        # Crop the first face detected
        (x, y, w, h) = faces[0]
        cropped_face = image[y:y+h, x:x+w]

        # Resize to 227x227
        resized_face = cv2.resize(cropped_face, (48,48))

        return resized_face

# import cv2
# img=cv2.imread("../Dataset/CKPLUS/ck/CK+48/disgust/S032_005_00000016.png")
# plt.subplot(121)
# out=Viola_jones(img)
# plt.imshow(out)
# plt.subplot(122)
# plt.imshow(out)
# plt.show()
# print(out.shape)