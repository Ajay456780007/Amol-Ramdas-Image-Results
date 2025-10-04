import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import ResNet50
from keras.models import Model
from Sub_Functions.Local_Binary_pattern import local_binary_pattern

# Loading the ResNet50 model and extracting the second last layer
model = ResNet50(include_top=False, weights="imagenet")
# this line extracts the output from the second layer of Resnet50
model2 = Model(inputs=model.input, outputs=model.layers[2].output)


# Defining Function
def deep_edge_based_text_pattern(img):
    # Converting the image from BGR format to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # As thw required shape for the Resnet50 id 224,224,3 we are resizing the image to 224,224
    img = cv2.resize(img, (224, 224))
    # Resnet has batch shape inorder to add the batch size we are adding a new axis
    input1 = np.expand_dims(img, axis=0)
    # predicting the output from the second last layer of Resnet50
    output = model2.predict(input1)
    # Squeezing the output to remove the batch size
    output = np.squeeze(output)
    feat = output[:, :, 1]  # pick one channel
    feat_norm = cv2.normalize(feat, None, 0, 255, cv2.NORM_MINMAX)
    feat_uint8 = feat_norm.astype("uint8")
    # Apply Gaussian blur
    gaussian_blur = cv2.GaussianBlur(feat_uint8, (3, 3), 0)
    # Applying canny edge detection to the blurred output
    edges = cv2.Canny(gaussian_blur, 100, 200)
    # Applying Local Binary pattern to the canny edges
    LBP = local_binary_pattern(edges)
    # resizing the image to match the output for the final concating
    LBP = cv2.resize(LBP, (50, 50))
    LBP = LBP / 255.0
    return LBP


# img = cv2.imread("../Dataset/CKPLUS/ck/CK+48/anger/S010_004_00000017.png")
# out=deep_edge_based_text_pattern(img)
# plt.subplot(1,2,1)
# plt.imshow(out,cmap="gray")
# plt.subplot(1,2,2)
# plt.imshow(img)
# plt.show()
