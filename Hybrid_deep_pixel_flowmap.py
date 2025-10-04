import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications import ResNet101
from keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the ResNet101 model with weights (include_top True or False depends on your need)
model = ResNet101(include_top=True, weights='imagenet')

# Redefine model output to intermediate layer 2
model2 = Model(inputs=model.input, outputs=model.layers[2].output)

def hybrid_deep_pixel_flowmap(img):
    # Ensure the input image has 3 channels in RGB order
    if len(img.shape) == 2:  # grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # Resize image to 224x224 (required by ResNet)
    image = cv2.resize(img, (224, 224))

    # Convert to float32, preprocess, and add batch dimension
    image = preprocess_input(np.expand_dims(image.astype(np.float32), axis=0))

    # Predict the output of intermediate layer through model2
    out = model2.predict(image)

    # Remove batch dimension
    output = np.squeeze(out, axis=0)

    # Select the 2nd channel (index 1) of the feature maps for processing
    final_out = output[:, :, 2]

    # Compute gradients (Sobel edge detection)
    grad_x = cv2.Sobel(final_out, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(final_out, cv2.CV_32F, 0, 1, ksize=3)

    # Compute magnitude and angle of gradients
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

    # Normalize magnitude to 0-255 for visualization
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Resize normalized magnitude to desired smaller output size (50x50)
    resized_magnitude = cv2.resize(magnitude_norm, (50, 50))

    # Normalize again and convert to uint8
    normalized_final = cv2.normalize(resized_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    final_uint8 = normalized_final.astype(np.uint8)

    # Convert to float32 normalized between 0 and 1
    final_float = final_uint8 / 255.0

    return final_float


# import cv2
# img=cv2.imread("../Dataset/CKPLUS/ck/CK+48/anger/S010_004_00000017.png")
# out=hybrid_deep_pixel_flowmap(img)
# print(out)
# plt.imshow(out,cmap="gray")
# plt.show()

# img = cv2.imread("../Dataset/CKPLUS/ck/CK+48/anger/S010_004_00000017.png")
# out = hybrid_deep_pixel_flowmap(img)
# plt.subplot(1, 2, 1)
# plt.imshow(out, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(img)
# plt.show()