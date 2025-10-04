from keras.applications import ResNet101
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
from Sub_Functions.Structural_pattern import StructuralPattern
import numpy as np

# loading the ResNet101 model
model = ResNet101(weights='imagenet', include_top=True)
# creating a new model with only the last layer of the ResNet101 model
model = Model(inputs=model.input, outputs=model.layers[2].output)


def Deep_Structural_Pattern(img):
    # resize the image to 224x224
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    if len(img.shape) == 2:  # H,W only
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # make it 3 channels

    img = cv2.resize(img, (224, 224))
    # predicting the output of the last layer of the ResNet101 model
    input1 = np.expand_dims(img, axis=0)
    # output image
    output = model.predict(input1)
    # squeezing to remove the batch size from the image
    output = np.squeeze(output)
    # creating instance of StructuralPattern class
    SP = StructuralPattern(output[:, :, 2])
    # getting the structural pattern
    final_out = SP.get_structural_pattern()
    # returning the final output
    final_out = cv2.resize(final_out, (48, 48))
    # if final_out is not None and final_out.size > 0:
    #     final_out = (final_out/255).astype(np.uint8) if final_out.max() <= 255 else final_out.astype(np.uint8)
    final_out = final_out / 255.0
    return final_out


# import cv2
# img=cv2.imread("../Dataset/CKPLUS/ck/CK+48/anger/S010_004_00000017.png")
# out=Deep_Structural_Pattern(img)
# print(out)
# plt.imshow(out,cmap="gray")
# plt.show()

# img = cv2.imread("../Dataset/CKPLUS/ck/CK+48/anger/S010_004_00000017.png")
# out = Deep_Structural_Pattern(img)
# plt.subplot(1, 2, 1)
# plt.imshow(out, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(img)
# plt.show()
