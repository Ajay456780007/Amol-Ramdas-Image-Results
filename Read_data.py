import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from Feature_Extraction.Deep_Structural_pattern import Deep_Structural_Pattern
from Feature_Extraction.Hybrid_deep_pixel_flowmap import hybrid_deep_pixel_flowmap
from Feature_Extraction.deep_edge_based_text_pattern import deep_edge_based_text_pattern
from Feature_Extraction.Viola_jones import Viola_jones


# Function used to perform Feature Extraction
def feature_Extraction(img):
    # Extracting the deep edge based pattern for the given the image
    DEBTP = deep_edge_based_text_pattern(img)
    #  Extracting Structural Pattern
    DSP = Deep_Structural_Pattern(img)
    # Extracting Hybrid deep pixel flowmap
    HDPF = hybrid_deep_pixel_flowmap(img)
    # returning all the extracted three features by concating
    stacked = np.stack([DEBTP, DSP, HDPF], axis=-1)
    return stacked


def Preprocessing(DB, save=True):
    if DB == "CKPLUS":
        # This line indicates that the preprocessing is started for CKPLUS dataset.
        print("Preprocessing Starts for CKPLUS dataset ...........👍🏽")

        # Initializes folder paths for CKPLUS dataset.
        folder_path1 = r"Dataset/CKPLUS/ck/CK+48/anger"
        folder_path2 = r"Dataset/CKPLUS/ck/CK+48/contempt/"
        folder_path3 = r"Dataset/CKPLUS/ck/CK+48/disgust/"
        folder_path4 = r"Dataset/CKPLUS/ck/CK+48/fear/"
        folder_path5 = r"Dataset/CKPLUS/ck/CK+48/happy/"
        folder_path6 = r"Dataset/CKPLUS/ck/CK+48/sadness/"
        folder_path7 = r"Dataset/CKPLUS/ck/CK+48/surprise/"

        # Reads the image names in the folders
        image_names1 = [f for f in os.listdir(folder_path1)]
        image_names2 = [f for f in os.listdir(folder_path2)]
        image_names3 = [f for f in os.listdir(folder_path3)]
        image_names4 = [f for f in os.listdir(folder_path4)]
        image_names5 = [f for f in os.listdir(folder_path5)]
        image_names6 = [f for f in os.listdir(folder_path6)]
        image_names7 = [f for f in os.listdir(folder_path7)]

        print("The length of angry is:", len(image_names1))
        print("The length of disgust is:", len(image_names2))
        print("The length of fear is:", len(image_names3))
        print("The length of happy is:", len(image_names4))
        print("The length of neutral is:", len(image_names5))
        print("The length of sad is:", len(image_names6))
        print("The length of surprise is:", len(image_names7))

        # Storing all the images names in a list
        collections = [image_names1, image_names2, image_names3, image_names4, image_names5, image_names6, image_names7]
        # Storing all the folder names in a list
        folders = [folder_path1, folder_path2, folder_path3, folder_path4, folder_path5, folder_path6, folder_path7]
        # empty list to store the data
        data_all = []
        # empty list to store the labels
        labels_all = []
        i = 0

        for col1, fold1 in zip(collections, folders):
            print("Currently Processing:", fold1)
            print("The total no of images in ", fold1, "is:", len(col1))
            k = 1
            for name in col1:
                img_path = os.path.join(fold1, str(name))
                img = cv2.imread(img_path)
                # Extracting face
                Viola_j = Viola_jones(img)
                # Extracting features
                out = feature_Extraction(Viola_j)
                # appending the data
                data_all.append(out)
                labels_all.append(i)
                print(f"Completed processing {k}/{len(col1)} images in {fold1}")
                k = k + 1
            i += 1  # incrementing the count to match the label
        if save:
            os.makedirs(f"data_loader/{DB}/", exist_ok=True)
            # saving the data in the npy format
            np.save(f"data_loader/{DB}/features.npy", np.array(data_all))
            # saving the labels in the npy format
            np.save(f"data_loader/{DB}/labels.npy", np.array(labels_all))
        # This line indicates the data saved successfully
        print(f"The data saved successfully for {DB} dataset..................")

    if DB == "FER2013":
        # This line indicates that the preprocessing is started for CKPLUS dataset.
        print(f"Preprocessing Starts for {DB} dataset ...........👍🏽")

        # Initializes folder paths for CKPLUS dataset.
        folder_path1 = r"Dataset/FER2013/train/angry"
        folder_path2 = r"Dataset/FER2013/train/disgust/"
        folder_path3 = r"Dataset/FER2013/train/fear/"
        folder_path4 = r"Dataset/FER2013/train/happy/"
        folder_path5 = r"Dataset/FER2013/train/neutral/"
        folder_path6 = r"Dataset/FER2013/train/sad/"
        folder_path7 = r"Dataset/FER2013/train/surprise/"

        # Reads the image names in the folders
        image_names1 = [f for f in os.listdir(folder_path1)][:500]
        image_names2 = [f for f in os.listdir(folder_path2)][:436]
        image_names3 = [f for f in os.listdir(folder_path3)][:500]
        image_names4 = [f for f in os.listdir(folder_path4)][:500]
        image_names5 = [f for f in os.listdir(folder_path5)][:500]
        image_names6 = [f for f in os.listdir(folder_path6)][:500]
        image_names7 = [f for f in os.listdir(folder_path7)][:500]

        print("The length of angry is:", len(image_names1))
        print("The length of disgust is:", len(image_names2))
        print("The length of fear is:", len(image_names3))
        print("The length of happy is:", len(image_names4))
        print("The length of neutral is:", len(image_names5))
        print("The length of sad is:", len(image_names6))
        print("The length of surprise is:", len(image_names7))
        # Storing all the images names in a list
        collections = [image_names1, image_names2, image_names3, image_names4, image_names5, image_names6, image_names7]
        # Storing all the folder names in a list
        folders = [folder_path1, folder_path2, folder_path3, folder_path4, folder_path5, folder_path6, folder_path7]
        # empty list to store the data
        data_all = []
        # empty list to store the labels
        labels_all = []
        i = 0

        for col1, fold1 in zip(collections, folders):
            print("Currently Processing:", fold1)
            print("The total no of images in ", fold1, "is:", len(col1))
            k = 1
            for name in col1:
                img_path = os.path.join(fold1, str(name))
                img = cv2.imread(img_path)
                # Extracting face
                Viola_j = Viola_jones(img)
                # Extracting features
                out = feature_Extraction(Viola_j)
                # appending the data
                data_all.append(out)
                labels_all.append(i)
                print(f"Completed processing {k}/{len(col1)} images in {fold1}")
                k = k + 1
            i += 1  # incrementing the count to match the label
        if save:
            os.makedirs(f"data_loader/{DB}/", exist_ok=True)
            # saving the data in the npy format
            np.save(f"data_loader/{DB}/features.npy", np.array(data_all))
            # saving the labels in the npy format
            np.save(f"data_loader/{DB}/labels.npy", np.array(labels_all))
        # This line indicates the data saved successfully
        print(f"The data saved successfully for {DB} dataset..................")

    if DB == "MaskedDFER":
        # This line indicates that the preprocessing is started for CKPLUS dataset.
        print(f"Preprocessing Starts for {DB} dataset ...........👍🏽")

        # Initializes folder paths for CKPLUS dataset.
        folder_path1 = r"Dataset/MaskedDFER/train/angry/"
        folder_path2 = r"Dataset/MaskedDFER/train/disgust/"
        folder_path3 = r"Dataset/MaskedDFER/train/fear/"
        folder_path4 = r"Dataset/MaskedDFER/train/happy/"
        folder_path5 = r"Dataset/MaskedDFER/train/neutral/"
        folder_path6 = r"Dataset/MaskedDFER/train/sad/"
        folder_path7 = r"Dataset/MaskedDFER/train/surprise/"

        # Reads the image names in the folders
        image_names1 = [f for f in os.listdir(folder_path1)][:500]
        image_names2 = [f for f in os.listdir(folder_path2)][:335]
        image_names3 = [f for f in os.listdir(folder_path3)][:500]
        image_names4 = [f for f in os.listdir(folder_path4)][:500]
        image_names5 = [f for f in os.listdir(folder_path5)][:500]
        image_names6 = [f for f in os.listdir(folder_path6)][:500]
        image_names7 = [f for f in os.listdir(folder_path7)][:500]

        print("The length of angry is:", len(image_names1))
        print("The length of disgust is:", len(image_names2))
        print("The length of fear is:", len(image_names3))
        print("The length of happy is:", len(image_names4))
        print("The length of neutral is:", len(image_names5))
        print("The length of sad is:", len(image_names6))
        print("The length of surprise is:", len(image_names7))

        # Storing all the images names in a list
        collections = [image_names1, image_names2, image_names3, image_names4, image_names5, image_names6, image_names7]
        # Storing all the folder names in a list
        folders = [folder_path1, folder_path2, folder_path3, folder_path4, folder_path5, folder_path6, folder_path7]
        # empty list to store the data
        data_all = []
        # empty list to store the labels
        labels_all = []
        i = 0
        for col1, fold1 in zip(collections, folders):
            print("Currently Processing:", fold1)
            print("The total no of images in ", fold1, "is:", len(col1))
            k = 1
            for name in col1:
                img_path = os.path.join(fold1, str(name))
                img = cv2.imread(img_path)
                # Extracting face
                Viola_j = Viola_jones(img)
                # Extracting features
                out = feature_Extraction(Viola_j)
                # appending the data
                data_all.append(out)
                labels_all.append(i)
                print(f"Completed processing {k}/{len(col1)} images in {fold1}")
                k = k + 1
            i += 1  # incrementing the count to match the label

        if save:
            os.makedirs(f"data_loader/{DB}/", exist_ok=True)
            # saving the data in the npy format
            np.save(f"data_loader/{DB}/features.npy", np.array(data_all))
            # saving the labels in the npy format
            np.save(f"data_loader/{DB}/labels.npy", np.array(labels_all))
        # This line indicates the data saved successfully
        print(f"The data saved successfully for {DB} dataset..................")