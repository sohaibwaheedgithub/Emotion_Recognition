# Importing Libraries


import os
import cv2
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from contextlib import ExitStack
import matplotlib.pyplot as plt
from constants import Constants
from sklearn.model_selection import train_test_split


class Data_Preparation():

    def __init__(self):
        self.constants = Constants()


    
    def __prepare_arrays__(self):
        print("Data saving started At : ", time.ctime(time.time()), "\n")
        root = r'datasets\Emotions'
        emotions = os.listdir(root)
        #emotions.pop(0)
        emotions = ['happy']
        
        for emotion in emotions:
            print("Saving data from {} dataset".format(emotion), "\n")
            X = []
            y = []
            neg = True
            imgs_dirs = glob.glob(os.path.join(root, emotion, 'images', '*'))
            for idx, img_dir in enumerate(imgs_dirs):
                img_fs = glob.glob(os.path.join(img_dir, '*'))
                for img_f in img_fs:
                    img = plt.imread(img_f)
                    X.append(img)
                    y.append(idx)
                if neg:
                    X_neg = np.array(X)
                    y_neg = np.array(y)
                    X = []
                    y = []
                neg = False
            X_pos = np.array(X)
            y_pos = np.array(y)
    

            """min_len = min(X_pos.shape[0], X_neg.shape[0])

            indices = np.random.permutation(min_len)
            
            X_neg = X_neg[indices]
            y_neg = y_neg[indices]

            X_pos = X_pos[indices]
            y_pos = y_pos[indices]"""

            X = np.concatenate([X_neg, X_pos], axis = 0)
            y = np.concatenate([y_neg, y_pos], axis = 0)

            np.save(os.path.join(root, emotion, 'X_biased_D123B1SR12F.npy'), X)
            np.save(os.path.join(root, emotion, 'y_biased_D123B1SR12F.npy'), y)
            break
            
        
        print("Data saving ended At : ", time.ctime(time.time()), "\n")



    def __prepare_multiclass_arrays__(self):
        root = r'datasets\Emotions'
        emotions = os.listdir(root)
        X = []
        y = []
        for idx, emotion in enumerate(emotions):
            print("Saving data from {} dataset".format(emotion), "\n")
            

            img_fs = glob.glob(os.path.join(root, emotion, 'images', 'positive', '*'))
            img_fs = np.array(img_fs)
            indices = np.random.permutation(img_fs.shape[0])[:25734]
            img_fs = img_fs[indices]
            
            for img_f in img_fs:
                img = plt.imread(img_f)
                X.append(img)
                y.append(idx)
            
        
        X = np.array(X)
        y = np.array(y)
    

        np.save(r'datasets\X_multiclass_unbiased_NDF_D123B12SR12F.npy', X)
        np.save(r'datasets\y_multiclass_unbiased_NDF_D123B12SR12F.npy', y)
        
            
        
        #print("Data saving ended At : ", time.ctime(time.time()), "\n")"""

    


    
    def split_data(self, X_path, y_path):
        X, y = np.load(X_path), np.load(y_path)
        print(X.shape, y.shape)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size = 0.2,
            shuffle = True,
            random_state = 42,
        )
        return X_train, y_train, X_valid, y_valid





if __name__ == "__main__":
    data_preparation = Data_Preparation()
    #data_preparation.__prepare_multiclass_arrays__()
    data_preparation.__prepare_arrays__()
    
    