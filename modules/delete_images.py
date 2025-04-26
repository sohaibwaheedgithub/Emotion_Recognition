import os
import glob
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import re


def augment_images(imgs_dir):
    img_files = glob.glob(imgs_dir)
    for img_file in img_files:
        #img = Image.open(img_file)
        root, img_name = os.path.split(img_file)
        

        
        """if not (img_name[0] == 'P' or img_name[0] == 'T'):
            os.remove(img_file)"""
        """if not re.search('rotated', img_name) == None:
            os.remove(img_file)
        elif not re.search('dark', img_name) == None:
            os.remove(img_file)
        elif not re.search('shifted', img_name) == None:
            os.remove(img_file)
        elif not re.search('flipped', img_name) == None:
            os.remove(img_file)
        elif not re.search('bright', img_name) == None:
            os.remove(img_file)
        elif not re.search('low_dark', img_name) == None:
            os.remove(img_file)
        elif not re.search('lowest_dark', img_name) == None:
            os.remove(img_file)
        elif not re.search('more_dark', img_name) == None:
            os.remove(img_file)
        elif not re.search('high_bright', img_name) == None:
            os.remove(img_file)
        """
        """if not re.search('filtered', img_name) == None:
            os.remove(img_file)"""
        if not re.search('dark', img_name) == None:
            os.remove(img_file)
        elif not re.search('bright', img_name) == None:
            os.remove(img_file)
        elif not re.search('flipped', img_name) == None:
            os.remove(img_file)
        elif not re.search('rotated', img_name) == None:
            os.remove(img_file)
        elif not re.search('shifted', img_name) == None:
            os.remove(img_file)
        
        

        


        
        
        

        
        



if __name__ == "__main__":
    dirs = glob.glob(r'datasets\Emotions\*\images\positive')

    for _dir in dirs:
        augment_images(os.path.join(_dir, '*'))
        
        
        
        
    #augment_images(os.path.join(r'datasets\Emotions\sad\images\positive', '*'))
    #augment_images(os.path.join(r'datasets\Emotions\surprise', '*'))
    """import shutil
    for i in ['neutral', 'surprise']:
        shutil.rmtree(os.path.join(r'datasets\Emotions\{}\images\negative'.format(i)))"""



    

