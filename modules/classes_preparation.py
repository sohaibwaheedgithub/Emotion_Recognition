import os
import glob
from PIL import Image
import re



EMR_D = r'datasets\Emotions'

emotions = os.listdir(r'datasets\Emotions')

class_name = 'happy'
emotions.remove(class_name)


for emotion in emotions:
    print("Saving images from {}".format(emotion))
    """img_files = glob.glob(os.path.join(EMR_D, 'train', emotion, '*'))
    img_files += glob.glob(os.path.join(EMR_D, 'test', emotion, '*'))"""
    #img_files = img_files[:800]
    #img_files = glob.glob(os.path.join(EMR_D, emotion, '*'))

    img_files = glob.glob(os.path.join(EMR_D, emotion, 'images', 'positive', "*"))

    
    for img_f in img_files:
        _, img_n = os.path.split(img_f)
            
        img_n = emotion + "_" + img_n
        img = Image.open(img_f)
        img.save(os.path.join(r'datasets\Emotions', class_name, 'images', 'negative', img_n))
        
        
        """
        for emotion_2 in emotions:
            if emotion == emotion_2:
                img.save(os.path.join(r'datasets\FER2013_2', emotion_2, 'images', 'positive', img_n))
            else:
                img.save(os.path.join(r'datasets\FER2013_2', emotion_2, 'images', 'negative', img_n))
        """




          