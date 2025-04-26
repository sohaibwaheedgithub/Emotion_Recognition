import os
import glob
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter


def augment_images(imgs_dir):
    img_files = glob.glob(imgs_dir)
    for img_file in img_files:
        img = Image.open(img_file)
        root, img_name = os.path.split(img_file)
        

        angle_1 = np.random.permutation(range(20, 40))[0]
        angle_2 = -(np.random.permutation(range(20, 40))[0])
        sign = np.random.permutation([1, -1])[0]
        x_shift = sign * np.random.permutation(range(5, 10))[0]
        y_shift = sign * np.random.permutation(range(5, 10))[0]
        
        """
        rand = np.random.permutation([0, 1])[0]
        if rand == 0:
            new_img_name = img_name[:-4] + f'_rotated_{angle_1}.jpg'
            rot_img_1 = img.rotate(angle_1, resample=Image.BICUBIC)
            rot_img_1.save(os.path.join(root, new_img_name))
        else:
            new_img_name = img_name[:-4] + f'_rotated_{angle_2}.jpg'
            rot_img_2 = img.rotate(angle_2, resample=Image.BICUBIC)
            rot_img_2.save(os.path.join(root, new_img_name))
        """

        new_img_name = img_name[:-4] + f'_rotated_{angle_1}.jpg'
        rot_img_1 = img.rotate(angle_1, resample=Image.BICUBIC)
        rot_img_1.save(os.path.join(root, new_img_name))
    
        new_img_name = img_name[:-4] + f'_rotated_{angle_2}.jpg'
        rot_img_2 = img.rotate(angle_2, resample=Image.BICUBIC)
        rot_img_2.save(os.path.join(root, new_img_name))

        new_img_name = img_name[:-4] + f'_shifted_({x_shift},{y_shift}).jpg'
        shift_img = ImageChops.offset(img, x_shift, y_shift)
        shift_img.save(os.path.join(root, new_img_name))
        
        new_img_name = img_name[:-4] + '_flipped.jpg'
        flip_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flip_img.save(os.path.join(root, new_img_name))




        
        new_img_name = img_name[:-4] + '_bright1.jpg'
        enhancer = ImageEnhance.Brightness(img)
        factor = 1.3
        bright_img = enhancer.enhance(factor)
        bright_img.save(os.path.join(root, new_img_name))


        new_img_name = img_name[:-4] + '_bright2.jpg'
        enhancer = ImageEnhance.Brightness(img)
        factor = 1.5
        bright_img = enhancer.enhance(factor)
        bright_img.save(os.path.join(root, new_img_name))


        
        
        new_img_name = img_name[:-4] + '_dark1.jpg'
        enhancer = ImageEnhance.Brightness(img)
        factor = 0.5
        bright_img = enhancer.enhance(factor)
        bright_img.save(os.path.join(root, new_img_name))


        new_img_name = img_name[:-4] + '_dark2.jpg'
        enhancer = ImageEnhance.Brightness(img)
        factor = 0.3
        bright_img = enhancer.enhance(factor)
        bright_img.save(os.path.join(root, new_img_name))

        new_img_name = img_name[:-4] + '_dark3.jpg'
        enhancer = ImageEnhance.Brightness(img)
        factor = 0.2
        bright_img = enhancer.enhance(factor)
        bright_img.save(os.path.join(root, new_img_name))

        




        
        



if __name__ == "__main__":
    dirs = glob.glob(r'datasets\Emotions\*\images\positive')
    #dirs.pop(0)
    
    for _dir in dirs:
        augment_images(os.path.join(_dir, '*'))
        
        
        
        
        
      
    #augment_images(os.path.join(r'datasets\Emotions\sad\images\positive', '*'))

        



    

