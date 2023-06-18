import os
import multiprocessing
import shutil
import glob

import cv2
import numpy as np
from tqdm import tqdm

# input variables
IMAGE_FORMAT = 'png'
IMAGE_SIZE = 300
STEP_SIZE = 150
SOURCE_DATASET_DIR = "/home/taft/SISR/DIV2K_raw/DIV2K_valid_HR"
TARGET_DATASET_DIR = "/home/taft/SISR/DIV2K_300/valid_HR"


def prepare_dataset():
    if os.path.exists(TARGET_DATASET_DIR):
        shutil.rmtree(TARGET_DATASET_DIR)
    os.makedirs(TARGET_DATASET_DIR)
    
    list_image_path = glob.glob(SOURCE_DATASET_DIR + "/*." + IMAGE_FORMAT)
    print("Total images: {}".format(len(list_image_path)))
    
    progress_bar = tqdm(total=len(list_image_path), unit="image", desc="Prepare split image")
    pool = multiprocessing.Pool(processes=16)
    for image_path in list_image_path:
        pool.apply_async(split_image, args=(image_path,), callback=lambda _: progress_bar.update(1))
        
    pool.close()
    pool.join()

    progress_bar.close()


def split_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_height, image_width = image.shape[:2]
    
    index = 1
    if image_height >= IMAGE_SIZE and image_width >= IMAGE_SIZE:
        for y in range(0, image_height - IMAGE_SIZE + 1, STEP_SIZE):
            for x in range(0, image_width - IMAGE_SIZE + 1, STEP_SIZE):
                # crop image
                crop_image = image[y:y + IMAGE_SIZE, x:x + IMAGE_SIZE, ...]
                crop_image = np.ascontiguousarray(crop_image)
                # save image
                crop_image_path = os.path.join(TARGET_DATASET_DIR, os.path.basename(image_path)[:-4] +
                                               "_" + str(index) + "." + IMAGE_FORMAT)
                cv2.imwrite(crop_image_path, crop_image)
                index += 1


if __name__ == "__main__":
    prepare_dataset()