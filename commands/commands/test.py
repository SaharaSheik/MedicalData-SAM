from PIL import Image
import numpy as np

mask_path = 'output/BBBC038v1/test/0c6507d493bf79b2ba248c5cca3d14df8b67328b89efa5f4a32f97a06a88c92c/0.png'
mask_image = Image.open(mask_path)

# Convert the image to a binary NumPy array
mask_array = np.array(mask_image)
binary_mask = (mask_array != 0).astype(int)
contains_zero = np.any(binary_mask == 0)
print(contains_zero)
print("rbinary_mask ")
print(binary_mask)


def rle_encode(binary_mask):
    rle_list = []
    current_run = 0
    current_pixel = 0

    for pixel in binary_mask.flatten():
        if pixel == current_pixel:
            current_run += 1
        else:
            rle_list.extend([current_pixel, current_run])
            current_pixel = pixel
            current_run = 1

    # Add the last run
    rle_list.extend([current_pixel, current_run])

    return rle_list

rele = rle_encode(binary_mask)
print(rele)