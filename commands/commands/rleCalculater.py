from PIL import Image
import os
import numpy as np
from pycocotools import mask as mask_utils
import itertools

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

def load_mask(mask_path):
    # Load the PNG image using Pillow
    mask_image = Image.open(mask_path)


    # Convert the image to a binary NumPy array
    mask_array = np.array(mask_image)
    binary_mask = (mask_array != 0).astype(int)
    return binary_mask

def process_directory_Ground(root_directory):
    result_dict = {}

    for subdirectory in os.listdir(root_directory):
        subdirectory_path = os.path.join(root_directory, subdirectory)

        if os.path.isdir(subdirectory_path):
            subdirectories_dict = {}

            for subsubdirectory in os.listdir(subdirectory_path):
                subsubdirectory_path = os.path.join(subdirectory_path, subsubdirectory)

                if os.path.isdir(subsubdirectory_path) and subsubdirectory.lower() == 'masks':
                    masks_folder = subsubdirectory_path

                    # Assuming the images in the "masks" folder are binary PNG images
                    for mask_file in os.listdir(masks_folder):
                        mask_path = os.path.join(masks_folder, mask_file)

                        mask_image = Image.open(mask_path)
                        print(mask_image)
                        # Load the binary mask image using Pillow
                        mask = load_mask(mask_path)

                        # Calculate RLE encoding
                        rle_result = rle_encode(mask)

                        # Store RLE result in subsubdirectory dictionary
                        subdirectories_dict[mask_file] = rle_result

            result_dict[subdirectory] = subdirectories_dict

    return result_dict


def process_directory_Sam(root_directory):
    result_dict = {}

    for subdirectory in os.listdir(root_directory):
        subdirectory_path = os.path.join(root_directory, subdirectory)

        if os.path.isdir(subdirectory_path):
            rle_dict = {}

            for mask_file in os.listdir(subdirectory_path):
                mask_path = os.path.join(subdirectory_path, mask_file)

                # Check if the file has a recognized image extension
                valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
                if os.path.splitext(mask_file.lower())[1] in valid_extensions:
                    # Load the binary mask image using Pillow
                    mask = load_mask(mask_path)

                    # Calculate RLE encoding
                    rle_result = rle_encode(mask)


                    # Store RLE result in subdirectory dictionary
                    rle_dict[mask_file] = rle_result

            # Store RLE dictionary in the result dictionary with subdirectory name as key
            result_dict[subdirectory] = rle_dict


    return result_dict



def calculate_iou(dict1, dict2, output_file):
    all_ious = []
    max_ious = []
    mean_ious = []

    try:
        f = open(output_file, 'w')

        for key1 in dict1:
            if key1 in dict2:
                ious = []
                for sub_key1, sub_key2 in itertools.product(dict1[key1], dict2[key1]):
                    mask1 = dict1[key1][sub_key1]
                    mask2 = dict2[key1][sub_key2]



                    # # Convert masks to RLE format
                    rle1 = mask_utils.decode(np.asfortranarray(mask1))
                    rle2 = mask_utils.decode(np.asfortranarray(mask2))


                    # Calculate IoU using mask_utils.iou
                    iou = mask_utils.iou([rle1], [rle2], [0])

                    ious.append(iou)

                # Store IoUs in the array
                all_ious.extend(ious)

                # Store the maximum IoU for each key
                max_iou = max(ious, default=0)
                max_ious.append(max_iou)

                # Calculate and write the mean IoU to the file for each key
                mean_iou = np.mean(ious)
                f.write(f"{key1}_mean_iou: {mean_iou}\n")

                # Store the mean IoU for each key
                mean_ious.append(mean_iou)

        # Calculate and write the overall mean IoU to the file
        overall_mean_iou = np.mean(mean_ious)
        f.write(f"Overall Mean IoU: {overall_mean_iou}\n")

    finally:
        f.close()

    return all_ious, max_ious, overall_mean_iou







if __name__ == "__main__":
    # Specify the root directory to start the process

    #For BBBC038v1
    root_directory_ground = "dataset/BBBC038v1/BBBC038v1-First_100"
    root_directory_Sam = "output/BBBC038v1"
#########


    #Call the function to process the directory
    result_ground = process_directory_Ground(root_directory_ground)
    result_Sam = process_directory_Sam(root_directory_Sam)


    output_file = 'output/BBBC038v1/iou_results.txt'

    results = calculate_iou(result_ground, result_Sam, output_file)

    # Print the resulting dictionary
    print(results)

