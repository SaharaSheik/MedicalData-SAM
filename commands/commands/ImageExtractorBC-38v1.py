import os
import shutil

def process_folders(root_folder, photos_folder):

    count = 1

    # Get a sorted list of subdirectories in the root folder
    subdirectories = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])

    # Iterate through sorted subdirectories
    for subfolder in subdirectories:
        subfolder_path = os.path.join(root_folder, subfolder)

        # Check if it's a directory and contains an "images" subdirectory
        if os.path.isdir(subfolder_path) and 'images' in os.listdir(subfolder_path):
            images_folder_path = os.path.join(subfolder_path, 'images')

            # Check if the "images" subdirectory contains PNG, JPEG, or TIF files
            if os.path.exists(images_folder_path):
                image_files = [file for file in os.listdir(images_folder_path) if file.lower().endswith('.png') or file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.tif')]

                for file in image_files:
                    # Copy each image file to the photos folder with the same name
                    source_path = os.path.join(images_folder_path, file)
                    destination_path = os.path.join(photos_folder, file)

                    try:
                        shutil.copy2(source_path, destination_path)
                        print(f"Copied: {file} to {destination_path}")
                        count += 1
                    except Exception as e:
                        print(f"Error copying {file}: {e}")

                    # Break the loop when 100 images are copied
                    if count > 100:
                        break



if __name__ == "__main__":
    # Specify the root folder to start the process
    root_directory = "dataset/BBBC038v1/BBBC038v1-First_100"
    destination_directory = 'dataset/BBBC038v1/images'

    # Call the function to process folders
    process_folders(root_directory, destination_directory)


