import os
import shutil

def copy_first_n_directories(source_folder, destination_folder, n=100):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all directories in the source folder
    all_directories = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]

    # Take the first n directories
    selected_directories = all_directories[:n]

    for directory in selected_directories:
        source_path = os.path.join(source_folder, directory)
        destination_path = os.path.join(destination_folder, directory)

        try:
            # Copy the entire directory to the destination folder
            shutil.copytree(source_path, destination_path)
            print(f"Directory '{directory}' copied to '{destination_folder}' successfully.")
        except Exception as e:
            print(f"Error copying directory '{directory}': {e}")

if __name__ == "__main__":
    # Specify the source and destination folders


    ## Take 100 from BBBC038v1
    source_directory = "dataset/BBBC038v1/stage1_train"
    destination_directory = "dataset/BBBC038v1/BBBC038v1-First_100"

    # Specify the number of directories to copy (default is 100)
    number_of_directories_to_copy = 100

    # Call the function to copy directories
    copy_first_n_directories(source_directory, destination_directory, number_of_directories_to_copy)



