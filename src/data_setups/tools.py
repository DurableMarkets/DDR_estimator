import os

def create_folder(folder_name):
    """
    Create a folder if it does not exist.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

