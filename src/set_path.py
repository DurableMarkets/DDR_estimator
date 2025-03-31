from pathlib import Path


def get_paths():
    # Get current file path
    src_folder = str(Path(__file__).parent.parent.resolve()) + "/"

    path_dict = {
        "sim_data": src_folder + "sim_data/"
    }
    return path_dict