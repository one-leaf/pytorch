# Remove data from the data folder
import os
from model import PolicyValueNet, data_dir, data_wait_dir, model_file

def remove_data():
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if name.endswith(".pkl"):
                print(f"Removing file: {os.path.join(root, name)}")
                os.remove(os.path.join(root, name))
                
if __name__ == "__main__":
    remove_data()
    print("All .pkl files removed from the data folder.")
    