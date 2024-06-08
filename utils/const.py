import os

script_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(root_dir, "data")
img_dir = os.path.join(data_dir, "images")
faces_dir = os.path.join(data_dir, "faces")
checkpoints_dir = os.path.join(root_dir, "checkpoints")
