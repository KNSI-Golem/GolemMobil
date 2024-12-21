import numpy as np
import threading
import ipywidgets
import torch
from torch2trt import TRTModule

from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg

# Add src/ directory for imports
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir1 = current_dir[:current_dir.rfind(os.path.sep)]
parent_dir2 = parent_dir1[:parent_dir1.rfind(os.path.sep)]
sys.path.insert(0, parent_dir2)

from config import model_paths
from dataset.utils import preprocess
from jetracer.nvidia_racecar import NvidiaRacecar

# Tips:
# * If the car wobbles left and right,  lower the steering gain
# * If the car misses turns,  raise the steering gain
# * If the car tends right, make the steering bias more negative (in small increments like -0.05)
# * If the car tends left, make the steering bias more postive (in small increments +0.05)

if __name__ == "__main__":
    # Load the model
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_paths["ride_track"]))

    # Create the racecar&camera class
    car = NvidiaRacecar()
    camera = CSICamera(width=224, height=224)

    car.steering_gain = 1.0
    car.steering_offset = 0.00

    car.throttle = 0.3
    car.throttle_gain = 1.

    while True:
        image = camera.read()
        image = preprocess(image).half()
        output = model_trt(image).detach().cpu().numpy().flatten()
        car.steering = float(output[0])