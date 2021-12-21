import SimpleITK
import matplotlib.pyplot as plt
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from utils import *
from typing import Dict
import json
from skimage.measure import regionprops
import imageio
from pathlib import Path
import time
import pandas as pd
import random
from random import randrange
import os
from submission.src.create_simulated_data import generate_node, visualise_bounding_box, merge_node_and_image, \
    preprocess_image, inverse_preprocess_image
NODES_EXAMPLES_PATH = "./submission/nodes_examples"

# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch
# between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = True


class Nodulegeneration(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_path=Path("/output/") if execute_in_docker else Path("./output/"),
            output_file=Path("/output/results.json") if execute_in_docker else Path("./output/results.json")

        )

        # load nodules.json for location
        with open("/input/nodules.json" if execute_in_docker else "test/nodules.json") as f:
            self.data = json.load(f)

        # If True, will use matplotlib to visualise the node and the results
        self.visualize = False

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        input_image = SimpleITK.GetArrayFromImage(input_image)
        total_time = time.time()
        if len(input_image.shape) == 2:
            input_image = np.expand_dims(input_image, 0)

        nodule_images = np.zeros(input_image.shape)
        for j in range(len(input_image)):
            cxr_img_scaled = input_image[j, :, :]
            nodule_data = [i for i in self.data['boxes'] if i['corners'][0][2] == j]

            # Image preprocessing step
            cxr_img_scaled = preprocess_image(cxr_img_scaled)
            for nodule in nodule_data:
                try:
                    # Extract coordinates
                    boxes = nodule['corners']
                    y_min, x_min, y_max, x_max = boxes[2][0], boxes[2][1], boxes[0][0], boxes[0][1]
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    width = x_max - x_min
                    height = y_max - y_min

                    # Generates the nodule
                    node = generate_node(NODES_EXAMPLES_PATH, width, height, contrast_intensity=0.4,
                                         visualize=self.visualize)
                    if self.visualize:
                        visualise_bounding_box(cxr_img_scaled, x_min, y_min, width, height)

                    # Simulates the nodule at the specified coordinates
                    cxr_img_scaled = merge_node_and_image(cxr_img_scaled, node, x_min, y_min, width, height)
                    if self.visualize:
                        visualise_bounding_box(cxr_img_scaled, x_min, y_min, width, height)

                except:
                    continue

            cxr_img_scaled = inverse_preprocess_image(cxr_img_scaled)
            nodule_images[j, :, :] = cxr_img_scaled
        print('total time took ', time.time() - total_time)
        return SimpleITK.GetImageFromArray(nodule_images)


if __name__ == "__main__":
    Nodulegeneration().process()
