import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from .perlin_numpy.perlin_numpy import generate_perlin_noise_2d
from scipy import ndimage


def visualise_bounding_box(image: np.ndarray, x: int, y: int, w: int, h: int):
    """
    Superposes the bounding box with the original x-ray.

    :param image: x-ray image
    :param x: x coordinate of the box
    :param y: y coordinate of the box
    :param w: width of the box
    :param h: height of the box
    """

    image = cv2.rectangle(image.copy(), (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Make some preprocessing steps to the image

    :param image: x-ray image
    :return: Preprocessed image
    """

    # From 16 bit to 8 bit
    image = (255 * (image / 4095))
    # From one channel to 3 channels
    height, width = image.shape
    image = image.reshape((height, width, 1))
    image = np.concatenate([image] * 3, axis=2).astype(np.uint8())

    return image


def inverse_preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Applies preprocessing steps to make it compatible with Cycle-Gan.

    :param image: x-ray image
    :return: Preprocessed image
    """

    # From 8 bit to 16 bit
    image = (4095 * (image / 255))
    # From 3 channels to one channel
    image = image[:, :, 0]

    return image


def load_random_node(dir_path: str, width: int, height: int) -> np.ndarray:
    """
    Loads a random mask from NODES_EXAMPLES_PATH folder, applies a gaussian blur to smooth the edges and makes
    random rotation.

    :param dir_path: path to the directory containing the nodes
    :param width: width of the bounding box
    :param height: height of the bounding box
    :return: The mask of the node
    """

    node = cv2.imread(os.path.join(dir_path, "%s.png" % random.randint(0, 12))).astype(float)
    node = cv2.resize(node, (200, 200))
    node = cv2.GaussianBlur(node, (45, 45), 0)
    node = ndimage.rotate(node, random.randint(0, 365))
    node = cv2.resize(node, (width, height))

    return node


def generate_noise(width: int, height: int) -> np.ndarray:
    """
    Generates perlin noise and resizes it

    :param width: width of the resized window
    :param height: height of the resized window
    :return: noise array as image
    """

    noise = generate_perlin_noise_2d((256, 256), (8, 8))
    noise = np.array(noise)
    noise -= noise.min()
    noise += 1
    noise /= noise.max()
    noise *= 255
    noise = noise.reshape((256, 256, 1))
    noise = np.concatenate([noise] * 3, axis=2)
    noise = cv2.resize(noise, (width, height)).astype(float)

    return noise


def generate_node(dir_path: str, window_width: int, window_height: int,
                  contrast_intensity: float, visualize: bool) -> np.ndarray:
    """
    Creates the node

    :param dir_path: path to the node templates directory
    :param visualize: visualize node if True
    :param window_width: width of the resized window
    :param window_height: height of the resized window
    :param contrast_intensity: contrast intensity of the node vs background
    :return: node as image
    """

    node = load_random_node(dir_path, window_width, window_height)
    noise = generate_noise(window_width, window_height)

    node = (noise * node).astype(float)
    node /= node.max()
    node *= contrast_intensity * 255
    node[node > 255] = 255

    if visualize:
        plt.imshow(node.astype(np.uint8()))
        plt.show()

    return node


def merge_node_and_image(image: np.ndarray, node: np.ndarray, x: int, y: int, width: int, height: int):
    """
    Merges node and x-ray image

    :param image: x-ray image
    :param node: simulated node
    :param x: x coordinate of the box
    :param y: y coordinate of the box
    :param width: width of the box
    :param height: height of the box
    :return:
    """

    cropped_image = image.astype(float)[y: y + height, x: x + width]
    difference_from_255 = 255 * np.ones(cropped_image.shape) - cropped_image
    difference_from_255 /= 255

    node = node * difference_from_255 + cropped_image
    node[node > 255] = 255
    node = node.astype(np.uint8())

    image[y: y + height, x: x + width] = node.astype(np.uint8())

    return image

