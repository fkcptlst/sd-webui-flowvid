import numpy as np
import cv2


def load_flow_to_numpy(path):
    """
    Load optical flow from .flo file
    :param path: path to the .flo file
    :return: optical flow in numpy array, (h, w, 2)
    """
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * h * w)
    data2D = np.resize(data, (h, w, 2))
    return data2D


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to RGB image
    :param flow: (h, w, 2)
    :return: image in RGB format, (h, w, 3)
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # hue is the direction
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # saturation is the magnitude
    # hsv[..., 1] = (mag - mag.min()) / (mag.max() - mag.min()) * 255
    hsv[..., 2] = 255  # not used
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def rgb_to_flow(rgb: np.ndarray, flow_min: float, flow_max: float) -> np.ndarray:
    """
    Convert a visualization of flow image to flow in numpy array
    :param rgb: a visualization of flow image, (h, w, 3)
    :param flow_min: the minimum value of flow
    :param flow_max: the maximum value of flow
    :return: optical flow in numpy array, (h, w, 2)
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    mag = cv2.normalize(hsv[..., 1], None, abs(flow_min), abs(flow_max), cv2.NORM_MINMAX)
    ang = hsv[..., 0] * np.pi * 2 / 180
    flow = np.zeros((rgb.shape[0], rgb.shape[1], 2), dtype=np.float32)
    flow[..., 0], flow[..., 1] = cv2.polarToCart(mag, ang)
    return flow


def draw_color_wheel():
    """
    Draw a color wheel for optical flow visualization
    :return: a color wheel image, (h, w, 3)
    """
    image = np.zeros((360, 360, 3), dtype=np.uint8)
    # rotate 360 degrees with 5 degree step
    for i in range(0, 360, 5):
        ang = i * np.pi / 180
        hue = ang * 180 / np.pi / 2  # hue is the direction
        sat = 255  # saturation is the magnitude
        val = 255  # not used
        hsv = np.array([hue, sat, val], dtype=np.uint8)

        rgb = cv2.cvtColor(hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)

        # draw a line indicating the direction and the color
        x = int(180 + 180 * np.cos(ang))
        y = int(180 + 180 * np.sin(ang))
        # cv2.line(image, (180, 180), (x, y), rgb[0, 0, :].tolist(), 1)
        cv2.line(image, (180, 180), (x, y), rgb[0, 0, :].tolist(), 2)
    cv2.circle(image, (180, 180), 100, (0, 0, 0), -1)
    return image
