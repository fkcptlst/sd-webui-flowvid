import json

import cv2
import numpy as np
from matplotlib import pyplot as plt

from flow.warp import warp
from flow.common import load_flow_to_numpy, rgb_to_flow, flow_to_rgb

if __name__ == '__main__':
    first_frame_gt = cv2.imread('0.png')  # HWC, BGR
    second_frame_gt = cv2.imread('1.png')  # HWC, BGR

    first_frame_gt = cv2.cvtColor(first_frame_gt, cv2.COLOR_BGR2RGB)
    second_frame_gt = cv2.cvtColor(second_frame_gt, cv2.COLOR_BGR2RGB)

    with open('_metadata.json', 'r') as f:
        metadata = json.load(f)
        flow_max, flow_min = float(metadata['max']), float(metadata['min'])

    flow_rgb = cv2.cvtColor(cv2.imread('flow.jpg'), cv2.COLOR_BGR2RGB)

    flow = rgb_to_flow(flow_rgb, flow_min, flow_max)
    # flow = load_flow_to_numpy('flow.flo')

    # frame_warp = warp(first_frame_gt, flow, warp_mode='forward')
    frame_warp = warp(second_frame_gt, flow, warp_mode='backward')

    plt.imshow(first_frame_gt)
    plt.title('first_frame_gt')
    plt.show()
    plt.imshow(second_frame_gt)
    plt.title('second_frame_gt')
    plt.show()
    plt.imshow(frame_warp)
    plt.title('frame_warp')
    plt.show()
