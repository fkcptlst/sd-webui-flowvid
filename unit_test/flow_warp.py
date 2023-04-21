import json

import cv2
import numpy as np
from matplotlib import pyplot as plt

from flow.warp import warp
from flow.common import load_flow_to_numpy, rgb_to_flow, flow_to_rgb
from flow_mask import get_moving_mask

if __name__ == '__main__':
    # first_frame_gt = cv2.imread('0.png')  # HWC, BGR
    # second_frame_gt = cv2.imread('1.png')  # HWC, BGR

    first_frame_gt = cv2.imread('first_frame.png')  # HWC, BGR
    second_frame_gt = cv2.imread('second_frame.png')  # HWC, BGR

    first_frame_gt = cv2.cvtColor(first_frame_gt, cv2.COLOR_BGR2RGB)
    second_frame_gt = cv2.cvtColor(second_frame_gt, cv2.COLOR_BGR2RGB)

    # with open('_metadata.json', 'r') as f:
    #     metadata = json.load(f)
    #     flow_max, flow_min = float(metadata['max']), float(metadata['min'])

    flow_min_and_max = np.load('flow_min_and_max.npy')  # shape = (N, 2)
    flow_min, flow_max = flow_min_and_max[0, 0], flow_min_and_max[0, 1]

    flow_rgb = cv2.cvtColor(cv2.imread('flow_0.png'), cv2.COLOR_BGR2RGB)

    flow = rgb_to_flow(flow_rgb, flow_min, flow_max)
    # flow = load_flow_to_numpy('flow.flo')

    frame_warp = warp(first_frame_gt, flow, warp_mode='forward')
    # frame_warp = warp(second_frame_gt, flow, warp_mode='backward')


    plt.imshow(first_frame_gt)
    plt.title('first_frame_gt')
    plt.show()
    plt.imshow(second_frame_gt)
    plt.title('second_frame_gt')
    plt.show()
    plt.imshow(frame_warp)
    plt.title('frame_warp')
    plt.show()

    moving_mask = get_moving_mask(flow, binarize=True, use_fwd_and_bkwd=True, threshold=6)
    frame_warp_masked = frame_warp.copy()
    frame_warp_masked[moving_mask == 0] = first_frame_gt[moving_mask == 0]

    plt.imshow(frame_warp_masked)
    plt.title('frame_warp_masked')
    plt.show()
