import cv2
from matplotlib import pyplot as plt

from flow.mask import get_moving_mask
from flow.common import load_flow_to_numpy, flow_to_rgb

if __name__ == '__main__':
    first_frame_gt = cv2.imread('0.png')  # HWC, BGR
    second_frame_gt = cv2.imread('1.png')  # HWC, BGR

    first_frame_gt = cv2.cvtColor(first_frame_gt, cv2.COLOR_BGR2RGB)
    second_frame_gt = cv2.cvtColor(second_frame_gt, cv2.COLOR_BGR2RGB)

    flow_rgb = cv2.cvtColor(cv2.imread('flow.jpg'), cv2.COLOR_BGR2RGB)

    flow = load_flow_to_numpy('flow.flo')

    flow_rgb = flow_to_rgb(flow)

    # moving_mask = get_moving_mask(flow, binarize_threshold=5)
    moving_mask = get_moving_mask(flow)

    plt.imshow(first_frame_gt)
    plt.title('first_frame_gt')
    plt.show()

    plt.imshow(second_frame_gt)
    plt.title('second_frame_gt')
    plt.show()

    plt.imshow(flow_rgb)
    plt.title('flow_rgb')
    plt.show()

    # apply moving mask
    flow_rgb[moving_mask == 0] = 0
    plt.imshow(flow_rgb)
    plt.title('flow_rgb_masked')
    plt.show()

    # apply mask to first frame
    first_frame_gt[moving_mask == 0] = 0
    plt.imshow(first_frame_gt)
    plt.title('first_frame_gt_masked')
    plt.show()

    # apply mask to second frame
    second_frame_gt[moving_mask == 0] = 0
    plt.imshow(second_frame_gt)
    plt.title('second_frame_gt_masked')
    plt.show()
