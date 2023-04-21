from itertools import product
import cv2
import numpy as np


def get_moving_mask(flow: np.ndarray, binarize: bool = False, threshold: float = 3,
                    use_fwd_and_bkwd: bool = True) -> np.ndarray:
    """
    Get moving mask from optical flow
    Args:
        binarize:
        threshold:
        flow: optical flow, (h, w, 2)
        use_fwd_and_bkwd: use forward and backward flow to get moving mask

    Returns: a mask indicating moving pixels, (h, w)

    """
    forward_mag = np.linalg.norm(flow, axis=2)
    backward_mag = np.zeros_like(forward_mag)

    if use_fwd_and_bkwd:
        h, w, _ = flow.shape

        xx = np.arange(w)
        yy = np.arange(h)

        xx = np.tile(xx, (h, 1))
        yy = np.tile(yy, (w, 1)).T

        new_xx = np.clip(xx + flow[..., 0], 0, w - 1).astype(np.int)
        new_yy = np.clip(yy + flow[..., 1], 0, h - 1).astype(np.int)

        for y, x in product(range(h), range(w)):
            backward_mag[new_yy[y, x], new_xx[y, x]] = forward_mag[y, x]

    if binarize:
        moving_mask = np.maximum(forward_mag, backward_mag)
        moving_mask[moving_mask < threshold] = 0
        moving_mask[moving_mask > 0] = 1
        # use morphological closing to fill holes
        moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    else:
        moving_mask = np.maximum(forward_mag, backward_mag)
        moving_mask[moving_mask < threshold] = 0

    return moving_mask
