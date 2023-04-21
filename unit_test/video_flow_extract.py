from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from mmflow.apis import init_model, inference_model
from typing import List

model = init_model(config='./checkpoints/raft_8x2_100k_mixed_368x768.py',
                   checkpoint='./checkpoints/raft_8x2_100k_mixed_368x768.pth',
                   device='cuda:0' if torch.cuda.is_available() else 'cpu')


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # hue is the direction
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # saturation is the magnitude
    # hsv[..., 1] = (mag - mag.min()) / (mag.max() - mag.min()) * 255
    hsv[..., 2] = 255  # not used
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def parse_args():
    parser = ArgumentParser(description='MMFlow video flow extraction')
    parser.add_argument('video', help='video file')
    parser.add_argument('output', help='output file')
    parser.add_argument(
        '--fourcc',
        default='mp4v',
        help='fourcc of the output video')
    return parser.parse_args()


def video_frames_generator(video_path):
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        yield frame
    video.release()


def calc_flow(img1, img2):
    # flow = inference_model(model, img1, img2)
    flow = inference_model(model, img2, img1)
    flow_rgb = flow_to_rgb(flow)
    return flow_rgb, flow


def get_flow_vid(video_path):
    results = []
    flow_max = []
    flow_min = []
    prev_frame = None
    for i, frame in enumerate(video_frames_generator(video_path)):
        if i == 0:
            prev_frame = frame
            continue
        flow_rgb, flow = calc_flow(prev_frame, frame)
        flow_max.append(flow.max())
        flow_min.append(flow.min())
        prev_frame = frame
        results.append(flow_rgb)

    flow_min_and_max = np.stack([flow_min, flow_max], axis=1)
    return results, flow_min_and_max


def list2video(imgs: List[np.ndarray], output_path: str):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    size = imgs[0].shape[:2][::-1]
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    for img in imgs:
        video_writer.write(img)
    video_writer.release()


if __name__ == '__main__':
    args = parse_args()
    results, flow_min_and_max = get_flow_vid(args.video)
    list2video(results, args.output)
    np.save('flow_min_and_max.npy', flow_min_and_max)
