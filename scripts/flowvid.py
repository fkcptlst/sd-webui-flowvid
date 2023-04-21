import os
from typing import List, Tuple
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from flow import warp, get_moving_mask, rgb_to_flow
from PIL import Image
from tqdm import tqdm
import cv2

import modules.scripts as scripts
from modules.processing import process_images
from modules.processing import Processed, StableDiffusionProcessingImg2Img

# global variables
vfps = None
vframesize = None


def video2list(video_path: str, crop=False) -> []:
    global vfps, vframesize
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    vfps = cap.get(cv2.CAP_PROP_FPS)
    vframesize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    framenum = int(cap.get(7))
    video = []
    for _ in tqdm(range(framenum), desc="Video2ImageList"):
        a, b = cap.read()
        if not a:
            break
        b = np.array(b)
        if crop:
            b = crop_img(b)
        video.append(b)
    cap.release()
    return video


def list2video(imgs: List[np.ndarray]) -> str:
    fps = vfps if vfps is not None else 24
    framesize = vframesize if vframesize is not None else (256, 256)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    path = f"{datetime.now().timestamp()}.mp4"
    video = cv2.VideoWriter(path, fourcc=fourcc, fps=float(fps), frameSize=framesize)
    for i in tqdm(imgs, desc="List to Videos"):
        video.write(i.astype('uint8'))
    video.release()
    return path


def unwrap_param(*args, **kwargs):
    guide_frame = args[0]
    flow_list = video2list(args[1], crop=True)
    print(type(flow_list[0]))
    mask_type = str(args[2])
    use_forward_warp = bool(args[3])
    use_moving_mask = bool(args[4])
    moving_mask_threshold = float(args[5])
    adaptive_inpainting = bool(args[6])
    return guide_frame, flow_list, mask_type, use_forward_warp, \
           use_moving_mask, moving_mask_threshold, adaptive_inpainting


def crop_img(image: np.ndarray, shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Crop image to the given shape.
    Args:
        image: image to crop, (H, W, C)
        shape: target shape, (H, W)

    Returns:

    """
    h, w, _ = image.shape
    # resize the shorter side to the target size proportionally
    if h < w:
        new_h, new_w = shape[0], int(w * shape[0] / h)
    else:
        new_h, new_w = int(h * shape[1] / w), shape[1]
    image = cv2.resize(image, (new_w, new_h))
    h, w, _ = image.shape
    h_start = (h - shape[0]) // 2
    w_start = (w - shape[1]) // 2
    image = image[h_start:h_start + shape[0], w_start:w_start + shape[1], :]
    return image


class Script(scripts.Script):

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "Flow guided video generation"

    # Determines when the script should be shown in the dropdown menu via the
    # returned value. As an example:
    # is_img2img is True if the current tab is img2img, and False if it is txt2img.
    # Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img: bool):
        return is_img2img

    # How the script's is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.

    def ui(self, is_img2img: bool) -> dict:
        gr.Markdown("# 光流制导视频生成插件")
        with gr.Accordion("说明", open=False):
            content = """
                第一帧上传前导帧，视频上传光流视频
            """
            gr.Markdown(content)

        with gr.Box():
            gr.Markdown("Input")
            with gr.Column():
                with gr.Row():
                    first_frame = gr.Image(label="第一帧")
                    video_flow = gr.PlayableVideo(format='mp4', source='upload', label="光流")
                mask_type = gr.Dropdown(choices=["image_mask", "latent_mask"], label="Mask Type", value="image_mask")
                use_forward_warp = gr.Checkbox(value=True, label="use_forward_warp")
                use_moving_mask = gr.Checkbox(value=True, label="use_moving_mask")
                adaptive_inpainting = gr.Checkbox(value=True, label="adaptive_inpainting")
                moving_mask_threshold = gr.Slider(1, 10, value=6, label="moving_mask_threshold")
        return [
            first_frame,  # 0
            video_flow,  # 1
            mask_type,  # 2
            use_forward_warp,  # 3
            use_moving_mask,  # 4
            moving_mask_threshold,  # 5
            adaptive_inpainting,  # 6
        ]

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.

    def run(self, p: StableDiffusionProcessingImg2Img, *args, **kwargs) -> Processed:

        guide_frame, flow_list, mask_type, use_forward_warp, \
        use_moving_mask, moving_mask_threshold, adaptive_inpainting = unwrap_param(*args, **kwargs)

        # guide_frame: np.ndarray = None
        # flow_list: List[np.ndarray] = []
        # mask_type: str = 'image_mask'  # 'image_mask' or 'latent_mask'
        # use_forward_warp: bool = True
        # use_moving_mask: bool = True
        # moving_mask_threshold: float = 3.0  # threshold for segmenting moving pixels
        # adaptive_inpainting: bool = False  # blur the inpainted region based on magnitude of flow

        assert flow_list, "No flow provided"

        result_images: List[np.ndarray] = []

        # TODO flow max and min numpy file
        print("loading flow_min_and_max")
        flow_min_and_max = np.load(
            r'D:\Project_repository\stable-diffusion-webui\extensions\sd-webui-flowvid\scripts\flow_min_and_max.npy')  # shape = (N, 2)

        # prev_frame = guide_frame  # the first frame is the guide frame
        prev_frame = np.array(p.init_images[0]) if guide_frame is None else np.array(guide_frame)
        prev_frame = crop_img(prev_frame)
        for i, flow_rgb in enumerate(tqdm(flow_list, desc="generating frames")):
            # p_now = deepcopy(p)  # make a copy of the processing object  # TODO check
            p_now = p

            # TODO
            flow = rgb_to_flow(flow_rgb, flow_min_and_max[0, 0], flow_min_and_max[0, 1])

            second_frame_warped = warp(
                prev_frame,
                flow,
                warp_mode='forward' if use_forward_warp else 'backward',
            )

            # apply moving mask as inpainting mask
            if use_moving_mask:
                moving_mask = get_moving_mask(
                    flow,
                    binarize=True,
                    threshold=6,  # TODO moving_mask_threshold,
                )

                if adaptive_inpainting:
                    raise NotImplementedError("Adaptive inpainting is not implemented yet")  # TODO

                # leave background intact
                # second_frame_warped_masked = second_frame_warped * np.expand_dims(moving_mask, 2)  # expand to 3 channels and broadcast
                # background = (1 - np.expand_dims(moving_mask, 2)) * prev_frame
                # second_frame_warped = second_frame_warped_masked + background
                second_frame_warped_masked = second_frame_warped
                second_frame_warped_masked[moving_mask == 0] = 0
                background = prev_frame
                background[moving_mask > 0] = 0
                second_frame_warped = background + second_frame_warped_masked

                mask = moving_mask * 255  # convert to 0-255 range
                mask = mask.astype(np.uint8)
                mask = Image.fromarray(mask)  # convert to PIL image

                # apply the moving mask to the second frame
                if mask_type == 'image_mask':
                    p_now.image_mask = mask
                elif mask_type == 'latent_mask':
                    p_now.latent_mask = mask
                else:
                    raise ValueError(f"Unknown mask type: {mask_type}")

                # save mask
                if os.path.exists('mask.png'):
                    os.remove('mask.png')
                mask.save('mask.png')

            # generate the frame
            second_frame_warped_pil = Image.fromarray(np.uint8(second_frame_warped))
            p_now.init_images = [second_frame_warped_pil]
            # p.do_not_save_grid = True  # TODO check
            # p.control_net_input_image = Image.open(reference_imgs[i].name).convert("RGB").resize(
            #     (initial_width, p.height), Image.ANTIALIAS)  # TODO check

            proc = process_images(p_now)  # run ddpm
            image = proc.images[0]  # PIL image
            image_np = np.array(image)

            result_images.append(image_np)
            print(f"len(result_images): {len(result_images)}")

            # TODO
            warped_img_path = Path(f"result_images/warp_{len(result_images)}.png")
            if not warped_img_path.parent.exists():
                warped_img_path.parent.mkdir(parents=True)
            second_frame_warped_pil.save(warped_img_path)

            # save image, if parent directory does not exist, create it
            img_save_path = Path(f"result_images/gen_{len(result_images)}.png")
            if not img_save_path.parent.exists():
                img_save_path.parent.mkdir(parents=True)
            image.save(img_save_path)

            prev_frame = image_np  # the current frame becomes the previous frame for the next iteration

        result_images_pil = [Image.fromarray(img) for img in result_images]

        # TODO convert to mp4 and save to disk
        mp4_save_path = list2video(result_images)
        print(f"Video saved to {mp4_save_path}.")

        return Processed(p, images_list=result_images_pil)
