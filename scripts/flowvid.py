import gradio as gr
import os

import modules.scripts as scripts
import numpy as np
from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed, StableDiffusionProcessingImg2Img
from modules.shared import opts, cmd_opts, state

from flow import rgb_to_flow, warp, get_moving_mask
from typing import List, Tuple, Union, Optional
from copy import deepcopy
from PIL import Image

vfps = None
vframesize = None
def video2list(video_path: str) -> []:
    global vfps, vframesize
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    vfps = cap.get(cv2.CAP_PROP_FPS)
    vframesize = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(vframesize)
    framenum = int(cap.get(7))
    video = []
    for i in tqdm(range(framenum), desc="Video2ImageList"):
        a, b=cap.read()
        if a == False:
            break
        b = np.array(b)
        video.append(b)
    cap.release()
    return video


def list2video(imgs: []) -> str:
    fps = vfps if vfps is not None else 24
    framesize = vframesize if vframesize is not None else (256, 256)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    path = f"{datetime.now().timestamp()}.mp4"
    video = cv2.VideoWriter(path, fourcc=fourcc, fps=float(fps), frameSize=framesize)
    for i in tqdm(imgs, desc="List to Videos"):
        # TODO: Numpy 到底怎么转成 cv2.mat ???
        video.write(i.astype('uint8'))
    video.release()
    cv2.destroyAllWindows()
    return path


def unwrap_param(*args, **kwargs):
    guide_frame = args[0]
    flow_list = video2list(args[1])
    print(type(flow_list[0]))
    mask_type = str(args[2])
    user_forward_warp = bool(args[3])
    user_moving_mask = bool(args[4])
    moving_mask_threshold = float(args[5])
    adaptive_inpainting = bool(args[6])
    return guide_frame, flow_list, mask_type, user_forward_warp, user_moving_mask, moving_mask_threshold, adaptive_inpainting

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
        gr.Markdown("# 测试")
        with gr.Accordion("README", open=False):
            content = """
                # 这是一个标题
                ## 这是二级标题
            """
            gr.Markdown(content)

        with gr.Box():
            gr.Markdown("Input")
            with gr.Column():
                with gr.Row():
                    first_frame = gr.Image(label="第一帧")
                    video_flow = gr.PlayableVideo(format='mp4', source='upload', label="光流")
                mask_type = gr.Dropdown(choices = ["cat", "image_mask2", "image_mask1"], label="Mask Type", value="cat")
                user_forward_warp = gr.Checkbox(value=True, label="user_forward_warp")
                user_moving_mask = gr.Checkbox(value=True, label="user_moving_mask")
                moving_mask_threshold = gr.Slider(1, 5, value=3, label="moving_mask_threshold")
                adaptive_inpainting = gr.Checkbox(value=True, label="adaptive_inpainting")
        return = [
            first_frame,                # 0 
            video_flow,                 # 1
            mask_type,                  # 2
            user_forward_warp,          # 3
            user_moving_mask,           # 4
            moving_mask_threshold,      # 5
            adaptive_inpainting,        # 6
        ]
        pass

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.

    def run(self, p: StableDiffusionProcessingImg2Img) -> Processed:

        guide_frame: np.ndarray = None
        flow_list: List[np.ndarray] = []
        mask_type: str = 'image_mask'  # 'image_mask' or 'latent_mask'
        use_forward_warp: bool = True
        use_moving_mask: bool = True
        moving_mask_threshold: float = 3.0  # threshold for segmenting moving pixels
        adaptive_inpainting: bool = False  # blur the inpainted region based on magnitude of flow

        assert len(flow_list) > 0, "No flow provided"

        result_images : List[np.ndarray] = []

        prev_frame = guide_frame  # the first frame is the guide frame
        for n in range(len(flow_list)):  # generate frames
            # p_now = deepcopy(p)  # make a copy of the processing object  # TODO check
            p_now = p

            second_frame_warped = warp(prev_frame, flow_list[n],  # get the second frame by warping the first frame
                                       warp_mode='forward' if use_forward_warp else 'backward')

            # apply moving mask as inpainting mask
            if use_moving_mask:
                moving_mask = get_moving_mask(flow_list[n],
                                              binarize=not adaptive_inpainting,
                                              # if adaptive inpainting is enabled, we don't need to binarize the mask
                                              threshold=moving_mask_threshold)

                if not adaptive_inpainting:
                    mask = moving_mask *255  # convert to 0-255 range
                    mask = mask.astype(np.uint8)
                else:
                    raise NotImplementedError("Adaptive inpainting is not implemented yet")  #TODO

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
            p_now.init_images = [second_frame_warped]
            # p.do_not_save_grid = True  # TODO check
            # p.control_net_input_image = Image.open(reference_imgs[i].name).convert("RGB").resize(
            #     (initial_width, p.height), Image.ANTIALIAS)  # TODO check

            proc = process_images(p)  # run ddpm
            image = proc.images[0]

            result_images.append(image)

            prev_frame = image  # the current frame becomes the previous frame for the next iteration

        result_images_pil = [Image.fromarray(img) for img in result_images]

        # TODO convert to mp4 and save to disk


        return Processed(images_list=result_images_pil)

