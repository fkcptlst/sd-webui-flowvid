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
        # TODO: add UI components, return a dictionary
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









