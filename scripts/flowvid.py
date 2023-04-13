import gradio as gr
import os

import modules.scripts as scripts
from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed, StableDiffusionProcessingImg2Img
from modules.shared import opts, cmd_opts, state


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

    def run(self, p: StableDiffusionProcessingImg2Img, args_dict: dict) -> Processed:
        pass
