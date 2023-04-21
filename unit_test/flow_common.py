from flow.common import load_flow_to_numpy, rgb_to_flow, flow_to_rgb, draw_color_wheel
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    flow_numpy = load_flow_to_numpy('flow.flo')
    print(flow_numpy.shape)
    flow_max = flow_numpy.max()
    flow_min = flow_numpy.min()
    rgb = flow_to_rgb(flow_numpy)
    plt.imshow(rgb)
    plt.title('flow_rgb')
    plt.show()
    flow_back = rgb_to_flow(rgb, flow_min, flow_max)
    rgb_back = flow_to_rgb(flow_back)
    plt.imshow(rgb_back)
    plt.title('flow_rgb_back')
    plt.show()
    # calculate mean error
    error = np.abs(flow_numpy - flow_back)
    print(f"mean error: {error.mean()}")
    plt.imshow(draw_color_wheel())
    plt.title('color wheel')
    plt.show()
