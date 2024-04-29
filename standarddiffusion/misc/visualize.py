import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from standarddiffusion.misc.utils import detach_to_numpy


def image_grid(
    images: Tensor | np.ndarray,
    n_cols: int = 5,
    figsize: tuple = (10, 10),
):
    if images.dtype == Tensor:
        images = detach_to_numpy(images)

    n_images = len(images)

    fig, ax = plt.subplots(n_cols % n_images, n_cols, figsize=figsize)
    for i in range(n_images):
        ax[i].imshow(images[i])

    plt.tight_layout()

    return fig
