import numpy as np 
from numpy.random import multivariate_normal as mvn 
import matplotlib.pyplot as plt 

from typing import List


def show_images(imgs: List[float], num_rows, num_cols, scale=2, 
                titles: List[int] = None, save_fig=True, fig_name='image'):
    """Show (a list of) binary images.
    """

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img, cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles is not None:
            ax.set_title(titles[i])

    if save_fig: 
        plt.savefig(f'images/{fig_name}')
        
    plt.show()

    
