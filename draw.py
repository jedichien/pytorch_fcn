from matplotlib import pyplot
import numpy as np
from data_helper import (denormal, toRGB, findmax)
from labelmap import label_map

def draw_img(rgb, segmentation, n_class, opacity):
    rgb[segmentation>0] *= 1-opacity
    # mask
    mask = np.zeros_like(rgb, dtype=np.float32)
    for clsid in range(n_class):
        mask += np.dot((segmentation==clsid)[..., np.newaxis], [label_map[clsid]])
    # paste
    rgb = np.clip(np.round(rgb+mask*opacity), 0, 255.0).astype(np.uint8)
    return rgb
        
def paint_grid(_X, predict_map, n_class=2, col=5, opacity=0.5, figsize=(20, 6)):
    batch_size = _X.shape[0]
    row = int(batch_size/col)
    if batch_size%col > 0: row += 1
    _rgbX = denormal(_X)
    _rgbX = toRGB(_rgbX, dtype=np.float32)
    fig, ax = pyplot.subplots(row, col, figsize=figsize)
    for i in range(batch_size):
        render = draw_img(_rgbX[i], predict_map[i], n_class=n_class, opacity=opacity)
        ax[i//col][i%col].imshow(render, aspect='auto')
    pyplot.show()

def direct_render(_X, predict, n_class=2, opacity=0.5):
    renders = []
    rgb = denormal(_X)
    rgb = toRGB(rgb, dtype=np.float32)
    predict_map = findmax(predict, n_class)
    for i, segmentation in enumerate(predict_map):
        render = draw_img(rgb[i], segmentation, n_class=n_class, opacity=opacity)
        renders.append(render)
    renders = np.array(renders)
    return renders