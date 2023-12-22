from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np


def inpaint_area(image, mask):
    """
    Inpaints the given image with the given mask.
    Must ensure that the image and mask are PIL images.
    """
    simple_lama = SimpleLama()
    mask = mask.convert('L')

    # make sure the mask is binary
    mask = np.array(mask)
    mask[mask > 0] = 255
    mask = Image.fromarray(mask).convert('L')

    return simple_lama(image, mask)
