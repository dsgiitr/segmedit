from simple_lama_inpainting import SimpleLama
from PIL import Image, ImageFilter
import numpy as np


def inpaint_area(image, mask, blur_radius=5):
    """
    Inpaints the given image with the given mask.
    Must ensure that the image and mask are PIL images.
    blur_radius is the radius of the Gaussian blur that is applied to the mask.
    this is done to make the mask borders larger and to avoid the object getting inpainted.
    """
    simple_lama = SimpleLama()
    mask = mask.convert('L').filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # make sure the mask is binary
    mask = np.array(mask)
    mask[mask > 0] = 255
    mask = Image.fromarray(mask).convert('L')

    return simple_lama(image, mask)

def inpaint_area_using_box():
    pass