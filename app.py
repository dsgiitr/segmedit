import os
import cv2
import gradio as gr
import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import ImageDraw, ImageFilter
from utils.tools import box_prompt, format_results, point_prompt
from utils.tools_gradio import fast_process

from utils.editing import inpaint_area
from PIL import Image, ImageOps
from utils.text_editing_SD2 import BlendedLatnetDiffusion


# Latent diffusion
def latent_diffusion(prompt):
    bld = BlendedLatnetDiffusion()
    results = bld.edit_image(
        'input.png',
        'mask.png',
        prompts = [prompt] * bld.args.batch_size,
        blending_percentage=bld.args.blending_start_percentage,
    )
    results_flat = np.concatenate(results, axis=1)
    im = Image.fromarray(results_flat)
    return im

TARGET_WIDTH = 512
TARGET_HEIGHT = 512

def diffuse_image_drag(im_editor, prompt):
    
    # Input Image
    inputImage = Image.open(im_editor['background'])
    input_image = inputImage.resize((512,512))
    input_image = input_image.save('./input.png', 'PNG')
    print('input.png is saved')
    
    # Convert mask to black and white
    mask_image = Image.open(im_editor['layers'][0])
    mask_gray = mask_image.convert('1')
    mask_resized = mask_gray.resize((512, 512))
    mask_resized = mask_resized.save('./mask.png', "PNG")
    
    image = latent_diffusion(prompt)
    
    # Split the 2048x512 image into 4 images of 512x512 pixels each
    images = []
    for i in range(4):
        x = i * 512
        img = image.crop((x, 0, x + 512, 512))
        images.append(img)
    
    return images


def diffuse_image_box(prompt):
    image = latent_diffusion(prompt)
    return image

def mask_with_segmentation(segm_img_p, prompt):
    global mask
    mask = mask.convert('L').filter(ImageFilter.GaussianBlur(radius=5))
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    binary_mask = (mask > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary_mask)
    mask_img.save("./mask.png", "PNG")
    print("mask.png saved")
    
    output_img = diffuse_image_box(prompt)
    
    return output_img


global_start_point = None
current_rectangle = None
x0,y0,x1,y1=0,0,0,0

def handle_rectangle_events(image,label, evt: gr.SelectData):
    global global_start_point, current_rectangle
    # Save image
    image.save('./input.png', 'PNG')
    print('input.png is saved')

    x, y = evt.index[0], evt.index[1]
    if global_start_point is None:
        global_start_point = (x, y)
    x0, y0 = global_start_point
    x1, y1 = x, y



    if (x0>x1):
        temp=x0
        x0=x1
        x1=temp

    if (y0>y1):
        temp=y0
        y0=y1
        y1=temp

    image_with_rectangle = image.copy()
    draw = ImageDraw.Draw(image_with_rectangle)
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)


    current_rectangle = [x0, y0, x1, y1]

    if (x0!=x1) and (y0!=y1):
        masked_image = image_with_rectangle.copy()

        masked_image=cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)

        mask = np.zeros(masked_image.shape[:2], dtype=np.uint8)
        print(masked_image.shape)
        print(mask.shape)
        
        if len(masked_image.shape) == 3 and masked_image.shape[2] == 3:
            mask = cv2.merge([mask, mask, mask])
        mask[y0:y1, x0:x1] = [255,255,255]
        masked_image[y0:y1, x0:x1] = [255, 255, 255]

        # Apply the mask to the image
        print(mask.shape)
        masked_image = cv2.bitwise_and(masked_image, mask)
        cv2.imwrite('./mask.png', masked_image)
        print('mask.png is saved')
        
        return image_with_rectangle
    
    return image_with_rectangle


def create_masked_image_from_rectange(image_with_rectangle):
    if (x0!=x1) and (y0!=y1):
        masked_image = image_with_rectangle.copy()

        masked_image=cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)

        mask = np.zeros(masked_image.shape[:2], dtype=np.uint8)
        print(masked_image.shape)
        print(mask.shape)
        
        if len(masked_image.shape) == 3 and masked_image.shape[2] == 3:
            mask = cv2.merge([mask, mask, mask])
        mask[y0:y1, x0:x1] = [255,255,255]
        masked_image[y0:y1, x0:x1] = [255, 255, 255]

        # Apply the mask to the image
        # print(mask.shape)
        masked_image = cv2.bitwise_and(masked_image, mask)

        return masked_image
    
    return image_with_rectangle


# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sam_checkpoint = "models/mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)
mask = None

# Description
title = "<center><strong><font size='8'>SEGMEDIT<font></strong></center>"

description_e = """
                
              """

description_p = """ 

              """

examples = [
    ["utils/assets/picture3.jpg"],
    ["utils/assets/picture4.jpg"],
    ["utils/assets/picture5.jpg"],
    ["utils/assets/picture6.jpg"],
    ["utils/assets/picture1.jpg"],
    ["utils/assets/picture2.jpg"],
]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global mask_generator

    input_size = int(input_size)
    
    
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)
    
    global mask

    fig, mask = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )    
    return fig


def segment_with_points(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label
    
    
    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_point_label = np.array(global_point_label)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, image

    print(scaled_points, scaled_points is not None)
    print(scaled_point_label, scaled_point_label is not None)

    nd_image = np.array(image)
    predictor.set_image(nd_image)
    if np.all(scaled_point_label==1):
        masks, scores, logits = predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_point_label,
        multimask_output=True,
        )
    else:
        masks, scores, logits = predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_point_label,
        multimask_output=False,
        )

    results = format_results(masks, scores, logits, 0)

    annotations, _ = point_prompt(
        results, scaled_points, scaled_point_label, new_h, new_w
    )
    annotations = np.array([annotations])
    
    global mask

    fig, mask = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    global_points = []
    global_point_label = []
    # return fig, None
    return fig, image


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label
    
    image.save('input.png', "PNG")
    print("input.png saved")
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0) if label == "Add Area" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Area" else 0)

    print(x, y, label == "Add Area")

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image


# cond_img_e = gr.Image(label="Input", value=default_example[0], type="pil")
cond_img_p = gr.Image(label="Input with points", value=default_example[0], type="pil")

# segm_img_e = gr.Image(label="Segmented Image", interactive=False, type="pil")
segm_img_p = gr.Image(
    label="Segmented Image with points", interactive=False, type="pil"
)

# Latent Diffusion
cond_img_b = gr.Image(label="Input with box", value=default_example[0], type="pil")
segm_img_b = gr.Image(label="Diffused image after box selection", interactive=False, type="pil")

global_points = []
global_point_label = []

input_size_slider = gr.components.Slider(
    minimum=512,
    maximum=1024,
    value=1024,
    step=64,
    label="Input_size",
    info="Our model was trained on a size of 1024",
)


def erase(image, display_img):
    # inpaint the image with the mask
    global mask
    inpainted_image = inpaint_area(image, mask)
    return display_img, inpainted_image


with gr.Blocks(css=css, title="SEGMEDIT") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown('''
        <h1 style="text-align: center;">Segment and Edit/Remove</h1>
        <h3 style="text-align: center;">DSG, IIT Roorkee</h3>
        <br> <br>
    ''')

   
    with gr.Tab("Point mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(
                        label="Point Prompts",
                        choices=["Add Area", "Remove Area"],
                        value="Add Area",
                        info="Positive points are included in the segment,negative points are excluded",
                    )

                    text_prompt = gr.Textbox(
                        label="Text Prompts", lines=6, interactive=True
                    )
                # with gr.Row():
                # with gr.Column():

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_p],
                    # outputs=segm_img_p,
                    # fn=segment_with_points,
                    # cache_examples=True,
                    examples_per_page=4,
                )

            with gr.Column():
                segment_btn_p = gr.Button("Start segmenting", variant="primary")
                clear_btn_p = gr.Button("Restart", variant="secondary")
                # create a button which finalizes the mask and takes to new block
                erase_btn = gr.Button("Erase", variant="secondary")
                edit_btn_p = gr.Button("Edit", variant="secondary")
                # Description
                gr.Markdown(description_p)
    
    with gr.Tab("Box selection"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_b.render()

            with gr.Column(scale=1):
                segm_img_b.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove_box = gr.Radio(
                        label="Box Prompts",
                        choices=["Add Box"],
                        value="Add Box",
                    )

                    text_prompt_box = gr.Textbox(
                        label="Text Prompts", lines=6, interactive=True
                    )

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_b],
                )

            with gr.Column():
                edit_btn_b = gr.Button("Edit", variant="primary")
                clear_btn_b = gr.Button("Restart", variant="secondary")
                # erase_btn_b = gr.Button("Erase", variant="secondary")
                gr.Markdown(description_p)

    with gr.Tab("Drag Selection"):
        # Submit & Clear

        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                brush = gr.Brush(default_color="#ffffff")
                drag_img = gr.ImageEditor(brush=brush, label="Upload an Image and draw a Mask over it", type="filepath")

            with gr.Column(scale=1):
                drag_output = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery",
                                         columns=[4], rows=[1], object_fit="contain", height="auto")

        with gr.Row():
            with gr.Column():
                text_prompt_box = gr.Textbox(
                    label="Text Prompts", lines=6, interactive=True
                )   
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=drag_img,
                    outputs=drag_img,
                    examples_per_page=4,
                )

            with gr.Column():
                diffuse_image_btn = gr.Button("Start editing", variant="primary")
                clear_btn_d = gr.Button("Restart", variant="secondary")
                gr.Markdown(description_p)
                
                


    cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)
    cond_img_b.select(handle_rectangle_events, [cond_img_b,add_or_remove_box], cond_img_b)
    
    cond_img_b = create_masked_image_from_rectange(cond_img_b)
    
    segment_btn_p.click(
        segment_with_points,
        inputs=[cond_img_p],
        outputs=[segm_img_p, cond_img_p],
    )
    
    # Box selection diffusion model button
    edit_btn_b.click(
        diffuse_image_box,
        inputs=[text_prompt_box],
        outputs=segm_img_b        
    )
    
    # Mask and Editing using point prompt
    edit_btn_p.click(
        mask_with_segmentation,
        inputs=[segm_img_p, text_prompt],
        outputs=segm_img_p    
    )
    # Drag selection diffusion model button
    diffuse_image_btn.click(
        diffuse_image_drag,
        inputs = [drag_img, text_prompt_box],
        outputs=[drag_output]
        )

    
    def clear():
            global global_start_point
            global current_rectangle

            global_start_point=None
            current_rectangle=None
            return None, None

    def clear_text():
        return None, None, None

    # clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])
    clear_btn_b.click(clear, outputs=[cond_img_b, segm_img_b])
    clear_btn_d.click(clear, outputs=[drag_img, drag_output])
    
    
    # close the demo and take to new block when finalize button is clicked
    erase_btn.click(erase, inputs=[cond_img_p, segm_img_p], outputs=[cond_img_p, segm_img_p])
    
    
    


if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True, share=True)
