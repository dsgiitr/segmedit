import gradio as gr

def update_point_prompts(points, type, evt: gr.SelectData):
    click_pixels = str((evt.index[1], evt.index[0]))

    # if type == 'negative':
    #     click_pixels = gr.Markdown('<span style="color:red">' + click_pixels + '</span>')
    # else:
    #     click_pixels = gr.Markdown('<span style="color:green">' + click_pixels + '</span>')
    
    if points:
        return points + " " + click_pixels
    return str(click_pixels)

def segment_with_prompts(image, point_prompt, text_prompt):
    return image
    