import numpy as np
import gradio as gr

from click_events import update_point_prompts, segment_with_prompts

with gr.Blocks() as demo:
    gr.Markdown('''
        <h1 style="text-align: center;">Segment and Edit/Remove</h1>
        <h3 style="text-align: center;">DSG, IIT Roorkee</h3>
        <br> <br>
    ''')

    with gr.Row():
        with gr.Column(scale=1):
            pass

        with gr.Column(scale=2):
            input_image = gr.Image(
                label="Input Image",
                sources="upload"
            )
            
            with gr.Row():
                with gr.Column():
                    point_prompt = gr.Textbox(
                        label="Point Prompts",
                        lines=1,
                        interactive=False
                    )
                    point_type = gr.Radio(
                        choices=["positive", "negative"],
                        value="positive",
                        label="Point Type",
                        info="Positive points are included in the segment, negative points are excluded."
                    )
                
                text_prompt = gr.Textbox(
                    label="Text Prompts",
                    lines=6,
                    interactive=True
                )

            with gr.Row():
                clear_prompts = gr.Button(value="Clear Prompts")
                submit_prompts = gr.Button(value="Submit Prompts")

        with gr.Column(scale=1):
            pass

        input_image.select(
            update_point_prompts,
            inputs=[point_prompt, point_type],
            outputs=point_prompt
        )
        
        clear_prompts.click(
            lambda: (None, None),
            inputs=None, 
            outputs=[point_prompt, text_prompt],
            queue=False
        )
        
        submit_prompts.click(
            segment_with_prompts,
            inputs=[input_image, point_prompt, text_prompt],
            outputs=input_image
        )

demo.launch()
