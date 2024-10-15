import os

import gradio as gr

from predictor import Predictor

root_path = os.path.split(os.path.realpath(__file__))[0]
print("__file__", __file__)
print("root_path", root_path)
predictor = Predictor(root_path + "/OUTPUT_MODEL/G_latest.pth", root_path + "/OUTPUT_MODEL/config.json")


def tts_fn(text, speaker):
    print("INFERENCE_UI:", text, speaker)
    audio = predictor.predict_id(text, speaker, 1.0, "EN")
    return "Success", (predictor.sampling_rate(), audio)


if __name__ == "__main__":
    lang = ["EN"]
    speakers = ["Jack", "Michael", "Bob"]

    app = gr.Blocks()
    with app:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value="hello world", elem_id=f"tts-input")
                    # select character
                    char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn,
                              inputs=[textbox, char_dropdown],
                              outputs=[text_output, audio_output])
    app.launch(share=False, server_port=7080, server_name="0.0.0.0")
