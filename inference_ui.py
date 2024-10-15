import os

import gradio as gr

from predictor import Predictor

root_path = os.path.split(os.path.realpath(__file__))[0]
print("__file__", __file__)
print("root_path", root_path)
predictor = Predictor(root_path + "/OUTPUT_MODEL/G_latest.pth", root_path + "/OUTPUT_MODEL/config.json")
os.makedirs(root_path + "/OUTPUT/", exist_ok=True)


def tts_fn(text, speaker, language, speed):
    print("hello world")
    return "Success", None


if __name__ == "__main__":
    speakerMap = predictor.get_speakers()
    speakers = speakerMap.keys()

    lang = ["EN"]

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
                    language_dropdown = gr.Dropdown(choices=lang, value=lang[0], label='language')
                    duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
                                                label='速度 Speed')
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn,
                              inputs=[textbox, char_dropdown, language_dropdown, duration_slider, ],
                              outputs=[text_output, audio_output])
    app.launch(share=False, server_port=7080)
