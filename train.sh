cd /workspace/VITS-fast-fine-tuning

ln -s /workspace/res/raw_audio ./raw_audio

rm -rf /workspace/res/segmented_character_voice
mkdir /workspace/res/segmented_character_voice
ln -s /workspace/res/segmented_character_voice ./segmented_character_voice

rm -rf /workspace/res/denoised_audio
mkdir /workspace/res/denoised_audio
ln -s /workspace/res/denoised_audio ./denoised_audio

rm -rf /workspace/res/pretrained_models
mkdir /workspace/res/pretrained_models
ln -s /workspace/res/pretrained_models ./pretrained_models


rm -rf /workspace/res/separated
mkdir /workspace/res/separated
ln -s /workspace/res/separated ./separated


wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/configs/uma_trilingual.json -O ./configs/finetune_speaker.json


python scripts/denoise_audio.py
python scripts/long_audio_transcribe.py --languages "CJE" --whisper_size large
python preprocess_v2.py --languages "CJE"

python finetune_speaker_v2.py -m ./OUTPUT_MODEL --max_epochs 100 --drop_speaker_embed True
