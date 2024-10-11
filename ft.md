```

docker run \
    -v ./res:/workspace/res \
    --shm-size=16g \
    -p 7080:7080 \
    -d --gpus all --name fastvits fastvits-base:v1 tail -f /dev/null

```

```

cd /workspace/VITS-fast-fine-tuning

ln -s /workspace/res/raw_audio ./raw_audio
sudo mkdir segmented_character_voice
sudo mkdir OUTPUT_MODEL


wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/configs/uma_trilingual.json -O ./configs/finetune_speaker.json


python scripts/denoise_audio.py
python scripts/long_audio_transcribe.py --languages "CJE" --whisper_size large
python scripts/resample.py
python preprocess_v2.py --languages "CJE"

python finetune_speaker_v2.py -m ./OUTPUT_MODEL --max_epochs 100 --drop_speaker_embed True



```
