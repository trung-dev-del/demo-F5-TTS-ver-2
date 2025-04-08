# F5-TTS-Vietnamese
Fine-tuning pipline for training a Vietnamese speech synthesis model using the F5-TTS architecture.

Try demo at: https://huggingface.co/spaces/hynt/F5-TTS-Vietnamese-100h

## Tips for training
- 100 hours of data is good enough for a Text-to-Speech model in Vietnamese with specific voices. However, to achieve optimal performance for voice cloning across diverse voices, I think more data is needed. I fine-tuned an F5-TTS version on a dataset of about 1000 hours, and the results were excellent for voice cloning.
- It’s crucial to have a large number of speaker hours with highly accurate transcriptions - the more, the better. This allows other speaker sets to generalize well, leading to a lower WER after training and reducing hallucinations.

## Tips for inference
- Suggest selecting sample audios that are clear and have minimal interruptions, as this will improve the synthesis results.
- In cases where the reference audio text is not provided, the default model used will be whisper-large-v3-turbo. As a result, there may be instances where Vietnamese is not accurately recognized, leading to poor speech synthesis results.

## Installation

### Create a separate environment if needed

```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n f5-tts python=3.10
conda activate f5-tts
```

### Install PyTorch

> ```bash
> # Install pytorch with your CUDA version, e.g.
> pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
> ```

### Install f5-tts module:

> ```bash
> cd F5-TTS-Vietnamese
> pip install -e .
> ```

### Install sox, ffmpeg

> ```bash
> sudo apt-get update
> sudo apt-get install sox ffmpeg
> ```

## Fine-tuning pipline

Steps:

- Prepare audio_name data and corresponding text
- Add vocabulary from your dataset that is not present in the pretrained model's vocabulary
- Expand the pretrained model's embedding to support the new vocabulary set
- Feature extraction
- Perform fine-tuning

```bash
bash fine_tuning.sh
```

### Inference

```bash
bash infer.sh
```
