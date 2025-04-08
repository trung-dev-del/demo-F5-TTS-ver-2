import json
import pickle
import shutil
import uuid
from pathlib import Path
from importlib.resources import files
import torch
import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
import tempfile
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import re
import os

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False

def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

DATAS_DIR = Path("datas")
DATAS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = DATAS_DIR / "audios"
AUDIO_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = DATAS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
SPECTROGRAM_DIR = DATAS_DIR / "spectrograms"
SPECTROGRAM_DIR.mkdir(exist_ok=True)

INITIAL_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

tts_model_choice = ["F5-TTS", INITIAL_TTS_MODEL_CFG[0], INITIAL_TTS_MODEL_CFG[1], json.loads(INITIAL_TTS_MODEL_CFG[2])]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocoder = load_vocoder().to(device)

def load_f5tts():
    ckpt_path = str(cached_path(INITIAL_TTS_MODEL_CFG[0]))
    vocab_path = str(cached_path(INITIAL_TTS_MODEL_CFG[1]))
    F5TTS_model_cfg = json.loads(INITIAL_TTS_MODEL_CFG[2])
    print(f"Loading F5-TTS from {ckpt_path} with config: {F5TTS_model_cfg}")
    return load_model(DiT, F5TTS_model_cfg, ckpt_path, vocab_file=vocab_path).to(device)

def load_e2tts():
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
    print(f"Loading E2-TTS from {ckpt_path} with config: {E2TTS_model_cfg}")
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path).to(device)

def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if not ckpt_path:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path and vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    
    if model_cfg is None:
        model_cfg = json.loads(INITIAL_TTS_MODEL_CFG[2])
        print(f"Using default config: {model_cfg}")
    else:
        model_cfg = json.loads(model_cfg) if isinstance(model_cfg, str) else model_cfg
    
    print(f"Loading custom model from: {ckpt_path}, vocab: {vocab_path if vocab_path else 'None'}, config: {model_cfg}")
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path if vocab_path else None).to(device)

F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

def generate_uuid(name=None):
    if name and name.strip():
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, name.strip()))
    return str(uuid.uuid4())

def preprocess_and_save_checkpoint(
    ref_audio_orig, ref_text, person_name=None, base_checkpoint=None, checkpoint_dir=CHECKPOINT_DIR, show_info=gr.Info
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return None

    ref_text_input = ref_text.strip() if ref_text and isinstance(ref_text, str) else ""
    ref_audio, ref_text_auto = preprocess_ref_audio_text(ref_audio_orig, ref_text_input, show_info=show_info)
    ref_text = ref_text_input if ref_text_input else ref_text_auto

    if not ref_text.strip():
        gr.Warning("Generated ref_text is empty. Please enter manually or check audio.")
        return None

    checkpoint_id = str(uuid.uuid4())
    checkpoint_name = f"checkpoint_{checkpoint_id}"
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.ckpts"
    wav_dest = AUDIO_DIR / f"{checkpoint_name}.wav"
    shutil.copy(ref_audio_orig, wav_dest)

    if base_checkpoint and base_checkpoint.strip():
        base_checkpoint = base_checkpoint.strip()
    else:
        model_type = tts_model_choice[0]
        if model_type == "F5-TTS":
            base_checkpoint = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
        elif model_type == "E2-TTS":
            base_checkpoint = "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"
        elif model_type == "Custom":
            base_checkpoint = tts_model_choice[1] if tts_model_choice[1].strip() else ""

    vocab_path = tts_model_choice[2]

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        f.write(f"id: {checkpoint_id}\n")
        f.write(f"person_name: {person_name if person_name else 'Unnamed'}\n")
        f.write(f"wav_file: {wav_dest}\n")
        f.write(f"ref_text: {ref_text}\n")
        f.write(f"model_choice: {tts_model_choice[0]}\n")
        f.write(f"base_checkpoint_path: {base_checkpoint}\n")
        f.write(f"vocab_path: {vocab_path}\n")
    
    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path

@gpu_decorator
def generate_ref_text(ref_audio_orig, show_info=gr.Info):
    if not ref_audio_orig:
        gr.Warning("Vui lòng cung cấp âm thanh tham chiếu.")
        return ""
    _, ref_text = preprocess_ref_audio_text(ref_audio_orig, "", show_info=show_info)
    if not ref_text.strip():
        gr.Warning("Văn bản tham chiếu trống. Kiểm tra âm thanh hoặc nhập thủ công.")
    print(f"Generated ref_text: {ref_text}")
    return ref_text

def list_checkpoints(checkpoint_dir=CHECKPOINT_DIR):
    checkpoints = {}
    for f in checkpoint_dir.glob("*.ckpts"):
        try:
            with open(f, "r", encoding="utf-8") as checkpoint_file:
                lines = checkpoint_file.readlines()
                data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in lines}
                name = data["person_name"]
                base_name = name
                counter = 1
                while name in checkpoints:
                    name = f"{base_name}_{counter}"
                    counter += 1
                checkpoints[name] = str(f)
        except Exception as e:
            gr.Warning(f"Failed to load checkpoint {f}: {str(e)}")
    print(f"Available checkpoints: {checkpoints}")
    return checkpoints

def show_checkpoint_info(checkpoint_name):
    if not checkpoint_name:
        return "No checkpoint selected"
    checkpoints = list_checkpoints()
    checkpoint_path = checkpoints.get(checkpoint_name)
    if not checkpoint_path:
        return "Checkpoint not found"
    
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in f.readlines()}
        info = f"""
Person: {data['person_name']}
Model: {data['model_choice']}
Base: {data.get('base_checkpoint_path', 'None')}
Created: {Path(checkpoint_path).stat().st_ctime}
"""
        print(f"Checkpoint info: {info}")
        return info
    except Exception as e:
        return f"Error loading info: {str(e)}"

@gpu_decorator
def infer_from_checkpoint(
    checkpoint_person_name, gen_text, remove_silence, cross_fade_duration=0.15, nfe_step=16, speed=0.5, 
    auto_adjust_speed=True, show_info=gr.Info
):
    print(f"Torch: {torch.__version__}, Torchaudio: {torchaudio.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BASE_DIR = Path(__file__).resolve().parent

    # Tham số mặc định từ CLI
    cli_model_name = "F5TTS_Base"
    cli_ref_audio = str(BASE_DIR / "ref.wav")
    cli_ref_text = "cả hai bên hãy cố gắng hiểu cho nhau"
    cli_ckpt_file = "E:\\F5_TTS\\F5-TTS-Vietnamese\\ckpts\\your_training_dataset\\model_500000.safetensors"  # Dùng từ log của bạn
    cli_vocab_file = "E:\\F5_TTS\\F5-TTS-Vietnamese\\data\\your_training_dataset\\vocab.txt"  # Dùng từ log của bạn
    cli_vocoder_name = "vocos"
    output_dir = BASE_DIR / "tests"
    output_dir.mkdir(exist_ok=True)
    wave_path = output_dir / "output_raw.wav"

    # Tải thông tin từ checkpoint (nếu có)
    ref_audio = cli_ref_audio
    ref_text = cli_ref_text
    if checkpoint_person_name:
        checkpoints = list_checkpoints()
        checkpoint_path = checkpoints.get(checkpoint_person_name)
        if not checkpoint_path:
            gr.Warning("Please select a valid checkpoint or relying on default CLI ref_audio.")
        else:
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = {line.split(": ")[0]: line.split(": ")[1].strip() for line in f.readlines()}
                ref_audio = checkpoint_data["wav_file"]
                ref_text = checkpoint_data["ref_text"]
                cli_model_name = checkpoint_data.get("model_choice", cli_model_name)
                cli_ckpt_file = checkpoint_data.get("base_checkpoint_path", cli_ckpt_file)
                cli_vocab_file = checkpoint_data.get("vocab_path", cli_vocab_file)
                print(f"Loaded checkpoint data: {checkpoint_data}")
            except Exception as e:
                gr.Warning(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")

    # Kiểm tra tệp
    if not Path(ref_audio).exists():
        gr.Warning(f"Reference audio not found: {ref_audio}")
        return None, None, ref_text
    if not Path(cli_ckpt_file).exists():
        gr.Warning(f"Checkpoint file not found: {cli_ckpt_file}")
        return None, None, ref_text
    if not Path(cli_vocab_file).exists():
        gr.Warning(f"Vocab file not found: {cli_vocab_file}")
        return None, None, ref_text
    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return None, None, ref_text

    # Tải mô hình
    if cli_model_name == "F5TTS_Base":
        model_cfg = {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
        ema_model = load_model(DiT, model_cfg, cli_ckpt_file, vocab_file=cli_vocab_file).to(device)
        ema_model.eval()
        print(f"Loaded F5TTS_Base model from {cli_ckpt_file}")
    elif cli_model_name == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model.to(device)
        print("Using E2-TTS model")
    elif cli_model_name == "Custom":
        global custom_ema_model, pre_custom_path
        if pre_custom_path != cli_ckpt_file:
            custom_ema_model = load_custom(cli_ckpt_file, cli_vocab_file)
            pre_custom_path = cli_ckpt_file
        ema_model = custom_ema_model.to(device)
        print(f"Using Custom model from: {cli_ckpt_file}, vocab: {cli_vocab_file}")
    else:
        gr.Warning(f"Unknown model type: {cli_model_name}")
        return None, None, ref_text

    # Tải vocoder
    vocoder = load_vocoder(vocoder_name=cli_vocoder_name).to(device)
    vocoder.eval()
    print(f"Loaded vocoder: {cli_vocoder_name}")

    # Điều chỉnh tốc độ
    adjusted_speed = speed
    if auto_adjust_speed:
        ref_wave, ref_sample_rate = torchaudio.load(ref_audio)
        ref_duration = ref_wave.shape[-1] / ref_sample_rate
        ref_text_length = len(ref_text.split())
        gen_text_length = len(gen_text.split())
        ref_speed = ref_text_length / ref_duration
        target_duration = gen_text_length / ref_speed
        adjusted_speed = speed * (ref_duration / target_duration)
        adjusted_speed = max(0.3, min(2.0, adjusted_speed))
        print(f"Adjusted speed: {adjusted_speed} (original: {speed}, target_duration: {target_duration}s)")
    else:
        print(f"Using manual speed: {adjusted_speed}")

    # Chuẩn bị voices giống CLI
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = {"main": main_voice}
    voices["main"]["ref_audio"], voices["main"]["ref_text"] = preprocess_ref_audio_text(
        voices["main"]["ref_audio"], voices["main"]["ref_text"], show_info=show_info
    )

    # Chia văn bản thành các đoạn như CLI
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    generated_audio_segments = []

    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        gen_text_ = text.strip()
        print(f"Processing voice: {voice}, gen_text: {gen_text_}")

        try:
            audio_segment, final_sample_rate, spectrogram = infer_process(
                ref_audio_,
                ref_text_,
                gen_text_,
                ema_model,
                vocoder,
                mel_spec_type=cli_vocoder_name,
                target_rms=0.2,  # Mặc định từ CLI
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=2.0,  # Mặc định từ CLI
                sway_sampling_coef=-1,  # Mặc định từ CLI
                speed=adjusted_speed,
                fix_duration=None,
            )
            if isinstance(audio_segment, torch.Tensor):
                audio_segment = audio_segment.cpu().numpy()
            generated_audio_segments.append(audio_segment)
        except Exception as e:
            gr.Warning(f"Infer process failed for chunk '{gen_text_}': {str(e)}")
            return None, None, ref_text

    # Nối các đoạn âm thanh
    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
    else:
        gr.Warning("No audio segments generated.")
        return None, None, ref_text

    # Chuẩn hóa và chuyển sang int16
    if final_wave.max() > 1.0 or final_wave.min() < -1.0:
        final_wave = final_wave / max(abs(final_wave.max()), abs(final_wave.min()))
    final_wave_int16 = (final_wave * 32767).astype(np.int16)

    # Lưu âm thanh
    sf.write(wave_path, final_wave_int16, final_sample_rate)
    print(f"Saved raw audio to {wave_path}")

    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave_int16, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave_int16, _ = torchaudio.load(f.name)
        final_wave_int16 = final_wave_int16.squeeze().cpu().numpy()
        print("Silence removed")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(spectrogram, spectrogram_path)

    # Giải phóng bộ nhớ
    del ema_model, vocoder
    torch.cuda.empty_cache()
    gc.collect()

    return (final_sample_rate, final_wave_int16), spectrogram_path, ref_text

@gpu_decorator
def basic_tts(checkpoint_person_name, gen_text_input, remove_silence, cross_fade_duration_slider, nfe_slider, speed_slider, auto_adjust_speed):
    audio_out, spectrogram_path, ref_text_out = infer_from_checkpoint(
        checkpoint_person_name,
        gen_text_input,
        remove_silence,
        cross_fade_duration=cross_fade_duration_slider,
        nfe_step=nfe_slider,
        speed=speed_slider,
        auto_adjust_speed=auto_adjust_speed
    )
    return audio_out, spectrogram_path, ref_text_out

def reload_checkpoints():
    choices = list_checkpoints().keys()
    return gr.update(choices=choices), gr.update(choices=choices)

initial_checkpoints = list_checkpoints()
checkpoint_choices = initial_checkpoints.keys()

with gr.Blocks() as app:
    gr.Markdown(
        f"""
# E2/F5 TTS
This is {"a local web UI" if not USING_SPACES else "an online demo"} for F5-TTS with checkpoint support.
"""
    )
    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return INITIAL_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        print(f"Switching TTS model to: {new_choice}")
        if new_choice == "Custom":
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = load_last_used_custom()
            tts_model_choice = ["Custom", custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        elif new_choice == "F5-TTS":
            tts_model_choice = ["F5-TTS", "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors", "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt", json.loads(INITIAL_TTS_MODEL_CFG[2])]
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        elif new_choice == "E2-TTS":
            tts_model_choice = ["E2-TTS", "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors", "", dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)]
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        if not custom_ckpt_path or not custom_ckpt_path.strip():
            gr.Warning("Custom checkpoint path cannot be empty when selecting Custom model.")
            tts_model_choice = ["Custom", "", custom_vocab_path.strip(), json.loads(custom_model_cfg)]
        else:
            tts_model_choice = ["Custom", custom_ckpt_path.strip(), custom_vocab_path.strip(), json.loads(custom_model_cfg)]
            with open(last_used_custom, "w", encoding="utf-8") as f:
                f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")
        print(f"Updated tts_model_choice: {tts_model_choice}")

    with gr.Row():
        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=["F5-TTS", "E2-TTS", "Custom"], label="Choose TTS Model", value="F5-TTS"
            )
        else:
            choose_tts_model = gr.Radio(
                choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS"
            )
        custom_ckpt_path = gr.Dropdown(
            choices=[""],
            value=load_last_used_custom()[0],
            allow_custom_value=True,
            label="Checkpoint Path (local or hf://)",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[""],
            value=load_last_used_custom()[1],
            allow_custom_value=True,
            label="Vocab Path (local or hf://)",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[INITIAL_TTS_MODEL_CFG[2]],
            value=load_last_used_custom()[2],
            allow_custom_value=True,
            label="Model Config (JSON dict)",
            visible=False,
        )

    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

    with gr.Blocks() as app_tts:
        gr.Markdown("# Batched TTS with Checkpoint")
        user_name_input = gr.Textbox(label="User Name (Optional, for UUID)", placeholder="Enter your name here")
        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
        gen_text_input = gr.Textbox(label="Text to Generate", lines=10, placeholder="Enter text to synthesize")
        
        with gr.Row():
            base_checkpoint_dropdown = gr.Dropdown(
                label="Base Checkpoint (Optional)",
                choices=checkpoint_choices,
                value=None,
                interactive=True
            )
            checkpoint_dropdown = gr.Dropdown(
                label="Select Checkpoint",
                choices=checkpoint_choices,
                value=None,
                interactive=True
            )
        
        with gr.Row():
            generate_ref_btn = gr.Button("Generate Ref Text", variant="secondary")
            save_checkpoint_btn = gr.Button("Save Checkpoint", variant="primary")
            load_checkpoint_btn = gr.Button("Load Checkpoints", variant="secondary")
            synthesize_btn = gr.Button("Synthesize", variant="primary")
        
        with gr.Accordion("Advanced Settings", open=False):
            ref_text_input = gr.Textbox(
                label="Reference Text",
                info="Click 'Generate Ref Text' to auto-transcribe or enter manually.",
                lines=2,
            )
            remove_silence = gr.Checkbox(label="Remove Silences", value=False)
            speed_slider = gr.Slider(label="Speed", minimum=0.3, maximum=2.0, value=1.0, step=0.1)
            nfe_slider = gr.Slider(label="NFE Steps", minimum=4, maximum=64, value=32, step=2)
            cross_fade_duration_slider = gr.Slider(label="Cross-Fade Duration (s)", minimum=0.0, maximum=1.0, value=0.15, step=0.01)
            auto_adjust_speed = gr.Checkbox(label="Auto Adjust Speed", value=True)

        audio_output = gr.Audio(label="Synthesized Audio")
        spectrogram_output = gr.Image(label="Spectrogram")
        checkpoint_output = gr.Textbox(label="Generated Checkpoint Path")
        checkpoint_info_output = gr.Textbox(label="Checkpoint Info")

        generate_ref_btn.click(generate_ref_text, inputs=[ref_audio_input], outputs=[ref_text_input])
        
        save_checkpoint_btn.click(
            preprocess_and_save_checkpoint,
            inputs=[ref_audio_input, ref_text_input, user_name_input, base_checkpoint_dropdown],
            outputs=[checkpoint_output],
        )

        load_checkpoint_btn.click(reload_checkpoints, inputs=[], outputs=[base_checkpoint_dropdown, checkpoint_dropdown])
        
        checkpoint_dropdown.change(show_checkpoint_info, inputs=[checkpoint_dropdown], outputs=[checkpoint_info_output])
        
        synthesize_btn.click(
            basic_tts,
            inputs=[checkpoint_dropdown, gen_text_input, remove_silence, cross_fade_duration_slider, nfe_slider, speed_slider, auto_adjust_speed],
            outputs=[audio_output, spectrogram_output, ref_text_input],
        )

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option("--share", "-s", default=False, is_flag=True, help="Share the app via Gradio")
def main(port, host, share):
    global app
    print("Starting app...")
    app.queue().launch(server_name=host, server_port=port, share=share)

if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()