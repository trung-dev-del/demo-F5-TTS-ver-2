f5-tts_infer-cli \
--model "F5TTS_Base" \
--ref_audio trung_voice.wav \
--ref_text "Xin chào, hôm nay là một ngày tốt lành. Tôi được làm việc trong môi trường tốt, nơi mà đồng nghiệp tốt, nhân viên ổn, lãnh đạo trainning khá tốt. và Hà Nội đón mùa xuân đầu tiên của năm 2025, mùa xuân đẹp nhưng nồm quá. " \
--gen_text "nay là ngày mưa rét tiếp theo ở hà nội" \
--speed 1.0 \
--vocoder_name vocos \
--vocab_file data/your_training_dataset/vocab.txt \
--ckpt_file ckpts/your_training_dataset/model_350000.safetensors \