# Prerequisites

# g6e.12xlarge, Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4.1 (Ubuntu 22.04) 20241027


# Training/Inference

# git clone https://github.com/VITA-MLLM/VITA
# cd VITA
# conda create -n vita python=3.10 -y
# conda activate vita
# pip install --upgrade pip
# pip install -r requirements.txt
# pip install flash-attn --no-build-isolation

# sudo apt update
# sudo apt install -y git-lfs
# git lfs install
# git clone https://huggingface.co/VITA-MLLM/VITA-1.5

# CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
#     --model_path VITA-1.5 \
#     --image_path asset/vita_newlog.jpg \
#     --model_type qwen2p5_instruct \
#     --conv_mode qwen2p5_instruct \
#     --question "Describe this images."

# CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
#     --model_path VITA-1.5 \
#     --image_path asset/vita_newlog.jpg \
#     --model_type qwen2p5_instruct \
#     --conv_mode qwen2p5_instruct \
#     --audio_path asset/q1.wav

# CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
#     --model_path VITA-1.5 \
#     --image_path asset/vita_newlog.jpg \
#     --model_type qwen2p5_instruct \
#     --conv_mode qwen2p5_instruct \
#     --audio_path asset/q2.wav


# Demo

# # Fix pyaudio install problem
# sudo apt-get install libasound2-dev
# sudo apt-get install portaudio19-dev python3-pyaudio

# conda create -n vita_demo python==3.10
# conda activate vita_demo
# pip install -r web_demo/web_demo_requirements.txt
# cp -rL VITA-1.5/ demo_VITA_ckpt/
# mv demo_VITA_ckpt/config.json demo_VITA_ckpt/origin_config.json
# cp -rf web_demo/vllm_tools/qwen2p5_model_weight_file/* demo_VITA_ckpt/
# cp -rf web_demo/vllm_tools/vllm_file/* /opt/conda/envs/vita_demo/lib/python3.10/site-packages/vllm/model_executor/models/

# python -m web_demo.web_ability_demo demo_VITA_ckpt


#  Real-Time Interactive Demo

# mkdir -p web_demo/wakeup_and_vad/resources/
# wget https://raw.githubusercontent.com/snakers4/silero-vad/refs/tags/v4.0/files/silero_vad.onnx -O web_demo/wakeup_and_vad/resources/silero_vad.onnx
# wget https://raw.githubusercontent.com/snakers4/silero-vad/refs/tags/v4.0/files/silero_vad.jit -O web_demo/wakeup_and_vad/resources/silero_vad.jit

# For a better real-time interactive experience, you need to set max_dynamic_patch to 1 in demo_VITA_ckpt/config.json. When you run the basic demo, you can set it to the default value of 12 to enhance the model's visual capabilities.

# pip install flask==3.1.0 flask-socketio==5.5.0 cryptography==44.0.0 timm==1.0.12

python -m web_demo.server --model_path demo_VITA_ckpt --ip 0.0.0.0 --port 8081