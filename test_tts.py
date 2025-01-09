import os
import torch
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import time

def load_model_embemding(model_path):
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    config_path = os.path.join(model_path, 'origin_config.json')
    config = VITAQwen2Config.from_pretrained(config_path)
    model = VITAQwen2ForCausalLM.from_pretrained(model_path, config=config, low_cpu_mem_usage=True)
    embedding = model.get_input_embeddings()
    del model
    return embedding

def split_into_sentences(text):
    return [s.strip() for s in text.split('。') if s.strip()]

def float_to_int16(audio):
    audio = np.clip(audio * 32768, -32768, 32767)
    return audio.astype(np.int16)

def generate_and_save_audio(tts, text, output_path, tokenizer, llm_embedding, device):
    decoder_topk = 32
    codec_chunk_size = 1024
    codec_padding_size = 1024
    
    start_time = time.time()
    all_audio_chunks = []
    
    for idx, sentence in enumerate(split_into_sentences(text)):
        if not sentence:
            continue
            
        input_ids = torch.tensor(tokenizer.encode(sentence)).to(device)
        embeddings = llm_embedding(input_ids)
        embeddings = embeddings.reshape(-1, 896).unsqueeze(0)
        
        for seg in tts.run(embeddings, decoder_topk, None, codec_chunk_size, codec_padding_size):
            if idx == 0:
                try:
                    split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                    seg = seg[:, :, split_idx:]
                except:
                    print('Do not need to split')
                    pass
            
            if seg is not None and len(seg) > 0:
                # 转换为CPU上的numpy数组
                seg = seg.to(torch.float32).cpu().numpy()
                
                # 确保音频是2维的 (samples, channels)
                seg = seg.squeeze()  # 移除所有大小为1的维度
                if len(seg.shape) == 1:
                    seg = seg[:, np.newaxis]  # 添加channel维度
                elif len(seg.shape) > 2:
                    seg = seg.reshape(-1, 1)  # 重塑为(samples, 1)
                
                # 转换为int16并添加到列表
                all_audio_chunks.append(float_to_int16(seg))
                
    # 合并所有音频片段
    if all_audio_chunks:
        final_audio = np.concatenate(all_audio_chunks, axis=0)
        
        # 确保最终音频是正确的形状
        if len(final_audio.shape) == 1:
            final_audio = final_audio[:, np.newaxis]
        
        # 打印音频信息
        print(f"Final audio shape before saving: {final_audio.shape}")
        print(f"Final audio dtype: {final_audio.dtype}")
        print(f"Final audio min: {final_audio.min()}, max: {final_audio.max()}")
        
        # 保存音频文件
        sf.write(output_path, final_audio, samplerate=24000)
        print(f"Audio saved to {output_path}")
        
        end_time = time.time()
        print('Total TTS time:', end_time - start_time)

if __name__ == "__main__":
    cuda_devices = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    print(f"Setting CUDA_VISIBLE_DEVICES to {cuda_devices}")
        
    print(f"Process tts_worker CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Process tts_worker available devices: {torch.cuda.device_count()}")
    print(f"Process tts_worker current device: {torch.cuda.current_device()}")
    print(f"Process tts_worker device name: {torch.cuda.get_device_name(0)}")
    
    model_path = "demo_VITA_ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_embedding = load_model_embemding(model_path).to(device)
    print('load_model_embemding done')
    
    from vita.model.vita_tts.decoder.llm2tts import llm2TTS
    tts = llm2TTS(os.path.join(model_path, 'vita_tts_ckpt/'))
    print('llm2TTS done')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 测试TTS生成
    test_text = "喂，你好！歡迎致電安樂貸款公司，我係Amy，請問有咩可以幫到你？"
    output_path = "output.wav"
    
    # 生成并保存音频
    generate_and_save_audio(tts, test_text, output_path, tokenizer, llm_embedding, device)