import os
import torch
import multiprocessing
from multiprocessing import Process

from vllm import LLM

# from web_demo.server import tts_worker, load_model

from flask import Flask
from flask_socketio import SocketIO

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
socketio = SocketIO(app)


def worker1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"#1 Process on GPU: {torch.cuda.device_count()} visible GPUs, current device is {torch.cuda.current_device()}")
    
    from vita.model.vita_tts.decoder.llm2tts import llm2TTS
    model_path = 'demo_VITA_ckpt'
    tts = llm2TTS(os.path.join(model_path, 'vita_tts_ckpt/'))
    print(f"#2 Process on GPU: {torch.cuda.device_count()} visible GPUs, current device is {torch.cuda.current_device()}")

def worker(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Process on GPU {gpu_id}: {torch.cuda.device_count()} visible GPUs, current device is {torch.cuda.current_device()}")
    
    engine_args = 'demo_VITA_ckpt'
    llm = LLM(
            model=engine_args,
            dtype="float16",
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,  # default: 0.85
            disable_custom_all_reduce=True,
            limit_mm_per_prompt={'image':256,'audio':50}
        )

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    manager = multiprocessing.Manager()
    request_inputs_queue = manager.Queue() 
    tts_inputs_queue = manager.Queue() 
    tts_output_queue = manager.Queue() 

    worker_1_stop_event = manager.Event() 
    worker_2_stop_event = manager.Event() 

    worker_1_start_event = manager.Event() 
    worker_2_start_event = manager.Event()
    worker_1_start_event.set()

    worker_1_2_start_event_lock = manager.Lock()

    llm_worker_1_ready = manager.Event()
    llm_worker_2_ready = manager.Event()

    tts_worker_ready = manager.Event()
    gradio_worker_ready = manager.Event()

    global_history = manager.list()
    global_history_limit = 1
    
    processes = []
    
    p = Process(target=worker1)
    # p = Process(target=tts_worker, args=('demo_VITA_ckpt', tts_inputs_queue, tts_output_queue, tts_worker_ready, [llm_worker_1_ready, llm_worker_2_ready]))
    p.start()
    processes.append(p)
    
    # p = Process(target=load_model, kwargs={
    #         "llm_id": 1,
    #         "engine_args": 'demo_VITA_ckpt', 
    #         "cuda_devices": "1",  # default: "0"
    #         "inputs_queue": request_inputs_queue,
    #         "outputs_queue": tts_inputs_queue,
    #         "tts_outputs_queue": tts_output_queue,
    #         "start_event": worker_1_start_event,
    #         "other_start_event": worker_2_start_event,
    #         "start_event_lock": worker_1_2_start_event_lock,
    #         "stop_event": worker_1_stop_event,
    #         "other_stop_event": worker_2_stop_event,
    #         "worker_ready": llm_worker_1_ready,
    #         "wait_workers_ready": [llm_worker_2_ready, tts_worker_ready], 
    #         "global_history": global_history,
    #         "global_history_limit": global_history_limit,
    #     })
    # p.start()
    # processes.append(p)
    
    # p = Process(target=load_model, kwargs={
    #         "llm_id": 2,
    #         "engine_args": 'demo_VITA_ckpt',
    #         "cuda_devices": "2",  # default: "1"
    #         "inputs_queue": request_inputs_queue,
    #         "outputs_queue": tts_inputs_queue,
    #         "tts_outputs_queue": tts_output_queue,
    #         "start_event": worker_2_start_event,
    #         "other_start_event": worker_1_start_event,
    #         "start_event_lock": worker_1_2_start_event_lock,
    #         "stop_event": worker_2_stop_event,
    #         "other_stop_event": worker_1_stop_event,
    #         "worker_ready": llm_worker_2_ready,
    #         "wait_workers_ready": [llm_worker_1_ready, tts_worker_ready], 
    #         "global_history": global_history,
    #         "global_history_limit": global_history_limit,
    #     })
    # p.start()
    # processes.append(p)
    
    for i in range(4):  # 启动4个进程
        # p = Process(target=worker, args=(i,))
        p = Process(target=worker, kwargs={"gpu_id": i})
        p.start()
        processes.append(p)

    for p in processes:
        p.join()