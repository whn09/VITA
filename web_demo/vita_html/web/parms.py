import time
from collections import deque

from web_demo.vita_html.web.queue import PCMQueue, ThreadSafeQueue
from web_demo.wakeup_and_vad.wakeup_and_vad import WakeupAndVAD

class GlobalParams:
    def __init__(self):
        self.wakeup_and_vad = WakeupAndVAD("./web_demo/wakeup_and_vad/resources", cache_history=10)
        self.interrupt_signal = 1
        self.collected_images = deque(maxlen=8)  # 存储最近8帧图像
        self.last_image_time = time.time()  # 记录最后一帧图像的时间
        print("GlobalParams init")
        self.reset()

    def reset(self):
        self.stop_generate = False
        self.is_generate = False
        self.wakeup_and_vad.in_dialog = False
        self.whole_text = ""

        self.tts_over = False
        self.tts_over_time = 0
        self.tts_data = ThreadSafeQueue()
        self.pcm_fifo_queue = PCMQueue()

        self.stop_tts = False
        self.stop_pcm = False

        self.collected_images.clear()
        self.last_image_time = time.time()
    
    def interrupt(self):
        self.stop_generate = True
        self.tts_over = True
        while True:
            time.sleep(0.01)
            if not self.is_generate:
                self.stop_generate = False
                while True:
                    time.sleep(0.01)
                    if self.tts_data.is_empty():
                        self.whole_text = ""
                        self.tts_over = False
                        self.tts_over_time += 1
                        # 清空视频帧
                        self.collected_images.clear()
                        break
                break
    
    def release(self):
        # 清空视频帧
        self.collected_images.clear()
        pass

    def print(self):
        print("stop_generate:", self.stop_generate)
        print("is_generate:", self.is_generate)
        print("whole_text:", self.whole_text)
        print("tts_over:", self.tts_over)
        print("tts_over_time:", self.tts_over_time)

