import sounddevice as sd
import time
import queue
import numpy as np
import threading

import bot
from sylnet.lib import SylNet


# *************** Utils ***************

def run_daemon(callable):
    t = threading.Thread(None, callable)
    t.setDaemon(True)
    t.start()


# *************** Common Datatypes ***************

class States:
    IDLE = "IDLE"
    RUNNING = "RUNNING"

class ProgramState:

    def __init__(self) -> None:
        self.state = States.IDLE

    def is_running(self) -> bool:
        return state.state == States.RUNNING

    def is_idle() -> bool:
        return state.state == States.IDLE

class SpeechChunk:

    def __init__(self, data, duration_ms, sample_rate) -> None:
        self.data = data
        self.duration_ms = duration_ms
        self.sample_rate = sample_rate


# *************** Pipeline Stages ***************

def mk_audio_callback(state: ProgramState, frames_queue: queue.Queue):
    def audio_callback(indata, frames, time, status):
        if (state.state == States.RUNNING):
            frames_queue.put(indata[:,0])

    return audio_callback
    

def raw_audio_processor(raw_frames_queue: queue.Queue, speech_chunks_queue: queue.Queue, buffer_length_ms: int, window_length_ms: int, sample_rate: int):
    chunk_size = int(buffer_length_ms / 1000.0 * sample_rate)
    window_hop_size = int(window_length_ms / 1000.0 * sample_rate)
    buffer_size = chunk_size + (2 * window_hop_size)
    buffer = np.zeros(buffer_size)
    frame = None
    
    current_buffer_pos = 0

    
    while True:
        frame = raw_frames_queue.get()
        frame_size = len(frame)

        if frame_size == 0:
            print("[WARN] Frame is empty")
            continue

        if frame_size > window_hop_size:
            print(f"[WARN] Frame is too large! frame size: {frame_size}; hop window size: {window_hop_size}")

        if frame_size > buffer_size:
            print(f"[WARN] Frame is too large! frame size: {frame_size}; buffer size: {buffer_size}")
            continue
        
        buf_update_start = current_buffer_pos
        buf_update_end = current_buffer_pos + frame_size

        frame_update_start = 0
        frame_update_end = frame_size

        if buf_update_end >= chunk_size:
            to_take = chunk_size - current_buffer_pos
            buffer[current_buffer_pos : current_buffer_pos + to_take] = frame[:to_take]

            # send data for further processing
            chunk = SpeechChunk(data = buffer[:chunk_size], duration_ms=buffer_length_ms, sample_rate=sample_rate)
            speech_chunks_queue.put(chunk)

            buffer[0:chunk_size-window_hop_size] = buffer[window_hop_size:chunk_size]
            buf_update_start = chunk_size-window_hop_size
            buf_update_end = buf_update_start + frame_size - to_take

            frame_update_start = to_take
            frame_update_end = frame_size

        buffer[buf_update_start:buf_update_end] = frame[frame_update_start:frame_update_end]
        current_buffer_pos = buf_update_end

        

def count_syllables(sylnet: SylNet, audio, sampling_rate):
    return sylnet.run(audio, sampling_rate)

def notify_too_fast(bot_proxy: bot.BotProxy):
    try:
        bot_proxy.send_notification()
    except Exception as e:
        print("Failed to send notification")
        print(e)


def audio_chunks_processor(state: ProgramState, sylnet: SylNet, bot_proxy: bot.BotProxy, buffered_audio_queue: queue.Queue, target_speech_rate: float):
    """
    target_speech_rate - words per minute
    """

    target_syllables_per_seconds = target_speech_rate * 3.0 / 60
    print(f"target speech rate: {target_syllables_per_seconds} syl/sec")

    last_notification_sent_at = None
    notifications_interval_ms = 5000

    while True:
        try:
            speech_chunk: SpeechChunk = buffered_audio_queue.get()
            print("took audio chunk")

            syllables_per_second = count_syllables(sylnet, speech_chunk.data, speech_chunk.sample_rate) / (speech_chunk.duration_ms / 1000)

            print(f"syl/sec: {syllables_per_second}")

            if syllables_per_second > target_syllables_per_seconds:
                print(f"speech rate is too large: {syllables_per_second}")

                now = time.time_ns() // 1_000_000 
                if state.is_running() and (last_notification_sent_at is None or (now - last_notification_sent_at >= notifications_interval_ms)):
                    last_notification_sent_at = now
                    notify_too_fast(bot_proxy)
        except Exception as e:
            print(e)


# *************** Runloop ***************


def run_main_loop(state: ProgramState):
    input_text = None

    try:
        while input_text != 'finish':
            input_text = input("What should I do next? [start, stop, finish]\n").strip()

            if len(input_text) == 0:
                continue

            if input_text == 'start':
                state.state = States.RUNNING
            elif input_text == 'stop' or input_text == 'finish':
                state.state = States.IDLE
            else:
                print(f'Unknown command: {input_text}')

            
    except KeyboardInterrupt:
        print("Finished")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    try: 
        CHANNELS = 1
        SAMPLE_RATE = 22050
        USERNAME = "goodmove"
        sylnet_impl = SylNet()
        sylnet_impl.init()

        buffer_length_ms = 5000
        window_length_ms = 1000
        target_speech_rate = 100 # words per minute

        state = ProgramState()
        bot_proxy = bot.init_bot()
        raw_audio_frames_queue = queue.Queue()
        speech_chunks_queue = queue.Queue()

        bot_proxy.set_target_user(USERNAME)

        stream = sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, callback=mk_audio_callback(state, raw_audio_frames_queue))

        with stream:
            run_daemon(lambda: raw_audio_processor(raw_audio_frames_queue, speech_chunks_queue, buffer_length_ms, window_length_ms, SAMPLE_RATE))
            run_daemon(lambda: audio_chunks_processor(state, sylnet_impl, bot_proxy, speech_chunks_queue, target_speech_rate))
        
            run_main_loop(state)

        print("stopping bot...")
        bot_proxy._bot.stop_bot()
        

    except KeyboardInterrupt:
        print("Finished")
    except Exception as e:
        print(e)