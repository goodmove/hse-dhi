import sounddevice as sd
import time
import queue
import numpy as np

CHANNELS = 1
sampling_rate = 22050
block_time = 1 # seconds
block_size = block_time * sampling_rate


def mk_audio_callback(frames_queue):
    def audio_callback(indata, frames, time, status):
        frames_queue.put(indata[:,0])

    return audio_callback
    

def raw_audio_processor(raw_frames_queue: queue.Queue, buffered_audio_queue: queue.Queue, buffer_length_ms: int, window_length_ms: int, sample_rate: int):
    # insert incoming frames into cyclic buffer

    window_size = int(buffer_length_ms / 1000.0 * sample_rate)
    window_hop_size = int(window_length_ms / 1000.0 * sample_rate)
    buffer_size = int((buffer_length_ms + window_length_ms) / 1000.0 * sample_rate)
    buffer = np.zeros(buffer_size)
    frame = None
    
    current_buffer_pos = 0
    buffer_edge_pos = window_size

    
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

        if buf_update_end >= buffer_edge_pos:
            to_take = buffer_edge_pos - current_buffer_pos
            buffer[current_buffer_pos : current_buffer_pos + to_take] = frame[:to_take]

            # send data for further processing
            buffered_audio_queue.put(buffer[0:window_size])

            buffer[0:window_size-window_hop_size] = buffer[window_hop_size:window_size]
            buf_update_start = window_hop_size
            buf_update_end = buf_update_start + frame_size - to_take

            frame_update_start = to_take
            frame_update_end = frame_size

        buffer[buf_update_start:buf_update_end] = frame[frame_update_start:frame_update_end]
        current_buffer_pos = buf_update_end % buffer_edge_pos

        

def count_syllables(audio, sampling_rate):
    # FIXME: add implementation
    return 1

def notify_too_fast():
    # FIXME: add implementation
    pass

def audio_chunks_processor(buffered_audio_queue: queue.Queue, target_speech_rate: float):
    """
    target_speech_rate - words per minute
    """

    target_syllables_per_seconds = target_speech_rate * 4.0 / 60
    print(f"target speech rate: {target_syllables_per_seconds} syl/sec")

    last_notification_sent_at = None
    notifications_interval_ms = 5000

    while True:
        audio_block = buffered_audio_queue.get()
        sampling_rate = 1 # FIXME
        audio_length = 5 # seconds FIXME

        syllables_count = count_syllables(audio_block, sampling_rate) / audio_length

        if syllables_count > target_speech_rate:
            print(f"speech rate is too large: {syllables_count}")

            now = time.time_ns() // 1_000_000 
            if last_notification_sent_at is None or (now - last_notification_sent_at >= notifications_interval_ms):
                last_notification_sent_at = now
                notify_too_fast()





try: 
    raw_audio_frames_queue = queue.Queue()
    stream = sd.InputStream(channels=CHANNELS, samplerate=sampling_rate, callback=mk_audio_callback(raw_audio_frames_queue))

    with stream:
        time.sleep(10000)
except KeyboardInterrupt:
    print("Finished")
except Exception as e:
    print(e)