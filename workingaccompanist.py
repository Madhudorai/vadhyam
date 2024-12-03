import numpy as np
import sounddevice as sd
import threading
import queue
import time
import ddsp
import ddsp.training
import tensorflow as tf
ddsp.spectral_ops.reset_crepe()

# Audio parameters
SAMPLE_RATE = 16000  # Sample rate for both input and output
CHANNELS = 1         # Mono audio
FRAME_SIZE = 1024    # Size of each audio frame
SILENCE_THRESHOLD = 0.001  # Threshold for silence detection (adjusted for RMS)
SILENCE_DURATION = 0.2      # Duration in seconds to trigger processing

# Buffer parameters
BUFFER_DURATION = 3.0  # Maximum duration of audio to buffer
MAX_BUFFER_SIZE = int(BUFFER_DURATION * SAMPLE_RATE)
MIN_BUFFER_SIZE = int(0.5 * SAMPLE_RATE)  # Minimum required buffer size (e.g., 0.5 seconds)

# CREPE parameters
CREPE_FRAME_RATE = 31.25  # Frame rate for CREPE pitch estimation

# DDSP parameters
HOP_SIZE = 64

# Initialize buffers and variables
audio_buffer = np.zeros(0, dtype=np.float32)  # Start with an empty buffer
processing_queue = queue.Queue()
silence_duration_counter = 0.0
is_buffering = False  # Flag to indicate if we are buffering audio
buffer_lock = threading.Lock()  # Lock for thread-safe buffer access
playback_stop_event = threading.Event()
is_playing = False
playback_audio_data = None
playback_audio_position = 0

PLAYBACK_SILENCE_THRESHOLD = SILENCE_THRESHOLD * 90  # Adjust as needed

def is_silence(audio_chunk, threshold=SILENCE_THRESHOLD):
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms < threshold

def audio_callback(indata, frames, time_info, status):
    global audio_buffer, silence_duration_counter, is_buffering, is_playing
    audio_chunk = indata[:, 0]

    # Adjust threshold during playback
    current_threshold = SILENCE_THRESHOLD
    if is_playing:
        current_threshold = PLAYBACK_SILENCE_THRESHOLD

    # Check if the current chunk is silent
    chunk_is_silent = is_silence(audio_chunk, threshold=current_threshold)

    if is_playing:
        if not chunk_is_silent:
            # Sound detected during playback
            print("Sound detected during playback. Stopping playback.")
            playback_stop_event.set()
            # Start buffering new audio
            is_buffering = True
            silence_duration_counter = 0.0
            with buffer_lock:
                audio_buffer = audio_chunk.copy()
                return 

    if is_buffering:
        if chunk_is_silent:
            # Increment silence duration counter
            silence_duration_counter += len(audio_chunk) / SAMPLE_RATE
            if silence_duration_counter >= SILENCE_DURATION:
                # Silence has persisted, stop buffering and process the buffer
                is_buffering = False
                silence_duration_counter = 0.0
                with buffer_lock:
                    buffer_length = len(audio_buffer)
                    if buffer_length >= MIN_BUFFER_SIZE:
                        print("Silence detected. Processing audio.")
                        # Send the buffered audio for processing
                        processing_queue.put(audio_buffer.copy())
                    else:
                        print("Buffer too short. Skipping processing.")
                    # Clear the buffer
                    audio_buffer = np.zeros(0, dtype=np.float32)
        else:
            # Reset silence duration counter if sound is detected
            silence_duration_counter = 0.0
            # Continue buffering audio
            with buffer_lock:
                audio_buffer = np.concatenate((audio_buffer, audio_chunk))
                if len(audio_buffer) > MAX_BUFFER_SIZE:
                    # Keep only the last MAX_BUFFER_SIZE samples
                    audio_buffer = audio_buffer[-MAX_BUFFER_SIZE:]
    else:
        if not chunk_is_silent:
            # Sound detected after silence, start buffering
            is_buffering = True
            silence_duration_counter = 0.0
            # Start a new buffer with the current chunk
            with buffer_lock:
                audio_buffer = audio_chunk.copy()
            print("Sound detected. Starting new buffer.")

def playback_callback(outdata, frames, time_info, status):
    global playback_audio_data, playback_audio_position, playback_stop_event
    if playback_stop_event.is_set():
        outdata.fill(0)
        raise sd.CallbackAbort()
    if playback_audio_data is None:
        # No audio data to play
        outdata.fill(0)
        raise sd.CallbackStop()
    # Determine how many frames to output
    chunk = playback_audio_data[playback_audio_position:playback_audio_position+frames]
    if len(chunk) < frames:
        # End of audio data
        outdata[:len(chunk), 0] = chunk
        outdata[len(chunk):, 0] = 0
        raise sd.CallbackStop()
    else:
        outdata[:, 0] = chunk
        playback_audio_position += frames

def processing_thread():
    global playback_stop_event, is_playing
    while True:
        audio_data = processing_queue.get()
        if audio_data is None:
            break  # Exit signal received
        try:
            # Process the audio_data
            synthesized_audio = process_audio(audio_data)
            if synthesized_audio is not None:
                # Play back the synthesized audio using OutputStream
                print("Playing back synthesized audio.")
                playback_stop_event.clear()
                is_playing = True
                global playback_audio_data, playback_audio_position
                playback_audio_data = synthesized_audio.copy()
                playback_audio_position = 0
                playback_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    callback=playback_callback
                )
                playback_stream.start()
                # Wait until playback is finished or stopped
                while playback_stream.active:
                    if playback_stop_event.is_set():
                        playback_stream.abort()
                        break
                    time.sleep(0.01)
                playback_stream.close()
                is_playing = False
            else:
                print("Synthesized audio is empty. Skipping playback.")
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            # Clear the buffer after processing
            with buffer_lock:
                global audio_buffer
                audio_buffer = np.zeros(0, dtype=np.float32)
            print("Processing done. Buffer cleared.")

def process_audio(audio_data):
    if len(audio_data) < MIN_BUFFER_SIZE:
        print("Audio data too short to process.")
        return None

    # Add batch dimension
    audio = audio_data[np.newaxis, :]
    # Ensure dimensions and sampling rates are equal
    n_samples = audio.shape[1]
    hop_size = HOP_SIZE
    time_steps = n_samples // hop_size
    n_samples = time_steps * hop_size
    audio = audio[:, :n_samples]

    if n_samples == 0 or time_steps == 0:
        print("Not enough samples after trimming. Skipping processing.")
        return None

    # Predict Pitch using CREPE with Viterbi decoding
    f0_crepe, f0_confidence = ddsp.spectral_ops.compute_f0(
        audio[0], frame_rate=CREPE_FRAME_RATE, viterbi=True
    )

    # Ensure f0_crepe has the correct length
    f0_crepe = f0_crepe[:time_steps]

    # Check for NaNs or invalid values in f0_crepe
    if np.isnan(f0_crepe).any():
        print("Invalid f0 values detected. Skipping processing.")
        return None

    # Synthesize the CREPE audio
    try:
        synth = ddsp.synths.Wavetable(n_samples=n_samples, scale_fn=None)
        wavetable = np.sin(np.linspace(0, 2.0 * np.pi, 2048))[np.newaxis, np.newaxis, :]
        amps = np.ones([1, time_steps, 1]) * 0.1

        audio_crepe = synth(
            amps, wavetable, f0_crepe[np.newaxis, :, np.newaxis]
        )
        # Squeeze the synthesized audio to 1D array for playback
        synthesized_audio = np.squeeze(audio_crepe)
        return synthesized_audio
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return None

# Start the processing thread
thread = threading.Thread(target=processing_thread)
thread.start()

# Open the audio input stream
with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE, callback=audio_callback):
    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping...")
        processing_queue.put(None)  # Signal the processing thread to exit
        thread.join()
