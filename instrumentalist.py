import numpy as np
import sounddevice as sd
import threading
import queue
import time
import os
import tempfile
from pathlib import Path
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model
import pretty_midi
import soundfile as sf
import pygame.midi
import pygame.mixer

# Audio parameters
SAMPLE_RATE = 16000  # Sample rate for both input and output
CHANNELS = 1         # Mono audio
FRAME_SIZE = 1024    # Size of each audio frame
SILENCE_THRESHOLD = 0.001  # Threshold for silence detection (adjusted for RMS)
SILENCE_DURATION = 0.2      # Duration in seconds to trigger processing

# Buffer parameters
BUFFER_DURATION = 5.0  # Maximum duration of audio to buffer
MAX_BUFFER_SIZE = int(BUFFER_DURATION * SAMPLE_RATE)
MIN_BUFFER_SIZE = int(0.5 * SAMPLE_RATE)  # Minimum required buffer size (e.g., 0.5 seconds)

# Saxophone soundfont path - adjust this to your soundfont location
SOUNDFONT_PATH = "tenor_exp.sf2"  # Replace with your actual soundfont path

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

# Create a temporary directory for storing intermediate files
temp_dir = tempfile.mkdtemp()
#print(f"Using temporary directory: {temp_dir}")

# Load Basic Pitch model at startup
try:
    basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
    print("Basic Pitch model loaded successfully")
except Exception as e:
    print(f"Error loading Basic Pitch model: {e}")
    basic_pitch_model = None

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
            # print("Sound detected during playback. Stopping playback.")
            print("Listening...")
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
                        #print("Silence detected. Processing audio.")
                        print("Processing...")
                        # Send the buffered audio for processing
                        processing_queue.put(audio_buffer.copy())
                    else:
                        #print("Buffer too short. Skipping processing.")
                        print("Please sing for longer!")
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
            #print("Sound detected. Starting new buffer.")
            print("Listening...")

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

def sonify_midi_with_fluidsynth(midi_data, output_path):
    """
    Sonify MIDI data using FluidSynth through pretty_midi
    and return the resulting audio as numpy array
    """
    # Save the MIDI file temporarily
    midi_path = os.path.join(temp_dir, "temp.mid")
    midi_data.write(midi_path)
    
    # Use pretty_midi to synthesize with the soundfont
    try:
        # Configure the synthesizer
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        for instrument in pm.instruments:
            instrument.program = 0  # Preset number
            instrument.bank = 0       # Bank number


        audio_data = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=SOUNDFONT_PATH)
        
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        
        # Save to WAV file
        #sf.write(output_path, audio_data, SAMPLE_RATE)
        
        #print(f"Saved sonified MIDI to {output_path}")
        return audio_data
    except Exception as e:
        #print(f"Error in sonification: {e}")
        # Fallback: simple sine wave synthesis
        return simple_sine_synthesis(midi_data)

def simple_sine_synthesis(midi_data, duration=3.0):
    """
    Simple fallback synthesis in case FluidSynth fails
    """
    print("Woops! Instrumentalist is resting. Try again.") #Using fallback sine wave synthesis")
    sample_count = int(duration * SAMPLE_RATE)
    audio = np.zeros(sample_count)
    
    # Get all notes from the midi
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            start_idx = int(note.start * SAMPLE_RATE)
            end_idx = min(int(note.end * SAMPLE_RATE), sample_count)
            t = np.arange(start_idx, end_idx)
            sine_wave = 0.1 * np.sin(2.0 * np.pi * freq * t / SAMPLE_RATE)
            audio[start_idx:end_idx] += sine_wave
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    return audio

def processing_thread():
    global playback_stop_event, is_playing, basic_pitch_model
    
    # Create subdirectories in temp_dir
    audio_dir = os.path.join(temp_dir, "audio")
    midi_dir = os.path.join(temp_dir, "midi")
    wav_dir = os.path.join(temp_dir, "wav")
    
    for directory in [audio_dir, midi_dir, wav_dir]:
        os.makedirs(directory, exist_ok=True)
    
    file_counter = 0
    
    while True:
        audio_data = processing_queue.get()
        if audio_data is None:
            break  # Exit signal received
        
        try:
            # Save the audio buffer to a temporary WAV file
            file_counter += 1
            audio_path = os.path.join(audio_dir, f"singing_{file_counter}.wav")
            sf.write(audio_path, audio_data, SAMPLE_RATE)
            
            #print(f"Processing audio file: {audio_path}")
            
            # Use Basic Pitch to predict MIDI
            '''
            audio_path: Union[pathlib.Path, str],
            model_or_model_path: Union[Model, pathlib.Path, str] = ICASSP_2022_MODEL_PATH,
            onset_threshold: float = 0.5,
            frame_threshold: float = 0.3,
            minimum_note_length: float = 127.70,
            minimum_frequency: Optional[float] = None,
            maximum_frequency: Optional[float] = None,
            multiple_pitch_bends: bool = False,
            melodia_trick: bool = True,
            debug_file: Optional[pathlib.Path] = None,
            midi_tempo: float = 120,
            '''
            if basic_pitch_model is not None:
                model_output, midi_data, note_events = predict(
                    audio_path, 
                    basic_pitch_model,
                    onset_threshold=0.7,
                    frame_threshold=0.5,
                    minimum_note_length=127,
                    multiple_pitch_bends=True
                )
                
                # Save MIDI file
                #midi_path = os.path.join(midi_dir, f"singing_{file_counter}.mid")
                #midi_data.write(midi_path)
                #print(f"Saved MIDI to {midi_path}")
                
                # Sonify MIDI with saxophone soundfont
                wav_path = os.path.join(wav_dir, f"singing_{file_counter}.wav")
                synthesized_audio = sonify_midi_with_fluidsynth(midi_data, wav_path)
                
                if synthesized_audio is not None:
                    # Play back the synthesized audio
                    #print("Playing back synthesized audio.")
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
                    #print("Synthesized audio is empty. Skipping playback.")
                    print(" ")
            else:
                print("Basic Pitch model not loaded. Cannot process audio.")
                
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            # Clear the buffer after processing
            with buffer_lock:
                global audio_buffer
                audio_buffer = np.zeros(0, dtype=np.float32)
            #print("Processing done. Buffer cleared.")

def initialize_system():
    """Initialize all necessary components"""
    # Initialize pygame for MIDI
    pygame.midi.init()
    pygame.mixer.init(frequency=SAMPLE_RATE)
    
    # Check if soundfont exists
    # if not os.path.exists(SOUNDFONT_PATH):
    #     #print(f"Warning: Soundfont {SOUNDFONT_PATH} not found.")
    #     #print("Please download a soundfont and update SOUNDFONT_PATH.")
    # else:
    #     print(f"Found soundfont at {SOUNDFONT_PATH}")

def cleanup():
    """Cleanup resources before exiting"""
    pygame.midi.quit()
    pygame.mixer.quit()
    
    # Cleanup temporary files if needed
    # Uncomment the next line to remove temp files automatically
    # import shutil; shutil.rmtree(temp_dir)
    # print(f"Temporary files are stored in {temp_dir}")
    # print("You can delete this directory manually if you don't need the files.")

def main():
    # Initialize system
    initialize_system()
    
    try:
        # Start the processing thread
        thread = threading.Thread(target=processing_thread)
        thread.start()

        # Open the audio input stream
        with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE, callback=audio_callback):
            # print("\n==== Real-time Singing to MIDI Converter ====")
            # print("Recording... Sing into your microphone!")
            # print("When you stop singing (silence detected), your audio will be")
            # print("converted to MIDI and played back with a saxophone sound.")
            # print("Press Ctrl+C to stop the program.")
            # print("=================================================\n")
            print("\n==== Welcome to Voice 2 Instrument ====")
            print("Sing into your microphone!")
            print("When you stop singing, your voice will be")
            print("converted to a saxophone!")
            print("Press Ctrl+C to stop the program.")
            print("=================================================\n")            
            try:
                while True:
                    time.sleep(0.1)  # Keep the main thread alive
            except KeyboardInterrupt:
                print("\nStopping...")
                playback_stop_event.set()
                processing_queue.put(None)  # Signal the processing thread to exit
                thread.join()
    finally:
        cleanup()

if __name__ == "__main__":
    main()