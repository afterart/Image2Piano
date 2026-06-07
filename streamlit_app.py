import os
import io
import cv2
import warnings
import tempfile
import subprocess
import numpy as np
from PIL import Image
from gtts import gTTS
from datetime import datetime
from pydub import AudioSegment
import streamlit as st

warnings.filterwarnings("ignore")

# --- CORE CONFIGURATION ---
color_palette = {
    "Deep Red": {"rgb": (139, 0, 0), "freq": 138.59},
    "Red": {"rgb": (255, 0, 0), "freq": 155.56},
    "Orange": {"rgb": (255, 165, 0), "freq": 185.00},
    "Brown": {"rgb": (165, 42, 42), "freq": 207.65},
    "Olive": {"rgb": (128, 128, 0), "freq": 233.08},
    "Yellow": {"rgb": (255, 255, 0), "freq": 277.18},
    "Lime": {"rgb": (50, 205, 50), "freq": 311.13},
    "Green": {"rgb": (0, 128, 0), "freq": 369.99},
    "Cyan": {"rgb": (0, 255, 255), "freq": 415.30},
    "Sky Blue": {"rgb": (135, 206, 235), "freq": 466.16},
    "Blue": {"rgb": (0, 0, 255), "freq": 554.37},
    "Magenta": {"rgb": (255, 0, 255), "freq": 622.25}
}

def generate_grand_piano(frequency):
    sample_rate = 44100
    duration = 0.8
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 1.0 * np.sin(2 * np.pi * frequency * t) + 0.8 * np.sin(2 * np.pi * (frequency * 0.5) * t)
    envelope = np.exp(-3.0 * t)
    final_wave = wave * envelope
    audio_ints = (final_wave * 32767 / np.max(np.abs(final_wave))).astype(np.int16)
    return AudioSegment(audio_ints.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

# --- USER INTERFACE ---
st.title("🎹 Image-to-Piano Sequencer")
st.write("Upload 8 images to generate a synchronized audio-visual performance.")

uploaded_files = st.file_uploader("1. Upload 8 Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("2. Generate Performance", type="primary"):
    if not uploaded_files or len(uploaded_files) < 8:
        st.error("Error: Please upload at least 8 images.")
    else:
        with st.spinner("Processing visual assets and compiling audio..."):
            # Sort files systematically by name
            uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
            
            intro_audio = AudioSegment.empty()
            melody_unit = AudioSegment.empty()
            detected_colors = []

            for file_info in uploaded_files[:8]:
                # Streamlit uploaded files can be read directly by PIL
                img = Image.open(file_info).convert('RGB')
                avg_pixel = img.resize((1, 1)).getpixel((0, 0))
                color_name = min(color_palette.keys(), key=lambda x: sum((s - q) ** 2 for s, q in zip(color_palette[x]["rgb"], avg_pixel)))
                detected_colors.append(color_name)

                v_fp = io.BytesIO()
                gTTS(color_name).write_to_fp(v_fp)
                v_fp.seek(0)
                voice_seg = AudioSegment.from_file(v_fp, format="mp3")
                note_seg = generate_grand_piano(color_palette[color_name]["freq"])
                
                intro_audio += voice_seg + AudioSegment.silent(duration=150) + note_seg + AudioSegment.silent(duration=400)
                melody_unit += note_seg + AudioSegment.silent(duration=150)

            final_melody = melody_unit * 2
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime('%H%M%S')

            melody_p = os.path.join(temp_dir, f"mel_{timestamp}.wav")
            intro_p = os.path.join(temp_dir, f"int_{timestamp}.wav")
            final_melody.export(melody_p, format="wav")
            intro_audio.export(intro_p, format="wav")

            # --- VIDEO GENERATION ---
            video_p = os.path.join(temp_dir, f"vid_{timestamp}.mp4")
            temp_silent = os.path.join(temp_dir, f"silent_{timestamp}.mp4")

            out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*'mp4v'), 1.25, (1280, 720))
            for c in (detected_colors * 2):
                rgb = color_palette[c]["rgb"]
                frame = np.full((720, 1280, 3), (rgb[2], rgb[1], rgb[0]), dtype=np.uint8)
                cv2.putText(frame, c.upper(), (100, 360), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 4)
                out.write(frame)
            out.release()

            subprocess.run([
                'ffmpeg', '-y', '-i', temp_silent, '-i', melody_p, 
                '-c:v', 'libx264', '-c:a', 'aac', '-shortest', video_p
            ], capture_output=True)

            # Display outputs directly on screen
            st.success("✨ Generation Complete!")
            
            st.subheader("Intro Performance")
            st.audio(intro_p)
            
            st.subheader("Melody Audio Loop")
            st.audio(melody_p)
            
            st.subheader("3. Visual Performance (with Sound)")
            st.video(video_p)
