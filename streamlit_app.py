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
# Standardized palette configuration
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

def get_perceptual_distance(rgb1, rgb2):
    """Calculates weighted human-perceptual distance between two RGB vectors."""
    r_diff = rgb1[0] - rgb2[0]
    g_diff = rgb1[1] - rgb2[1]
    b_diff = rgb1[2] - rgb2[2]
    return (2 * r_diff**2) + (4 * g_diff**2) + (3 * b_diff**2)

# --- USER INTERFACE ---
st.title("🎹 Image-to-Piano Sequencer")
st.write("Upload 8 images to generate a synchronized audio-visual performance with inset tracking.")

uploaded_files = st.file_uploader("1. Upload 8 Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("2. Generate Performance", type="primary"):
    if not uploaded_files or len(uploaded_files) < 8:
        st.error("Error: Please upload at least 8 images.")
    else:
        with st.spinner("Processing visual assets and compiling audio..."):
            uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
            
            intro_audio = AudioSegment.empty()
            melody_unit = AudioSegment.empty()
            
            # Tracking list to preserve original images and their assigned colors
            pipeline_data = []

            for file_info in uploaded_files[:8]:
                img = Image.open(file_info).convert('RGB')
                avg_pixel = img.resize((1, 1)).getpixel((0, 0))
                
                # Correction: Perceptual color matching instead of naive Euclidean distance
                color_name = min(
                    color_palette.keys(), 
                    key=lambda x: get_perceptual_distance(color_palette[x]["rgb"], avg_pixel)
                )
                
                # Retain data matrix for video compilation loop
                pipeline_data.append({
                    "image": img,
                    "color_name": color_name,
                    "rgb": color_palette[color_name]["rgb"]
                })

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

            # --- VIDEO GENERATION WITH INSET WINDOWS ---
            video_p = os.path.join(temp_dir, f"vid_{timestamp}.mp4")
            temp_silent = os.path.join(temp_dir, f"silent_{timestamp}.mp4")

            # 1.25 frames per second matches the 0.8 second tone duration exactly
            out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*'mp4v'), 1.25, (1280, 720))
            
            # Repeat sequence twice to match final loops
            for data_node in (pipeline_data * 2):
                c_name = data_node["color_name"]
                rgb = data_node["rgb"]
                orig_img = data_node["image"]
                
                # Render the base background color frame (OpenCV uses BGR format)
                frame = np.full((720, 1280, 3), (rgb[2], rgb[1], rgb[0]), dtype=np.uint8)
                
                # Format original source asset as a 16:9 box inset thumbnail
                inset_w, inset_h = 320, 180
                cv2_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
                inset_thumb = cv2.resize(cv2_img, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
                
                # Define placement coordinates (Top-Right corner with 40px padding offset)
                y_offset = 40
                x_offset = 1280 - inset_w - 40
                
                # Add white border structure around the asset thumbnail
                cv2.rectangle(
                    frame, 
                    (x_offset - 2, y_offset - 2), 
                    (x_offset + inset_w + 2, y_offset + inset_h + 2), 
                    (255, 255, 255), 
                    2
                )
                
                # Inject the source image asset matrix directly into the background matrix frame
                frame[y_offset:y_offset+inset_h, x_offset:x_offset+inset_w] = inset_thumb
                
                # Draw text identifier overlay
                cv2.putText(frame, c_name.upper(), (100, 360), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 4)
                out.write(frame)
                
            out.release()

            # Compile audio and video streams via server binary
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_silent, '-i', melody_p, 
                '-c:v', 'libx264', '-c:a', 'aac', '-shortest', video_p
            ], capture_output=True)

            st.success("✨ Generation Complete!")
            
            st.subheader("Intro Performance")
            st.audio(intro_p)
            
            st.subheader("Melody Audio Loop")
            st.audio(melody_p)
            
            st.subheader("3. Visual Performance (with Sound)")
            st.video(video_p)
