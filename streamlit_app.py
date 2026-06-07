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

# --- CORE CONFIGURATION (Standardized Palette) ---
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

def rgb_to_lab(rgb):
    """Converts standard RGB coordinates to the perceptually uniform CIELAB space."""
    r, g, b = [x / 255.0 for x in rgb]
    
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
    
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883
    
    fx = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    fy = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    fz = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16 / 116)
    
    return ((116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz))

def extract_dominant_rgb(pil_img):
    """Exposes true dominant color using a quantized sampling matrix instead of global averaging."""
    thumb = pil_img.resize((50, 50), Image.Resampling.BILINEAR)
    pixels = np.array(thumb).reshape(-1, 3)
    
    quantized = (pixels // 32) * 32
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant_index = np.argmax(counts)
    
    matched_pixels = pixels[np.all(quantized == colors[dominant_index], axis=1)]
    return tuple(np.mean(matched_pixels, axis=0).astype(int))

# Convert the reference palette definitions to CIELAB values once during startup
lab_palette = {name: rgb_to_lab(data["rgb"]) for name, data in color_palette.items()}

# --- USER INTERFACE ---
st.title("🎹 Perceptual Image-to-Piano Sequencer")
st.write("Resolves color extraction skew using uniform CIELAB color-distance parameters.")

uploaded_files = st.file_uploader("1. Upload 8 Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("2. Generate Performance", type="primary"):
    if not uploaded_files or len(uploaded_files) < 8:
        st.error("Error: Please upload at least 8 images.")
    else:
        with st.spinner("Executing perceptual classification updates..."):
            uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
            
            intro_audio = AudioSegment.empty()
            melody_unit = AudioSegment.empty()
            pipeline_data = []

            for file_info in uploaded_files[:8]:
                img = Image.open(file_info).convert('RGB')
                
                # Extract real dominant color instead of a flat arithmetic mean blend
                dominant_rgb = extract_dominant_rgb(img)
                dominant_lab = rgb_to_lab(dominant_rgb)
                
                # Map distance via uniform CIELAB vectors to match human vision
                color_name = min(
                    color_palette.keys(),
                    key=lambda x: sum((a - b) ** 2 for a, b in zip(lab_palette[x], dominant_lab))
                )
                
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

            video_p = os.path.join(temp_dir, f"vid_{timestamp}.mp4")
            temp_silent = os.path.join(temp_dir, f"silent_{timestamp}.mp4")

            out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*'mp4v'), 1.25, (1280, 720))
            
            for data_node in (pipeline_data * 2):
                c_name = data_node["color_name"]
                rgb = data_node["rgb"]
                orig_img = data_node["image"]
                
                # Render clear background frame
                frame = np.full((720, 1280, 3), (rgb[2], rgb[1], rgb[0]), dtype=np.uint8)
                
                # Build localized 16:9 thumbnail picture-in-picture box
                inset_w, inset_h = 320, 180
                cv2_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
                inset_thumb = cv2.resize(cv2_img, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
                
                y_offset = 40
                x_offset = 1280 - inset_w - 40
                
                cv2.rectangle(
                    frame, 
                    (x_offset - 2, y_offset - 2), 
                    (x_offset + inset_w + 2, y_offset + inset_h + 2), 
                    (255, 255, 255), 
                    2
                )
                frame[y_offset:y_offset+inset_h, x_offset:x_offset+inset_w] = inset_thumb
                
                cv2.putText(frame, c_name.upper(), (100, 360), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 4)
                out.write(frame)
                
            out.release()

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
