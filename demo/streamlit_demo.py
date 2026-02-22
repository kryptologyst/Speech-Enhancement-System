"""Streamlit demo for speech enhancement."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torchaudio
from src.models.conv_tasnet import ConvTasNet
from src.models.spectral_methods import SpectralSubtraction, WienerFilter
from src.utils.audio import add_noise, generate_noise, generate_synthetic_speech
from src.utils.device import get_device, move_to_device
from src.metrics.enhancement import EnhancementMetrics

# Page config
st.set_page_config(
    page_title="Speech Enhancement Demo",
    page_icon="🎤",
    layout="wide",
)

# Title and description
st.title("🎤 Speech Enhancement System")
st.markdown("""
This is a research demonstration of speech enhancement techniques. 
**IMPORTANT**: This is for research and educational purposes only. 
Do not use for biometric identification or voice cloning in production.
""")

# Sidebar
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Enhancement Method",
    ["Conv-TasNet", "Spectral Subtraction", "Wiener Filter"],
    help="Choose the speech enhancement method"
)

# Audio parameters
st.sidebar.subheader("Audio Parameters")
sample_rate = st.sidebar.slider("Sample Rate", 8000, 48000, 16000)
duration = st.sidebar.slider("Duration (seconds)", 1.0, 10.0, 4.0)

# Noise parameters
st.sidebar.subheader("Noise Parameters")
noise_type = st.sidebar.selectbox("Noise Type", ["white", "pink", "brown"])
snr_db = st.sidebar.slider("SNR (dB)", -10, 30, 10)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Input Audio")
    
    # Audio input options
    input_option = st.radio(
        "Choose input method",
        ["Upload audio file", "Record audio", "Generate synthetic speech"]
    )
    
    if input_option == "Upload audio file":
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "flac", "m4a"],
            help="Upload an audio file to enhance"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Load audio
            try:
                waveform, sr = torchaudio.load(tmp_path)
                waveform = waveform.mean(dim=0)  # Convert to mono
                
                # Resample if needed
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = resampler(waveform)
                
                # Trim to duration
                max_samples = int(duration * sample_rate)
                if len(waveform) > max_samples:
                    waveform = waveform[:max_samples]
                elif len(waveform) < max_samples:
                    # Pad with zeros
                    padding = max_samples - len(waveform)
                    waveform = torch.cat([waveform, torch.zeros(padding)])
                
                clean_audio = waveform.numpy()
                
                st.success(f"Loaded audio: {len(clean_audio)} samples at {sample_rate} Hz")
                
            except Exception as e:
                st.error(f"Error loading audio: {e}")
                clean_audio = None
            
            # Clean up
            os.unlink(tmp_path)
    
    elif input_option == "Record audio":
        st.info("Audio recording functionality would be implemented here using streamlit-audio-recorder")
        clean_audio = None
    
    elif input_option == "Generate synthetic speech":
        if st.button("Generate Synthetic Speech"):
            clean_audio = generate_synthetic_speech(
                duration=duration,
                sample_rate=sample_rate,
                freq_range=(100, 8000)
            )
            st.success("Generated synthetic speech")
    
    # Display clean audio
    if clean_audio is not None:
        st.subheader("Clean Speech")
        st.audio(clean_audio, sample_rate=sample_rate)
        
        # Generate noisy audio
        noise = generate_noise(duration, sample_rate, noise_type)
        noisy_audio = add_noise(clean_audio, noise, snr_db)
        
        st.subheader("Noisy Speech")
        st.audio(noisy_audio, sample_rate=sample_rate)
        
        # Store in session state
        st.session_state.clean_audio = clean_audio
        st.session_state.noisy_audio = noisy_audio
        st.session_state.sample_rate = sample_rate

with col2:
    st.header("Enhanced Audio")
    
    if "clean_audio" in st.session_state and "noisy_audio" in st.session_state:
        # Load model
        device = get_device("cpu")  # Use CPU for demo
        
        try:
            if model_type == "Conv-TasNet":
                model = ConvTasNet()
                # Load pretrained weights if available
                checkpoint_path = Path("checkpoints/best.pth")
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                
                # Enhance audio
                with torch.no_grad():
                    noisy_tensor = torch.from_numpy(st.session_state.noisy_audio).float().unsqueeze(0)
                    enhanced_tensor, _ = model(noisy_tensor.unsqueeze(1))
                    enhanced_audio = enhanced_tensor.squeeze().numpy()
            
            elif model_type == "Spectral Subtraction":
                model = SpectralSubtraction()
                noisy_tensor = torch.from_numpy(st.session_state.noisy_audio).float()
                enhanced_tensor = model(noisy_tensor)
                enhanced_audio = enhanced_tensor.numpy()
            
            elif model_type == "Wiener Filter":
                model = WienerFilter()
                noisy_tensor = torch.from_numpy(st.session_state.noisy_audio).float()
                enhanced_tensor = model(noisy_tensor)
                enhanced_audio = enhanced_tensor.numpy()
            
            # Display enhanced audio
            st.subheader("Enhanced Speech")
            st.audio(enhanced_audio, sample_rate=st.session_state.sample_rate)
            
            # Compute metrics
            metrics = EnhancementMetrics()
            
            clean_tensor = torch.from_numpy(st.session_state.clean_audio).float()
            noisy_tensor = torch.from_numpy(st.session_state.noisy_audio).float()
            enhanced_tensor = torch.from_numpy(enhanced_audio).float()
            
            # Compute metrics
            metric_values = metrics.compute_batch_average(
                enhanced_tensor.unsqueeze(0),
                clean_tensor.unsqueeze(0),
                noisy_tensor.unsqueeze(0)
            )
            
            # Display metrics
            st.subheader("Enhancement Metrics")
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric("SI-SDR", f"{metric_values['si_sdr']:.2f} dB")
                st.metric("SI-SDR Improvement", f"{metric_values['si_sdr_improvement']:.2f} dB")
            
            with col_metric2:
                st.metric("SDR", f"{metric_values['sdr']:.2f} dB")
                st.metric("SDR Improvement", f"{metric_values['sdr_improvement']:.2f} dB")
            
            with col_metric3:
                st.metric("SNR", f"{metric_values['snr']:.2f} dB")
                st.metric("SNR Improvement", f"{metric_values['snr_improvement']:.2f} dB")
            
            # Perceptual metrics if available
            if "pesq" in metric_values:
                st.metric("PESQ", f"{metric_values['pesq']:.2f}")
            if "stoi" in metric_values:
                st.metric("STOI", f"{metric_values['stoi']:.3f}")
            
            # Download enhanced audio
            st.subheader("Download")
            
            # Convert to bytes
            enhanced_bytes = (enhanced_audio * 32767).astype(np.int16).tobytes()
            
            st.download_button(
                label="Download Enhanced Audio",
                data=enhanced_bytes,
                file_name="enhanced_speech.wav",
                mime="audio/wav"
            )
        
        except Exception as e:
            st.error(f"Error during enhancement: {e}")
            st.info("Make sure the model is properly trained and checkpoints are available.")
    
    else:
        st.info("Please provide input audio first.")

# Footer
st.markdown("---")
st.markdown("""
**Privacy Notice**: This demo processes audio locally and does not store or transmit your audio data. 
This is a research demonstration only and should not be used for production purposes.
""")

# Model information
with st.expander("Model Information"):
    st.markdown(f"""
    **Current Model**: {model_type}
    
    **Model Description**:
    - **Conv-TasNet**: Deep learning model using convolutional neural networks for speech separation
    - **Spectral Subtraction**: Traditional method that subtracts noise spectrum from noisy signal
    - **Wiener Filter**: Optimal filter that minimizes mean square error
    
    **Audio Parameters**:
    - Sample Rate: {sample_rate} Hz
    - Duration: {duration} seconds
    - Noise Type: {noise_type}
    - SNR: {snr_db} dB
    """)

# Technical details
with st.expander("Technical Details"):
    st.markdown("""
    **Enhancement Methods**:
    
    1. **Conv-TasNet**: 
       - Encoder-decoder architecture with separator network
       - Time-domain processing
       - End-to-end training
    
    2. **Spectral Subtraction**:
       - Frequency-domain processing
       - Noise spectrum estimation and subtraction
       - Spectral floor to prevent over-subtraction
    
    3. **Wiener Filter**:
       - Optimal linear filter
       - Minimizes mean square error
       - Frequency-domain implementation
    
    **Evaluation Metrics**:
    - **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio
    - **SDR**: Signal-to-Distortion Ratio
    - **SNR**: Signal-to-Noise Ratio
    - **PESQ**: Perceptual Evaluation of Speech Quality
    - **STOI**: Short-Time Objective Intelligibility
    """)
