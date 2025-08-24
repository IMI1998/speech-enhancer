import streamlit as st
import torchaudio
import torch
import os
from speechbrain.inference.separation import SepformerSeparation as Enhancer

# Set up the Streamlit page
st.set_page_config(page_title="Speech Enhancer", layout="centered")
st.title("🗣️ Voice Enhancement with SepFormer")
st.markdown("Upload a `.wav` file to enhance speech using the [SepFormer](https://huggingface.co/speechbrain/sepformer-whamr-enhancement) model.")
@st.cache_resource
def load_enhancer():
    enhancer = Enhancer.from_hparams(
        source="speechbrain/sepformer-whamr-enhancement",
        savedir="pretrained_models/sepformer-whamr-enhancement"
    )
    return enhancer

enhancer = load_enhancer()

uploaded_file = st.file_uploader("Choose a WAV file...", type="wav")

if uploaded_file is not None:
    temp_noisy_path = "temp_noisy.wav"
    with open(temp_noisy_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Original Noisy Audio")
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Processing audio... This might take a moment."):
        enhanced_path = "temp_enhanced.wav"

        try:
            enhanced_speech_tensor = enhancer.separate_file(path=temp_noisy_path)
            processed_tensor = enhanced_speech_tensor.squeeze().unsqueeze(0)
            target_rate = 8000
            torchaudio.save(enhanced_path, processed_tensor.cpu(), target_rate)

        except Exception as e:
            st.error(f"An error occurred during enhancement: {e}")
            st.info("Please try with a different audio file or check the file format.")
            if os.path.exists(temp_noisy_path):
                os.remove(temp_noisy_path)
            if os.path.exists(enhanced_path):
                os.remove(enhanced_path)
            st.stop()

    st.success("Enhancement complete!")
    st.subheader("Enhanced Audio")
    st.audio(enhanced_path, format='audio/wav')

    with open(enhanced_path, "rb") as f:
        st.download_button(
            label="Download Enhanced Audio",
            data=f.read(),
            file_name="enhanced_audio.wav",
            mime="audio/wav"
        )
    
    if os.path.exists(temp_noisy_path):
        os.remove(temp_noisy_path)
    if os.path.exists(enhanced_path):
        os.remove(enhanced_path)