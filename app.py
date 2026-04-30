import streamlit as st
from model_utils import predict_dog_breed
from PIL import Image

st.title("Dog Recognizer")

# Initialize session state
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = None

# Show upload section only if no image is uploaded
if st.session_state.image_uploaded is None:
    st.write("Upload an image to check if it's a dog and see the breed details.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        mode = image.mode
        if mode != 'RGB':
            st.info(f"Uploaded image mode is {mode}. It will be converted to RGB for classification.")
            image = image.convert('RGB')
        
        st.session_state.image_uploaded = image
        st.rerun()

# Show classification interface if image is uploaded
if st.session_state.image_uploaded is not None:
    image = st.session_state.image_uploaded
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Classify", key="classify_btn", use_container_width=True):
            with st.spinner("Classifying..."):
                try:
                    result = predict_dog_breed(image)
                    
                    if result['is_dog']:
                        st.subheader("🐶 Dog Detected!")
                        st.write(f"**Breed:** {result['breed'].title()}")
                        st.write(f"**Origin:** {result['origin']}")
                        st.write(f"**Lifespan:** {result['lifespan']}")
                        st.write(f"**Confidence:** {result['confidence'] * 100:.1f}%")
                    else:
                        st.subheader("❌ Not a Dog")
                        st.write(f"**Predicted class:** {result['breed']}")
                        st.write(f"**Confidence:** {result['confidence'] * 100:.1f}%")
                except Exception as e:
                    st.error(f"Classification failed: {e}")

        if st.button("Clear", key="clear_btn", use_container_width=True):
            st.session_state.image_uploaded = None
            st.rerun()
