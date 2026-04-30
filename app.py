import streamlit as st
from model_utils import analyze_image
from PIL import Image

MIN_CONFIDENCE = 0.70

st.title("Dog Recognizer")

# Initialize session state
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = None

# Show upload section only if no image is uploaded
if st.session_state.image_uploaded is None:
    st.write("Upload an image to check if it's a dog and see the breed details.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if getattr(image, "is_animated", False):
            st.warning(
                "Animated or cartoon images are converted to the first frame. "
                "This model is trained on real dog photos, so results may be unreliable for drawings or GIFs."
            )
            image.seek(0)

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
        st.image(image, caption='Uploaded Image', width=500)
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        classify_clicked = st.button("Classify", key="classify_btn", use_container_width=True)
        clear_clicked = st.button("Clear", key="clear_btn", use_container_width=True)

    analysis = None
    if classify_clicked:
        with st.spinner("Classifying..."):
            try:
                analysis = analyze_image(image)
            except Exception as e:
                st.error(f"Classification failed: {e}")

    if clear_clicked:
        st.session_state.image_uploaded = None
        st.rerun()

    if analysis is not None:
        detected_regions = analysis.get('detected_regions', [])
        if detected_regions:
            st.subheader(f"🐶 {len(detected_regions)} Dog(s) Detected!")
            st.write("---")

            if image.width > image.height:
                row_size = min(3, len(detected_regions))
                rows = [detected_regions[i:i + row_size] for i in range(0, len(detected_regions), row_size)]
            else:
                rows = [[region] for region in detected_regions]

            for row in rows:
                cols = st.columns(len(row), gap="small")
                for idx, result in enumerate(row):
                    confidence_pct = result['confidence'] * 100
                    bbox = result['bbox']
                    cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                    width = 180 if len(row) > 1 else 280
                    with cols[idx]:
                        st.image(cropped, caption=f"Dog {result.get('region_index', idx) + 1}", width=width)
                        if result['is_dog']:
                            st.write(f"**Breed:** {result['breed'].title()}")
                            st.write(f"**Origin:** {result['origin']}")
                            st.write(f"**Lifespan:** {result['lifespan']}")
                            st.write(f"**Confidence:** {confidence_pct:.1f}%")
                        else:
                            st.write(f"**Predicted:** {result['breed']}")
                            st.write(f"**Confidence:** {confidence_pct:.1f}%")
                        st.write(f"**Box score:** {result.get('detection_score', 0.0):.2f}")
        else:
            result = analysis.get('fallback_prediction')
            confidence_pct = result['confidence'] * 100

            if confidence_pct < MIN_CONFIDENCE * 100:
                st.warning(
                    f"Prediction confidence is only {confidence_pct:.1f}%. "
                    "Results may be unreliable. Try another image or use a clearer photo."
                )

            if result['is_dog']:
                st.subheader("🐶 Dog Detected!")
                st.write(f"**Breed:** {result['breed'].title()}")
                st.write(f"**Origin:** {result['origin']}")
                st.write(f"**Lifespan:** {result['lifespan']}")
                st.write(f"**Confidence:** {confidence_pct:.1f}%")
            else:
                st.subheader("❌ Not a Dog")
                st.write(f"**Predicted class:** {result['breed']}")
                st.write(f"**Confidence:** {confidence_pct:.1f}%")
