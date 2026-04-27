# Dog Recognizer App Explanation

## Files
- `app.py`: user interface file for the Streamlit web app.
- `model_utils.py`: backend helper file that loads the model, preprocesses images, and makes the dog/not-dog prediction.

## How it works
1. The user uploads an image through the Streamlit UI in `app.py`.
2. If the image is not already RGB, it is converted to RGB before classification.
3. The image is displayed in the web app.
4. When the user clicks "Classify", the image is passed to `model_utils.py`.
5. `model_utils.py` resizes and normalizes the image, runs it through a pre-trained ResNet50 model, and checks whether the predicted class is a dog.

## Fix applied
- Some uploaded images used RGBA or another mode with 4 channels.
- The model expects RGB images with 3 channels.
- The code now converts any non-RGB image to RGB before preprocessing, preventing the tensor channel mismatch error.

## Local app URL
- http://localhost:8501
