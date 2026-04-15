from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from flask_cors import CORS
import os
import sys
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="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")

app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")
CORS(app)

# Model and classes
model = None
classes = ["Acne", "Eczema", "Normal"]


def load_model():
    """Load the pre-trained model from h5 file"""
    global model
    model_file = "model.h5"

    if not os.path.exists(model_file):
        print(f"ERROR: Model file '{model_file}' not found!")
        return False

    try:
        print(f"Loading model from {model_file}...")

        # Try using keras directly (standalone library)
        try:
            print("Attempting to load with standalone keras...")
            import keras
            model = keras.models.load_model(model_file)
            print(f"✓ Model loaded with keras!")
            return True
        except Exception as e:
            print(f"Keras failed ({e}), trying tensorflow.keras...")
            import tensorflow as tf
            model = tf.keras.models.load_model(model_file)
            print(f"✓ Model loaded with tensorflow.keras!")
            return True

    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def predict(img):
    """Make prediction on image"""
    if model is None:
        return "Normal"  # Fallback

    try:
        # Resize to model's expected input size
        img = img.resize((224, 224))

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Print debug info
        print(f"Image shape: {img_array.shape}")
        print(f"Image min/max: {img_array.min()}/{img_array.max()}")

        # Make prediction
        pred = model.predict(img_array, verbose=0)
        print(f"Raw prediction: {pred}")

        # Get the class with highest probability
        class_idx = np.argmax(pred[0])
        confidence = pred[0][class_idx]

        print(
            f"Predicted class: {classes[class_idx]} (confidence: {confidence:.4f})")

        return classes[class_idx]

    except Exception as e:
        print(f"ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return "Normal"


def get_disease_info(disease):
    """Get detailed disease information from OpenAI"""
    try:
        # Special handling for Normal skin
        if disease == "Normal":
            return """
Your skin is in good condition!

✓ What is it: Normal/Healthy skin with no visible skin disease
✓ Characteristics: Neither too oily nor too dry, clear complexion
✓ Prevention:
   - Use a mild cleanser daily
   - Moisturize regularly
   - Use sunscreen (SPF 30+)
   - Stay hydrated
   - Get enough sleep
✓ Home care:
   - Use lukewarm water for washing
   - Pat skin dry gently
   - Drink 8-10 glasses of water daily
   - Keep stress levels low
            """
        
        prompt = f"""
        Give detailed information about {disease} skin disease in simple words.
        Include:
        - What is it
        - Causes
        - Symptoms
        - Prevention
        - Home remedies
        Keep it short and student friendly (max 200 words).
        """

        print(f"Sending request to OpenAI for {disease}...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        result = response.choices[0].message.content
        print(f"✓ Got response from OpenAI")
        return result

    except Exception as e:
        print(f"ERROR fetching disease info: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback remedies
        remedies = {
            "Acne": """
Acne Information:
- What: Skin condition with pimples and blackheads
- Causes: Bacteria, excess oil, clogged pores
- Symptoms: Bumps, redness, oily skin
- Prevention: Keep skin clean, avoid oily foods, wash twice daily
- Home remedies: Neem paste, turmeric, tea tree oil, aloe vera
            """,
            "Eczema": """
Eczema Information:
- What: Inflammatory skin condition
- Causes: Genetics, allergies, dry skin
- Symptoms: Itching, red patches, dry skin
- Prevention: Use moisturizer, avoid triggers, take lukewarm baths
- Home remedies: Coconut oil, oatmeal bath, moisturize regularly
            """,
            "Normal": """
Your skin is healthy! Keep up with basic skincare routine.
- Wash with mild cleanser
- Use moisturizer
- Apply sunscreen daily
            """
        }
        
        return remedies.get(disease, "Could not fetch detailed info")


@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        # Open image
        img = Image.open(file.stream)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Make prediction
        result = predict(img)

        # Get detailed information from OpenAI
        info = get_disease_info(result)

        return jsonify({
            "disease": result,
            "details": info
        })

    except Exception as e:
        print(f"ERROR in predict_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Load model on startup
    print("=" * 50)
    print("Starting Skin Disease Detector...")
    print("=" * 50)

    if load_model():
        print("✓ Ready to accept predictions")
    else:
        print("✗ Failed to load model - will use fallback")

    print("=" * 50)

    # Run Flask app
    app.run(debug=True)
