from flask import Flask, request, jsonify, render_template
from PIL import Image
from flask_cors import CORS
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in parent directory (root) or current directory
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if not os.path.exists(env_path):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# Initialize OpenAI client safely
client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        print("✓ OpenAI client initialized")
    else:
        print("⚠ OPENAI_API_KEY not set - disease info will use fallback")
except Exception as e:
    print(f"⚠ Failed to initialize OpenAI client: {e}")
    print("  Disease info will use fallback responses")

app = Flask(__name__, template_folder=".",
            static_folder=".", static_url_path="")
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


def simple_skin_analysis(img):
    """Simple image-based skin analysis without ML model or numpy"""
    try:
        # Resize to standard size
        img = img.resize((128, 128))

        # Extract features using pure Python
        # 1. Calculate red channel intensity (acne has more red due to inflammation)
        if img.mode == 'RGB':
            r, g, b = img.split()
            r_pixels = list(r.getdata())
            red_intensity = sum(r_pixels) / (len(r_pixels) * 255.0)
        else:
            red_intensity = 0.3

        # 2. Calculate roughness (variance) - eczema has more texture variation
        gray = img.convert('L')
        gray_pixels = list(gray.getdata())

        # Calculate mean
        mean_brightness = sum(gray_pixels) / (len(gray_pixels) * 255.0)

        # Calculate variance/std deviation
        squared_diffs = [(p/255.0 - mean_brightness) ** 2 for p in gray_pixels]
        variance = sum(squared_diffs) / len(squared_diffs)
        roughness = variance ** 0.5  # sqrt for std dev

        # 3. Calculate average brightness (already calculated above)
        brightness = mean_brightness

        print(
            f"DEBUG - Red intensity: {red_intensity:.3f}, Roughness: {roughness:.3f}, Brightness: {brightness:.3f}")

        # Simple decision logic
        # If high red intensity and moderate brightness -> likely Acne
        # If high roughness/variance -> likely Eczema
        # Otherwise -> Normal

        acne_score = red_intensity * 0.7 + (1 - brightness) * 0.3
        eczema_score = roughness * 0.6 + (brightness - 0.3) * 0.2

        print(
            f"DEBUG - Acne score: {acne_score:.3f}, Eczema score: {eczema_score:.3f}")

        if acne_score > 0.45:
            return "Acne"
        elif eczema_score > 0.35:
            return "Eczema"
        else:
            return "Normal"

    except Exception as e:
        print(f"ERROR in simple_skin_analysis: {e}")
        return "Normal"


def predict(img):
    """Make prediction on image"""
    if model is None:
        # Use simple image analysis when ML model is not available
        # (This is the normal case when TensorFlow is not installed)
        return simple_skin_analysis(img)

    try:
        # ML model prediction (requires TensorFlow)
        # This code path only executes if model.h5 is present AND TensorFlow is installed
        import numpy as np

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
        # Fallback to simple analysis
        return simple_skin_analysis(img)


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

        # Check if OpenAI client is available
        if client is None:
            print(
                f"OpenAI client not available, using fallback for {disease}...")
            raise Exception("OpenAI client not initialized")

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

    # Run Flask app - use environment variables for port/host
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    app.run(host="0.0.0.0", port=port, debug=debug_mode)
