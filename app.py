import streamlit as st
import pandas as pd
import pickle
import hashlib
from streamlit_geolocation import streamlit_geolocation
import datetime
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------
# Comprehensive Crop Information Database
# ---------------------------------------
CROP_INFO = {
    "rice": {
        "name": "Rice",
        "season": "Kharif (June-September)",
        "water": "High (1500-2500 mm)",
        "duration": "3-6 months",
        "soil": "Clayey loam, pH 5.5-6.5",
        "temperature": "20-35°C",
        "rainfall": "1500-2500 mm",
        "fertilizer": "High nitrogen requirement",
        "description": "Staple food crop requiring abundant water and warm temperatures",
    },
    "wheat": {
        "name": "Wheat",
        "season": "Rabi (October-March)",
        "water": "Medium (500-800 mm)",
        "duration": "3-4 months",
        "soil": "Well-drained loam, pH 6.0-7.5",
        "temperature": "10-25°C",
        "rainfall": "500-800 mm",
        "fertilizer": "Balanced NPK",
        "description": "Winter crop grown in cool temperatures with moderate water",
    },
    "maize": {
        "name": "Maize",
        "season": "Kharif (June-September)",
        "water": "Medium (600-900 mm)",
        "duration": "2-3 months",
        "soil": "Well-drained soil, pH 6.0-7.5",
        "temperature": "18-27°C",
        "rainfall": "600-900 mm",
        "fertilizer": "High nitrogen",
        "description": "Versatile crop used for food, fodder and industrial purposes",
    },
    "cotton": {
        "name": "Cotton",
        "season": "Kharif (June-December)",
        "water": "Low-Medium (600-1200 mm)",
        "duration": "5-6 months",
        "soil": "Black soil, pH 6.0-8.0",
        "temperature": "21-30°C",
        "rainfall": "600-1200 mm",
        "fertilizer": "Moderate nitrogen, high potassium",
        "description": "Cash crop requiring warm climate and well-drained soil",
    },
    "sugarcane": {
        "name": "Sugarcane",
        "season": "Throughout year",
        "water": "High (1500-2500 mm)",
        "duration": "10-12 months",
        "soil": "Deep heavy soil, pH 6.5-7.5",
        "temperature": "20-30°C",
        "rainfall": "1500-2500 mm",
        "fertilizer": "High nitrogen and potassium",
        "description": "Tropical crop requiring long growing season and abundant water",
    },
    "soybean": {
        "name": "Soybean",
        "season": "Kharif (June-September)",
        "water": "Medium (450-700 mm)",
        "duration": "3-4 months",
        "soil": "Well-drained loam, pH 6.0-7.0",
        "temperature": "20-30°C",
        "rainfall": "450-700 mm",
        "fertilizer": "Low nitrogen, high phosphorus",
        "description": "Oilseed crop that fixes atmospheric nitrogen",
    },
    "barley": {
        "name": "Barley",
        "season": "Rabi (October-March)",
        "water": "Low (300-500 mm)",
        "duration": "3-4 months",
        "soil": "Well-drained loam, pH 6.0-8.0",
        "temperature": "12-25°C",
        "rainfall": "300-500 mm",
        "fertilizer": "Moderate NPK",
        "description": "Hardy cereal crop tolerant to drought and salinity",
    },
    "mungbean": {
        "name": "Mung Bean",
        "season": "Kharif (June-September)",
        "water": "Low (400-600 mm)",
        "duration": "2-3 months",
        "soil": "Well-drained sandy loam, pH 6.2-7.2",
        "temperature": "25-35°C",
        "rainfall": "400-600 mm",
        "fertilizer": "Low nitrogen, medium phosphorus",
        "description": "Short duration pulse crop, drought resistant",
    },
    "pigeonpeas": {
        "name": "Pigeon Peas",
        "season": "Kharif (June-December)",
        "water": "Low (600-800 mm)",
        "duration": "5-6 months",
        "soil": "Well-drained loam, pH 6.0-7.5",
        "temperature": "20-30°C",
        "rainfall": "600-800 mm",
        "fertilizer": "Low nitrogen, medium phosphorus",
        "description": "Drought resistant pulse crop with deep root system",
    },
    "lentil": {
        "name": "Lentil",
        "season": "Rabi (October-March)",
        "water": "Low (350-500 mm)",
        "duration": "3-4 months",
        "soil": "Well-drained loam, pH 6.0-7.5",
        "temperature": "15-25°C",
        "rainfall": "350-500 mm",
        "fertilizer": "Low nitrogen, medium phosphorus",
        "description": "Cool season pulse crop, important protein source",
    },
    "potato": {
        "name": "Potato",
        "season": "Rabi (October-February)",
        "water": "Medium (500-700 mm)",
        "duration": "3-4 months",
        "soil": "Well-drained sandy loam, pH 5.0-6.5",
        "temperature": "15-20°C",
        "rainfall": "500-700 mm",
        "fertilizer": "High potassium, medium nitrogen",
        "description": "Cool season tuber crop requiring well-drained soil",
    },
    "coconut": {
        "name": "Coconut",
        "season": "Throughout year",
        "water": "High (1500-2500 mm)",
        "duration": "Perennial (8-10 years)",
        "soil": "Sandy loam, pH 5.5-7.0",
        "temperature": "25-32°C",
        "rainfall": "1500-2500 mm",
        "fertilizer": "High potassium, medium nitrogen",
        "description": "Tropical perennial crop, requires coastal climate",
    },
    "coffee": {
        "name": "Coffee",
        "season": "Perennial",
        "water": "High (1500-2500 mm)",
        "duration": "3-4 years to first harvest",
        "soil": "Volcanic soil, pH 6.0-6.5",
        "temperature": "15-28°C",
        "rainfall": "1500-2500 mm",
        "fertilizer": "Balanced NPK with micronutrients",
        "description": "Shade loving perennial crop, requires high altitude",
    },
    "jute": {
        "name": "Jute",
        "season": "Kharif (March-August)",
        "water": "High (1500-2000 mm)",
        "duration": "4-5 months",
        "soil": "Alluvial soil, pH 6.0-7.5",
        "temperature": "24-37°C",
        "rainfall": "1500-2000 mm",
        "fertilizer": "High nitrogen",
        "description": "Fiber crop requiring high humidity and temperature",
    },
    "tea": {
        "name": "Tea",
        "season": "Perennial",
        "water": "High (1500-3000 mm)",
        "duration": "3 years to first harvest",
        "soil": "Acidic soil, pH 4.5-5.5",
        "temperature": "15-30°C",
        "rainfall": "1500-3000 mm",
        "fertilizer": "Acidic fertilizers, high nitrogen",
        "description": "Evergreen shrub requiring acidic soil and high rainfall",
    },
    "blackgram": {
        "name": "Black Gram",
        "season": "Kharif (June-September)",
        "water": "Low (400-600 mm)",
        "duration": "2.5-3 months",
        "soil": "Well-drained loam, pH 6.5-7.5",
        "temperature": "25-35°C",
        "rainfall": "400-600 mm",
        "fertilizer": "Low nitrogen, medium phosphorus",
        "description": "Short duration pulse crop, heat tolerant",
    },
    "mothbeans": {
        "name": "Moth Beans",
        "season": "Kharif (June-September)",
        "water": "Very Low (300-500 mm)",
        "duration": "2-3 months",
        "soil": "Sandy loam, pH 7.0-8.5",
        "temperature": "25-35°C",
        "rainfall": "300-500 mm",
        "fertilizer": "Low nutrient requirement",
        "description": "Highly drought resistant pulse crop for arid regions",
    },
    "kidneybeans": {
        "name": "Kidney Beans",
        "season": "Rabi (October-December)",
        "water": "Medium (500-700 mm)",
        "duration": "3-4 months",
        "soil": "Well-drained loam, pH 6.0-7.0",
        "temperature": "15-25°C",
        "rainfall": "500-700 mm",
        "fertilizer": "Medium NPK",
        "description": "Cool season bean crop, sensitive to high temperatures",
    },
    "grapes": {
        "name": "Grapes",
        "season": "Perennial",
        "water": "Medium (600-800 mm)",
        "duration": "2-3 years to first harvest",
        "soil": "Well-drained loam, pH 6.5-7.5",
        "temperature": "15-35°C",
        "rainfall": "600-800 mm",
        "fertilizer": "Balanced with high potassium",
        "description": "Perennial fruit crop requiring pruning and training",
    },
    "watermelon": {
        "name": "Watermelon",
        "season": "Summer (February-May)",
        "water": "Medium (500-700 mm)",
        "duration": "3-4 months",
        "soil": "Sandy loam, pH 6.0-7.0",
        "temperature": "25-35°C",
        "rainfall": "500-700 mm",
        "fertilizer": "High potassium, medium nitrogen",
        "description": "Summer fruit crop requiring warm temperatures",
    },
    "apple": {
        "name": "Apple",
        "season": "Temperate (March-September)",
        "water": "Medium (800-1200 mm)",
        "duration": "3-5 years to first harvest",
        "soil": "Well-drained loam, pH 6.0-7.0",
        "temperature": "7-24°C",
        "rainfall": "800-1200 mm",
        "fertilizer": "Balanced NPK",
        "description": "Temperate fruit crop requiring chilling hours",
    },
    "mango": {
        "name": "Mango",
        "season": "Summer (March-June)",
        "water": "Medium (800-1200 mm)",
        "duration": "4-5 years to first harvest",
        "soil": "Deep well-drained soil, pH 5.5-7.5",
        "temperature": "24-30°C",
        "rainfall": "800-1200 mm",
        "fertilizer": "Balanced with high potassium during fruiting",
        "description": "Tropical fruit crop, king of fruits",
    },
    "banana": {
        "name": "Banana",
        "season": "Throughout year",
        "water": "High (1200-2000 mm)",
        "duration": "12-15 months to harvest",
        "soil": "Deep rich loam, pH 6.0-7.5",
        "temperature": "25-35°C",
        "rainfall": "1200-2000 mm",
        "fertilizer": "High potassium and nitrogen",
        "description": "Tropical fruit crop requiring high humidity and temperature",
    },
    "muskmelon": {
        "name": "Muskmelon",
        "season": "Summer (February-May)",
        "water": "Medium (500-700 mm)",
        "duration": "3-4 months",
        "soil": "Sandy loam, pH 6.0-7.0",
        "temperature": "25-35°C",
        "rainfall": "500-700 mm",
        "fertilizer": "Balanced with high potassium",
        "description": "Summer cucurbit, requires warm dry weather during ripening",
    },
}

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="🌾 Smart Crop Predictor",
    page_icon="🌱",
    layout="wide",
)

# ---------------------------------------
# Custom CSS Styling + Animation
# ---------------------------------------
st.markdown(
    """
<style>
body {
    background: linear-gradient(135deg, #e6f4ea, #f5fff7);
    animation: fadein 1s ease-in;
}

@keyframes fadein {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

.main {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

h1, h2, h3 {
    color: #2d6a4f;
}

/* Animated Crop Icon */
.floating-icon {
    width: 100px;
    margin: 0 auto;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-15px); }
    100% { transform: translateY(0px); }
}

/* Login Box */
.login-box {
    background-color: #ffffff;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    width: 400px;
    margin: 80px auto;
    text-align: center;
    animation: fadeup 1.5s ease;
}

@keyframes fadeup {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.login-box input {
    border-radius: 10px !important;
}

.stButton>button {
    background-color: #2d6a4f;
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.6em 1.4em;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #1b4332;
}

/* Premium Badge */
.premium-badge {
    background: linear-gradient(45deg, #ffd700, #ffed4e);
    color: #b8860b;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9em;
    display: inline-block;
    margin: 5px 0;
}

/* Crop Card */
.crop-card {
    background-color: #222;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.crop-card h3 {
    font-size: 1.2em !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------
# Dummy Credentials
# ---------------------------------------
USER_CREDENTIALS = {"admin": "1234", "farmer": "crop2025", "abhi": "1234"}

# ---------------------------------------
# Persist Login via Query Params
# ---------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Check for persisted token from query params
query_params = st.query_params
if "token" in query_params:
    token = query_params["token"]
    expected_tokens = {
        user: hashlib.sha256(f"{user}{pwd}".encode()).hexdigest()
        for user, pwd in USER_CREDENTIALS.items()
    }
    if token in expected_tokens.values():
        st.session_state.logged_in = True
        # Optionally set username
        for user, pwd in USER_CREDENTIALS.items():
            if expected_tokens[user] == token:
                st.session_state.username = user
                break


# ---------------------------------------
# Model Loader
# ---------------------------------------
@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {path}")
        st.info("💡 Please make sure the model files exist in the 'models' folder")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# Create a simple fallback model for demonstration
def create_fallback_model():
    """Create a simple fallback model if actual models are not available"""
    np.random.seed(42)
    X_dummy = np.random.rand(100, 7)
    y_dummy = np.random.choice(["rice", "wheat", "maize", "cotton", "sugarcane"], 100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    return model


MODEL_PATHS = {
    "🌿 Decision Tree": "./DecisionTree.pkl",
    "🌾 Random Forest": "./RandomForest.pkl",
    "🌱 NB Classifier": "./NBClassifier.pkl",
    "🌽 Logistic Regression": "./LogisticRegression.pkl",
}


def predict_crop(model, inputs):
    df = pd.DataFrame([inputs])
    prediction = model.predict(df)[0]
    return prediction


def validate_inputs(inputs):
    """Validate input ranges and types"""
    errors = []

    if not (0 <= inputs["ph"] <= 14):
        errors.append("pH must be between 0-14")

    if not (0 <= inputs["humidity"] <= 100):
        errors.append("Humidity must be between 0-100%")

    if inputs["temperature"] < -10 or inputs["temperature"] > 60:
        errors.append("Temperature seems unrealistic for crop growth")

    if inputs["rainfall"] < 0:
        errors.append("Rainfall cannot be negative")

    return errors


def display_crop_details(crop_name):
    """Display comprehensive information about the recommended crop"""
    # Convert to lowercase and handle different naming conventions
    crop_key = crop_name.lower().strip()

    # Handle common variations in crop names
    name_variations = {
        "mungbean": "mungbean",
        "mung": "mungbean",
        "pigeonpeas": "pigeonpeas",
        "pigeon pea": "pigeonpeas",
        "blackgram": "blackgram",
        "black gram": "blackgram",
        "mothbeans": "mothbeans",
        "moth bean": "mothbeans",
        "kidneybeans": "kidneybeans",
        "kidney bean": "kidneybeans",
        "muskmelon": "muskmelon",
        "muskmelon": "muskmelon",
    }

    # Check if crop exists in variations or directly in CROP_INFO
    crop_key = name_variations.get(crop_key, crop_key)

    if crop_key in CROP_INFO:
        info = CROP_INFO[crop_key]

        st.markdown("#### 📋 Crop Growing Information")

        # Create a nice card-like display
        st.markdown(
            f"""
        <div class="crop-card">
            <h3 style="color: #2d6a4f; margin-bottom: 15px;">🌱 {info['name']}</h3>
            <p><strong>Description:</strong> {info['description']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
    <style>
    /* Change font size for st.metric values and labels */
    [data-testid="stMetricValue"] {
        font-size: 22px !important;  /* Adjust value font size */
    }
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;  /* Adjust label font size */
    }

    /* Change font size for st.info box text */
    .stAlert {
        font-size: 18px !important;  /* Adjust info text size */
    }
    </style>
    """,
            unsafe_allow_html=True,
        )

        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌤 Growing Season", info["season"])
            st.metric("🌡 Temperature", info["temperature"])
        with col2:
            st.metric("💧 Water Needs", info["water"])
            st.metric("🌧 Rainfall", info["rainfall"])
        with col3:
            st.metric("📅 Duration", info["duration"])
            st.metric(
                "🔄 Crop Type",
                (
                    "Kharif"
                    if "Kharif" in info["season"]
                    else "Rabi" if "Rabi" in info["season"] else "Perennial"
                ),
            )
        with col4:
            st.metric("🌱 Soil Type", info["soil"])
            st.metric("🧪 Fertilizer", info["fertilizer"])

        # Additional tips
        st.info(f"💡 *Growing Tip*: {info['description']}")

    else:
        st.warning(
            f"ℹ Detailed information for '{crop_name}' is not available in our database."
        )
        st.info(
            "💡 The crop prediction is based on soil and climate conditions, but specific growing guidelines may vary by region."
        )


# Helper function to get all available crops (useful for debugging)
def get_available_crops():
    """Return list of all crops in the database"""
    return list(CROP_INFO.keys())


def display_statistics(model_name, inputs, prediction):
    """Display statistics and model information"""
    st.markdown("### 📊 Stats about data")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### 📈 Feature Values")
        feature_data = pd.DataFrame(
            {"Feature": list(inputs.keys()), "Value": list(inputs.values())}
        )
        st.dataframe(feature_data, use_container_width=True)

        # Create a simple bar chart for feature values
        st.bar_chart(feature_data.set_index("Feature"))
    with colB:
        st.markdown("#### 🧠 Prediction Details")
        st.json(
            {
                "Model Used": model_name,
                "Predicted Crop": prediction,
                "Confidence": "High",  # You can add actual confidence scores if your model provides them
                "Input Features": len(inputs),
            }
        )

    st.write(
        f"Based on the soil and climate conditions in your location, the model recommends *{prediction.upper()}* "
    )


# ---------------------------------------
# Reverse Geocoding Function
# ---------------------------------------
@st.cache_data
def get_place_name(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        response = requests.get(url, headers={"User-Agent": "SmartCropPredictor/1.0"})
        if response.status_code == 200:
            data = response.json()
            return data.get("display_name", "Unknown location")
        else:
            return "Unable to fetch location name"
    except Exception as e:
        return f"Error fetching location name: {str(e)}"


# ---------------------------------------
# Login Page
# ---------------------------------------
if not st.session_state.logged_in:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2909/2909592.png",
        width=100,
        output_format="PNG",
        caption="",
    )
    st.markdown("<div class='floating-icon'></div>", unsafe_allow_html=True)
    st.title("🌾 Smart Crop Predictor")
    st.subheader("Login to continue")
    st.markdown(
        """
    <style>
    div[data-testid="stTextInput"] {
        width: 300px;  /* Set your desired width */
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            token = hashlib.sha256(f"{username}{password}".encode()).hexdigest()
            st.query_params["token"] = token
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login Successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# Main Dashboard
# ---------------------------------------
else:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2909/2909592.png", width=120)
        st.title("🌿 Crop Predictor")
        if "username" in st.session_state:
            st.write(f"👋 Welcome, {st.session_state.username}!")
            st.markdown(
                '<span class="premium-badge">⭐ PREMIUM ACCESS</span>',
                unsafe_allow_html=True,
            )

        selected_model = st.selectbox(
            "Choose Prediction Model:", list(MODEL_PATHS.keys())
        )
        model = load_model(MODEL_PATHS[selected_model])
        if model is None:
            st.warning("⚠ Using fallback model (demo mode)")
            model = create_fallback_model()

        st.markdown("---")
        st.write("📍 *Select Location*")

        location_key = "location"
        if location_key not in st.session_state:
            st.session_state[location_key] = "Maharashtra, India"

        use_current_key = "use_current_location"
        use_current = st.checkbox("Use Current Location", key=use_current_key)

        if use_current:
            location_dict = streamlit_geolocation()
            if (
                location_dict
                and isinstance(location_dict, dict)
                and location_dict.get("latitude") is not None
            ):
                new_loc = (
                    f"{location_dict['latitude']:.6f}, {location_dict['longitude']:.6f}"
                )
                if st.session_state[location_key] != new_loc:
                    st.session_state[location_key] = new_loc
                    st.rerun()

        location = st.text_input(
            "Enter your Location (or Coordinates):", key=location_key
        )

        # Display place name if location is coordinates
        place_name = location
        if "," in location:
            try:
                lat_str, lon_str = location.split(",", 1)
                lat = float(lat_str.strip())
                lon = float(lon_str.strip())
                place_name = get_place_name(lat, lon)
                st.write(f"**Place Name:** {place_name}")
            except ValueError:
                st.warning("Invalid coordinate format. Please enter as 'lat, lon'.")

        st.markdown("---")
        show_stats = st.checkbox("📊 Stats about data")

        # Premium Feature: Usage History
        st.markdown("---")
        if st.checkbox("📋 Premium: View Prediction History"):
            if "history" not in st.session_state:
                st.session_state.history = []
            if st.session_state.history:
                st.subheader("Recent Predictions")
                for pred in st.session_state.history[-5:]:  # Last 5
                    st.write(
                        f"**{pred['crop']}** at {pred['location']} - {pred['date']}"
                    )
            else:
                st.info("No predictions yet. Make one to see history!")

        if st.button("🚪 Logout"):
            st.query_params.clear()
            st.session_state.logged_in = False
            if "username" in st.session_state:
                del st.session_state.username
            if "history" in st.session_state:
                del st.session_state.history
            st.rerun()

    # ---------------------------------------
    # Main App UI
    # ---------------------------------------
    st.title("🌾 Smart Crop Prediction System")
    st.write("### Predict the best crop to grow based on soil and climate conditions.")
    st.markdown(
        '<span class="premium-badge">🔒 Premium: Advanced AI Models & Insights</span>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input(
            "Nitrogen (N) ppm",
            0.0,
            150.0,
            90.0,
            help="Nitrogen content in parts per million",
        )
        P = st.number_input(
            "Phosphorus (P) ppm",
            0.0,
            150.0,
            42.0,
            help="Phosphorus content in parts per million",
        )
    with col2:
        K = st.number_input(
            "Potassium (K) ppm",
            0.0,
            200.0,
            43.0,
            help="Potassium content in parts per million",
        )
        temperature = st.number_input(
            "Temperature (°C)", 0.0, 50.0, 25.0, help="Average temperature in Celsius"
        )
    with col3:
        humidity = st.number_input(
            "Humidity (%)", 0.0, 100.0, 80.0, help="Relative humidity percentage"
        )
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, help="Soil pH level (0-14)")
        rainfall = st.number_input(
            "Rainfall (mm)", 0.0, 300.0, 100.0, help="Annual rainfall in millimeters"
        )

    if st.button("🌱 Predict Crop", type="primary"):
        inputs = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }

        # Validate inputs
        validation_errors = validate_inputs(inputs)
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            try:
                result = predict_crop(model, inputs)

                # Display results
                st.success(f"🌾 Recommended Crop: *{result.upper()}*")
                st.balloons()

                # Display crop details
                display_crop_details(result)

                # Premium: Save to history
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append(
                    {
                        "crop": result,
                        "location": place_name,  # Use place name if available
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": selected_model,
                    }
                )

                # Show statistics if requested
                if show_stats:
                    display_statistics(selected_model, inputs, result)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.info(
                    "💡 Please check if the model is properly loaded and input values are valid"
                )

        with st.expander("📊 Stats for Nerds"):
            colA, colB = st.columns(2)

            with colA:
                st.markdown("#### 📉 Model Training Accuracy")
                st.image(
                    "./assets/model_accuracy.png",
                    caption="Model Accuracies",
                )

            with colB:
                st.markdown("#### 📈 Dataset Statistics / Correlation")
                st.image(
                    "./assets/dataset_stats.png",
                    caption="Dataset Correlation / Feature Importance",
                )

            st.markdown("#### 🧮 Mathematical Dataset Description")
            st.image(
                "./assets/math_description.png",
                caption="Mathematical Overview of Dataset",
            )

            st.markdown("#### 🧠 Model & Input Details")
            st.json(
                {
                    "Model Used": selected_model,
                    "Location": place_name,
                    "Inputs": inputs,
                }
            )

    # Additional information section
    with st.expander("💡 How to use this tool"):
        st.markdown(
            """
        *Instructions:*
        1. *Select a model* from the sidebar (different models may give slightly different results)
        2. *Enter your location* for contextual recommendations
        3. *Input soil and climate parameters* based on your soil test results and local weather data
        4. *Click 'Predict Crop'* to get the recommended crop
       
        *Parameter Guidelines:*
        - *N, P, K*: Soil nutrient levels from soil testing (in ppm)
        - *Temperature*: Average growing season temperature
        - *Humidity*: Average relative humidity during growing season
        - *pH*: Soil acidity/alkalinity (most crops prefer 6.0-7.5)
        - *Rainfall*: Annual or seasonal rainfall in your region
        """
        )

    st.markdown(
        """
    ---
    ✨ Built with ❤ using Streamlit | Powered by AI & Data Science | Premium Edition
    """
    )
