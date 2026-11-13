
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ---------------- VISIBLE LEAF DISCOLORATION ESTIMATION ----------------

def estimate_visible_discoloration(image):
    # Resize image for consistent processing
    sample = image.resize((224, 224)).convert("HSV")

    pixels = np.array(sample, dtype=np.float32)

    hue = pixels[:, :, 0] * 360 / 255
    saturation = pixels[:, :, 1]
    brightness = pixels[:, :, 2]

    # Identify pixels with colours commonly found in leaves
    leaf_mask = (
        (hue >= 15) &
        (hue <= 150) &
        (saturation > 40) &
        (brightness > 25)
    )

    # Identify yellow/brown discoloured pixels
    affected_mask = (
        leaf_mask &
        (hue >= 10) &
        (hue <= 35) &
        (saturation > 45) &
        (brightness > 25)
    )

    total_leaf_pixels = np.sum(leaf_mask)
    affected_pixels = np.sum(affected_mask)

    if total_leaf_pixels == 0:
        return None

    percentage = (
        affected_pixels / total_leaf_pixels
    ) * 100

    return round(float(percentage), 2)



# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="PHYTO-SCAN",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 PHYTO-SCAN")
st.subheader("AI-Assisted Plant Health Analysis")

st.write(
    "Upload a leaf photograph to receive a preliminary "
    "AI-based plant health screening."
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "phytoscan_best.keras"
    )

@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)
model = load_model()
class_names = load_class_names()

# ---------------- PLANT SELECTION ----------------

plant_options = {
    "Tomato": "Tomato",
    "Potato": "Potato",
    "Bell Pepper": "Pepper,_bell"
}

selected_plant = st.selectbox(
    "🌿 Select your plant",
    list(plant_options.keys())
)

plant_prefix = plant_options[selected_plant] + "___"

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "📸 Upload a clear leaf photograph",
    type=["jpg", "jpeg", "png"]
)

# ---------------- CARE GUIDANCE ----------------

care_guidance = {
    "healthy": [
        "Continue regular watering according to soil moisture.",
        "Provide sufficient sunlight appropriate for the plant.",
        "Inspect leaves regularly for unusual spots or colour changes.",
        "Maintain good spacing and airflow around the plant."
    ],

    "bacterial spot": [
        "Avoid working with plants when their leaves are wet.",
        "Remove badly affected fallen leaves where practical.",
        "Clean gardening tools after handling suspected infected plants.",
        "Avoid splashing soil or water onto healthy foliage."
    ],

    "early blight": [
        "Remove fallen infected leaves to reduce the spread of spores.",
        "Water near the soil instead of wetting the leaves.",
        "Maintain adequate spacing for better airflow.",
        "Consider rotating crops with non-host plants in future growing seasons."
    ],

    "late blight": [
        "Check nearby leaves for rapidly developing dark or water-soaked patches.",
        "Avoid prolonged moisture on leaves.",
        "Do not move between suspected infected and healthy plants without cleaning tools.",
        "Seek prompt advice from a local agricultural expert."
    ],

    "leaf mold": [
        "Improve ventilation around the plant.",
        "Avoid overhead irrigation, especially in humid conditions.",
        "Remove affected fallen leaves carefully.",
        "Monitor the undersides of leaves for further changes."
    ],

    "septoria": [
        "Remove fallen leaves that may contain fungal spores.",
        "Avoid splashing water from soil onto lower leaves.",
        "Maintain sufficient spacing between plants.",
        "Monitor lower leaves for additional spotting."
    ],

    "spider mites": [
        "Inspect the undersides of leaves for tiny mites or fine webbing.",
        "Check whether stippling or yellow speckles are spreading.",
        "Maintain suitable watering to avoid plant stress.",
        "Seek agricultural advice if the infestation continues."
    ],

    "target spot": [
        "Inspect surrounding leaves for similar circular lesions.",
        "Avoid keeping foliage wet for extended periods.",
        "Remove fallen affected plant debris.",
        "Improve airflow by maintaining suitable plant spacing."
    ],

    "virus": [
        "Keep the suspected plant separate where practical.",
        "Clean tools after handling the suspected plant.",
        "Monitor nearby plants for unusual leaf patterns.",
        "Seek advice from a local agricultural expert about identification and management."
    ],

    "mosaic": [
        "Monitor leaves for irregular light and dark green patterns.",
        "Keep the suspected plant separate where practical.",
        "Clean gardening tools after handling the plant.",
        "Seek expert advice because different viruses can cause similar symptoms."
    ]
}
# ---------------- WHY THIS PREDICTION ----------------

prediction_explanations = {
    "healthy": [
        "The model did not detect a strong disease pattern in the uploaded leaf.",
        "The leaf appears relatively uniform in colour and visible pattern.",
        "Continue monitoring the plant for new spots, colour changes or unusual growth."
    ],

    "bacterial spot": [
        "Bacterial spot can produce small dark or water-soaked-looking spots on leaves.",
        "The model detected visual patterns that are associated with the Bacterial Spot class.",
        "Similar-looking symptoms can occur in other plant conditions."
    ],

    "early blight": [
        "Early blight can produce dark spots and circular lesion patterns on leaves.",
        "The model detected visual patterns associated with the Early Blight class.",
        "Similar symptoms can occur in other leaf diseases."
    ],

    "late blight": [
        "Late blight can cause dark or water-soaked-looking areas on leaves.",
        "The model detected visual patterns associated with the Late Blight class.",
        "Symptoms can overlap with other plant diseases."
    ],

    "leaf mold": [
        "Leaf mold can cause changes in leaf colour and visible patterns, especially under humid conditions.",
        "The model detected visual patterns associated with the Leaf Mold class.",
        "Checking the underside of leaves can provide additional information."
    ],

    "septoria": [
        "Septoria leaf spot commonly produces multiple small spots on leaves.",
        "The model detected visual patterns associated with the Septoria Leaf Spot class.",
        "Other leaf diseases can sometimes produce similar spots."
    ],

    "spider mites": [
        "Spider-mite damage can produce fine speckling or stippling on leaves.",
        "The model detected visual patterns associated with the Spider Mites class.",
        "The undersides of leaves should be checked for mites or fine webbing."
    ],

    "target spot": [
        "Target Spot can produce circular lesions with visible patterns on leaves.",
        "The model detected visual patterns associated with the Target Spot class.",
        "Similar circular lesions can occur in other diseases."
    ],

    "virus": [
        "Plant viruses can cause unusual leaf colour patterns, curling or distortion.",
        "The model detected visual patterns associated with the virus class.",
        "Different viral diseases can have similar symptoms."
    ],

    "mosaic": [
        "Mosaic viruses can produce irregular light and dark green patterns on leaves.",
        "The model detected visual patterns associated with the Tomato Mosaic Virus class.",
        "Different plant viruses can have similar visual symptoms."
    ]
}

# ---------------- ANALYSIS ----------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded leaf photograph",
        use_container_width=True
    )

    if st.button("🔍 Analyse Leaf", type="primary"):

        with st.spinner("PHYTO-SCAN is analysing your image..."):

            # Resize image
            resized_image = image.resize((224, 224))

            # Convert to model input format
            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Predict
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            # Select classes belonging to chosen plant
            plant_indices = [
                i for i, name in enumerate(class_names)
                if name.startswith(plant_prefix)
            ]

            if not plant_indices:
                st.error("Could not find classes for this plant.")
                st.stop()

            # Find highest-scoring class for selected plant
            best_index = max(
                plant_indices,
                key=lambda i: predictions[i]
            )

            confidence = float(predictions[best_index])

            raw_label = class_names[best_index]

            condition = raw_label.split("___", 1)[1]
        condition = condition.replace("_", " ")
        condition = condition.replace(",", " ")
        condition = " ".join(condition.split())

        # ---------------- RESULTS ----------------

        st.divider()
        st.header("🌱 PHYTO-SCAN Results")

        st.write("**Selected plant:**", selected_plant)
        st.write("**Predicted condition:**")

        if "healthy" in condition.lower():
            st.success("Healthy leaf — model prediction")
        else:
            st.warning(condition)

        st.metric(
            "Model confidence score",
            f"{confidence * 100:.2f}%"
        )

        # ---------------- VISIBLE DISCOLORATION RESULT ----------------

        discoloration = estimate_visible_discoloration(image)
        st.success("Visible discoloration analysis completed.")
        st.divider()
        st.subheader("🍃 Visible Leaf Discoloration")

        if discoloration is None:
            st.warning("Could not estimate visible discoloration.")
        else:
            st.metric(
                "Estimated yellow/brown colour proportion",
                f"{discoloration:.2f}%"
            )

        st.caption(
            "This is an approximate colour-based estimate, "
            "not a measurement of disease severity or actual affected leaf area."
        )

        
        # ---------------- PHYTO-CARE ----------------

        st.divider()
        st.header("🩺 PHYTO-CARE")
        st.write("General care and prevention suggestions:")

        condition_lower = condition.lower()

        if "healthy" in condition_lower:
            care_key = "healthy"
        elif "spider mites" in condition_lower:
            care_key = "spider mites"
        elif "bacterial spot" in condition_lower:
            care_key = "bacterial spot"
        elif "early blight" in condition_lower:
            care_key = "early blight"
        elif "late blight" in condition_lower:
            care_key = "late blight"
        elif "leaf mold" in condition_lower:
            care_key = "leaf mold"
        elif "septoria" in condition_lower:
            care_key = "septoria"
        elif "target spot" in condition_lower:
            care_key = "target spot"
        elif "virus" in condition_lower:
            care_key = "virus"
        elif "mosaic" in condition_lower:
            care_key = "mosaic"
        else:
            care_key = "healthy"

        for suggestion in care_guidance[care_key]:
            st.markdown(f"- {suggestion}")
        # ---------------- WHY THIS PREDICTION ----------------

        st.divider()
        st.subheader("🔎 Why this prediction?")

        for explanation in prediction_explanations[care_key]:
            st.markdown(f"- {explanation}")

        st.caption(
            "This section provides general educational context for the predicted class. "
            "It does not show the exact visual features used by the AI model."
        )

        # ---------------- DISCLAIMER ----------------

        st.divider()

        st.info(
            "PHYTO-SCAN is an educational AI screening prototype. "
            "Its predictions are not a definitive plant diagnosis. "
            "Similar symptoms can occur in different conditions. "
            "Consult an agricultural expert before making treatment decisions."
        )

# ---------------- ABOUT ----------------

with st.expander("About PHYTO-SCAN"):
    st.write(
        "PHYTO-SCAN uses a MobileNetV2-based deep learning model "
        "trained to classify 15 conditions across tomato, potato "
        "and bell pepper leaves."
    )

    st.write(
        "It combines image classification, model confidence "
        "and PHYTO-CARE guidance in one interface."
    )
