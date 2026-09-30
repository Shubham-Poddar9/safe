
import io
import requests
import streamlit as st
from huggingface_hub import InferenceClient


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Safe AI Image Generator",
    page_icon="🖼️"
)


# ============================================================
# HUGGING FACE
# ============================================================

HF_API_KEY = st.secrets["HF_API_KEY"]

if not HF_API_KEY:
    st.error("❌ HF_API_KEY is missing from Streamlit secrets.")
    st.stop()


client = InferenceClient(
    provider="auto",
    api_key=HF_API_KEY
)


# ============================================================
# IMAGE MODELS
# ============================================================

MODELS = {
    "SDXL Lightning": "ByteDance/SDXL-Lightning",
    "SDXL Base": "stabilityai/stable-diffusion-xl-base-1.0",
    "SDXL Turbo": "stabilityai/sdxl-turbo",
    "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5"
}


# ============================================================
# SAFETY FILTER
# ============================================================

FILTER_API = "https://filters-zeta.vercel.app/api/filter"


def check_prompt(prompt):
    try:
        response = requests.post(
            FILTER_API,
            json={"prompt": prompt},
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        return (
            data.get("ok", False),
            data.get("reason", "")
        )

    except Exception as e:
        return False, str(e)


# ============================================================
# UI
# ============================================================

st.title("🖼️ Safe AI Image Generator")

st.write(
    "Enter a description, choose an image model, "
    "and generate an image."
)


# Model selection
model_name = st.selectbox(
    "🎨 Image Model",
    list(MODELS.keys())
)

model = MODELS[model_name]


# Prompt
prompt = st.text_area(
    "📝 Image Description",
    placeholder="A beautiful sunset over the mountains...",
    height=120
)


# Generate
if st.button(
    "🎨 Generate Image",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Check empty prompt
    # --------------------------------------------------------

    if not prompt.strip():
        st.warning("Please enter an image description.")
        st.stop()


    prompt = prompt.strip()


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    with st.spinner("🔍 Checking prompt..."):

        ok, reason = check_prompt(prompt)


    if not ok:
        st.error(
            "⚠️ Prompt blocked: "
            + (reason or "Prompt did not pass safety check.")
        )
        st.stop()


    # --------------------------------------------------------
    # Generate image
    # --------------------------------------------------------

    with st.spinner(
        f"🎨 Generating with {model_name}..."
    ):

        try:

            image = client.text_to_image(
                prompt=prompt,
                model=model
            )

        except Exception as e:

            st.error("❌ Image generation failed.")

            st.code(
                str(e),
                language="text"
            )

            st.info(
                "Try another image model from the dropdown. "
                "The selected model may not currently have "
                "an enabled Inference Provider."
            )

            st.stop()


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    st.subheader("🖼️ Generated Image")

    st.image(
        image,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        "📥 Download Image",
        data=buffer.getvalue(),
        file_name="ai_image.png",
        mime="image/png",
        use_container_width=True
    )

