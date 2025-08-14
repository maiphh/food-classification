"""
Food Classification Streamlit App

A web application that allows users to upload food images and get predictions
from a trained MobileNetV2 model for 101 different food categories.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# Add the src directory to Python path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util import get_data_list  # noqa: E402


def load_model_robust(model_path: str) -> Optional[tf.keras.Model]:
    """Attempt to load a model using multiple fallback strategies.

    Returns the first successfully loaded model, or None if all strategies fail.
    """
    loading_strategies = [
        lambda path: tf.keras.models.load_model(path),
        lambda path: tf.keras.models.load_model(path, compile=False),
        lambda path: tf.keras.models.load_model(path, compile=False, safe_mode=False),
        lambda path: tf.keras.models.load_model(path, compile=False, custom_objects={"Functional": tf.keras.Model}),
    ]

    for i, strategy in enumerate(loading_strategies, start=1):
        try:
            model = strategy(model_path)
            # st.success(f"Successfully loaded model using strategy {i}")
            
            return model
        except Exception:  # noqa: BLE001 - broad to continue fallbacks
            continue
    return None


class FoodClassifierApp:
    """Encapsulates model loading, preprocessing, and prediction logic."""

    def __init__(self) -> None:
        self.model: Optional[tf.keras.Model] = None
        self.class_names: List[str] = []

    @st.cache_resource
    def load_model_and_classes(_self) -> Tuple[Optional[tf.keras.Model], List[str]]:  # noqa: D401
        """Load (and cache) the trained model and class names."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)

            model_files = [
                "converted_model.keras",
                "food101_mobilenetv2_finetuned.keras",
                "food101_mobilenetv2.keras",
                "best_model.h5",
            ]

            model: Optional[tf.keras.Model] = None

            for model_file in model_files:
                candidate_path = os.path.join(project_root, "notebook", model_file)
                if os.path.exists(candidate_path):
                    model = load_model_robust(candidate_path)
                    if model is not None:
                        break

            if model is None:
                return None, []

            classes_path = os.path.join(project_root, "data", "meta", "classes.txt")
            if not os.path.exists(classes_path):
                classes_path = os.path.join(script_dir, "classes.txt")
                if not os.path.exists(classes_path):
                    st.error("Classes file not found in any location")
                    return None, []

            class_names = get_data_list(classes_path)
            return model, class_names
        except Exception as e:  # noqa: BLE001
            st.error(f"Error loading model or classes: {e}")
            st.error("Please check that the model files exist in the 'notebook' directory")
            return None, []

    def preprocess_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Preprocess an image for prediction (resize, scale, batch dimension)."""
        try:
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array / 255.0
        except Exception as e:  # noqa: BLE001
            st.error(f"Error preprocessing image: {e}")
            return None

    def predict(self, image: Image.Image, top_k: int = 5) -> Optional[List[Tuple[str, float]]]:
        """Predict the top-k classes for an image."""
        if self.model is None:
            self.model, self.class_names = self.load_model_and_classes()

        if self.model is None:
            st.error("Model failed to load. Cannot make predictions.")
            return None
        if not self.class_names:
            st.error("Class names failed to load. Cannot make predictions.")
            return None

        img_array = self.preprocess_image(image)
        if img_array is None:
            return None

        try:
            with st.spinner("Making prediction..."):
                predictions = self.model.predict(img_array, verbose=0)[0]
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            return [(self.class_names[idx], float(predictions[idx])) for idx in top_indices]
        except Exception as e:  # noqa: BLE001
            st.error(f"Error making prediction: {e}")
            return None


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Food Classification App",
        page_icon="🍕",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🍕 Food Classification App")
    st.markdown(
        """
        Upload an image of food and get AI-powered predictions for what type of food it is!
        This app can classify **101 different types of food** using a fine-tuned MobileNetV2 model.
        """
    )

    classifier = FoodClassifierApp()

    st.sidebar.title("📋 Instructions")
    st.sidebar.markdown(
        """
        1. Upload a food image using the file uploader
        2. The app will automatically predict the food type
        3. View the top predictions with confidence scores
        4. Try different food images to test the model!
        """
    )

    st.sidebar.title("🍽️ Supported Food Types")
    st.sidebar.markdown(
        """
        The model can classify 101 different food categories including:
        - **Italian**: Pizza, Lasagna, Spaghetti, Ravioli
        - **Asian**: Sushi, Ramen, Pad Thai, Dumplings
        - **American**: Hamburger, Hot Dog, French Fries
        - **Desserts**: Ice Cream, Chocolate Cake, Donuts
        - **And many more!**
        """
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Food Image")
        uploaded_file = st.file_uploader(
            "Choose a food image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of food for best results",
        )
        top_k = st.slider(
            "Number of predictions to show:",
            min_value=1,
            max_value=10,
            value=5,
            help="Select how many top predictions to display",
        )

    with col2:
        st.subheader("🔍 Predictions")
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                # Center and scale the uploaded image preview so it appears smaller
                img_left, img_center, img_right = st.columns([1, 2, 1])
                with img_center:
                    st.image(image, caption="Uploaded Image", width=350)
                results = classifier.predict(image, top_k=top_k)
                if results:
                    st.subheader("🎯 Top Predictions")

                    # DataFrame preparation (kept logic identical for values)
                    df_results = pd.DataFrame(results, columns=["Food Type", "Confidence"])
                    df_results_display = df_results.copy()
                    df_results_display["Food Type"] = df_results_display["Food Type"].str.replace("_", " ").str.title()
                    df_results_display["Confidence"] = (df_results_display["Confidence"] * 100).round(2).astype(str) + "%"
                    df_results_display.index = range(1, len(df_results_display) + 1)

                    # Summary metrics
                    top_food = results[0][0].replace("_", " ").title()
                    top_confidence = results[0][1]
                    second_conf = results[1][1] if len(results) > 1 else 0
                    delta_vs_second = (top_confidence - second_conf) * 100

                    # Center the prediction statistics (metrics) block and give it more width
                    spacer_l, metrics_col, spacer_r = st.columns([1, 3, 1])
                    with metrics_col:
                        # Allocate more space to the first metric so long names don't truncate
                        mc1, mc2, mc3 = st.columns([2, 1, 1])
                        with mc1:
                            # Custom metric card to avoid truncation of long class names
                            st.markdown(
                                f"""
                                <div style='border:1px solid #e6e6e6; border-radius:8px; padding:12px; background:#fafafa; text-align:center;'>
                                    <div style='font-size:0.9rem; color:#666; margin-bottom:6px;'>Top Prediction</div>
                                    <div style='font-size:1.1rem; font-weight:700; color:#111; word-break:break-word;' title='{top_food}'>
                                        {top_food}
                                    </div>
                                    <div style='font-size:0.85rem; color:#2e7d32; margin-top:6px;'>+{delta_vs_second:.2f}% vs #2</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with mc2:
                            st.metric("Confidence", f"{top_confidence * 100:.2f}%")
                        with mc3:
                            st.metric("Classes Evaluated", str(top_k))

                    # Display table (styled)
                    st.dataframe(
                        df_results_display,
                        use_container_width=True,
                        hide_index=False,
                    )

                    # Visual confidence distribution (Altair)
                    try:
                        import altair as alt  # noqa: WPS433 (third-party import inside block for optional dep)

                        chart_df = df_results.copy()
                        chart_df["Food Type"] = chart_df["Food Type"].str.replace("_", " ").str.title()
                        chart_df = chart_df.sort_values("Confidence", ascending=True)
                        bar_color = alt.condition(
                            alt.datum["Food Type"] == top_food,
                            alt.value("#ff4b4b"),  # highlight top
                            alt.value("#1f77b4"),
                        )
                        chart = (
                            alt.Chart(chart_df)
                            .mark_bar(size=18)
                            .encode(
                                x=alt.X("Confidence:Q", axis=alt.Axis(format=".0%", title="Confidence")),
                                y=alt.Y("Food Type:N", sort=None, title=""),
                                color=bar_color,
                                tooltip=[
                                    alt.Tooltip("Food Type:N"),
                                    alt.Tooltip("Confidence:Q", format=".2%"),
                                ],
                            )
                        )
                        text = chart.mark_text(
                            align="left", baseline="middle", dx=5, color="#222"
                        ).encode(text=alt.Text("Confidence:Q", format=".2%"))
                        st.subheader("📊 Confidence Distribution")
                        st.altair_chart(
                            (chart + text).properties(
                                height=30 * len(chart_df),
                                padding={"left": 5, "right": 5, "top": 5, "bottom": 5},
                            ),
                            use_container_width=True,
                        )
                    except Exception:  # noqa: BLE001 - chart is optional
                        st.info("Altair visualization unavailable.")

                    # Progress style (retain original feel for top 3 inside expander)
                    with st.expander("Top 3 Progress View", expanded=False):
                        for i, (food_name, confidence) in enumerate(results[:3]):
                            formatted_name = food_name.replace("_", " ").title()
                            st.write(f"**{i + 1}. {formatted_name}** — {confidence * 100:.2f}%")
                            st.progress(confidence)

                    # Raw probabilities expander
                    with st.expander("Raw Prediction Data"):
                        st.dataframe(
                            df_results.assign(Probability=df_results.Confidence).drop(columns=["Confidence"]),
                            hide_index=True,
                            use_container_width=True,
                        )
                        csv = df_results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download CSV",
                            data=csv,
                            file_name="food_predictions.csv",
                            mime="text/csv",
                        )

                    # Outcome message (same thresholds)
                    if top_confidence * 100 > 50:
                        st.success(f"🎉 Prediction: {top_food} ({top_confidence * 100:.2f}%)")
                    elif top_confidence * 100 > 25:
                        st.warning(f"🤔 Likely: {top_food} ({top_confidence * 100:.2f}%)")
                    else:
                        st.info(f"🤷 Possible: {top_food} ({top_confidence * 100:.2f}%)")

                    st.toast("Prediction complete", icon="✅")
                else:
                    st.error("Failed to make prediction. Please try again with a different image.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Error processing image: {e}")
        else:
            st.info("👆 Please upload a food image to get started!")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🤖 Powered by TensorFlow and MobileNetV2 | Built with Streamlit</p>
            <p>📊 Trained on the Food-101 dataset with 101 food categories</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
