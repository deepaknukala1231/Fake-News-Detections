
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "fake_news_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📰 Fake News Detection System")
st.write("Enter a news headline and article to predict whether the news resembles fake or real news in the training dataset.")

st.warning(
    "This is a machine-learning prediction, not a verified fact-check."
)

title = st.text_input("News Headline")
article = st.text_area(
    "News Article",
    height=220,
    placeholder="Paste the news article here..."
)

if st.button("Detect News", type="primary"):
    if not title.strip() and not article.strip():
        st.warning("Please enter a headline or article.")
    else:
        try:
            model = load_model()
            news_text = (title + " " + article).strip()
            prediction = int(model.predict([news_text])[0])

            st.subheader("Prediction Result")

            if prediction == 1:
                st.success("Prediction: Likely Real")
            else:
                st.error("Prediction: Likely Fake")

            st.caption(
                "Verify important claims with reliable news sources."
            )

        except FileNotFoundError:
            st.error(
                "Model file not found. Check that fake_news_model.pkl "
                "is in the same folder as app.py."
            )
        except Exception as error:
            st.error(f"Error: {error}")

