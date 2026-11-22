import streamlit as st
from src.predict import predict_news

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    color: #ffffff;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #a5b4fc;
    font-size: 17px;
    margin-bottom: 35px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    margin-top: 20px;
}

/* White background and black text for input boxes */

.stTextInput input,
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border: 1px solid #cccccc !important;
    border-radius: 8px !important;
}

/* Placeholder text */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #666666 !important;
    -webkit-text-fill-color: #666666 !important;
    opacity: 1 !important;
}

.fake {
    background-color: #451a1a;
    color: #fca5a5;
}

.genuine {
    background-color: #123524;
    color: #86efac;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🔍 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered news classification using Machine Learning and NLP</div>',
    unsafe_allow_html=True
)

st.divider()

# Input section
st.subheader("📰 Analyze a News Article")

headline = st.text_input(
    "News Headline",
    placeholder="Enter the news headline..."
)

article = st.text_area(
    "News Article",
    placeholder="Paste the complete news article here...",
    height=220
)

# Prediction button
if st.button("🔍 Analyze News", use_container_width=True):

    if not headline.strip() or not article.strip():
        st.warning("Please enter both the headline and news article.")

    else:
        try:
            with st.spinner("Analyzing news..."):

                result = predict_news(headline + " " + article)

            st.divider()
            st.subheader("Prediction Result")

            if result == "fake":
                st.markdown(
                    '<div class="result-box fake">⚠️ FAKE NEWS DETECTED</div>',
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    '<div class="result-box genuine">✅ GENUINE NEWS</div>',
                    unsafe_allow_html=True
                )

        except Exception as error:
            st.error(f"Prediction error: {error}")

# Footer
st.divider()

st.markdown(
    "<center>Powered by TF-IDF & Passive Aggressive Classifier"
    "<br>Model Test Accuracy: 95.96%</center>",
    unsafe_allow_html=True
)

st.caption(
    "Disclaimer: Predictions are model classifications, "
    "not verification of the actual truth of a news story."
)