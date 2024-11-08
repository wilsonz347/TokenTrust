import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import time

def load_data():
    df = pd.read_csv("model/mail_data.csv")
    return df

def load_model():
    with open("model/log_model.pkl", "rb") as f:
        model = pickle.load(f)
        return model

def load_vectorizer():
    with open("model/vector.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        return vectorizer


def main():
    model = load_model()
    vectorizer = load_vectorizer()
    df = load_data()

    st.set_page_config(
        page_title="TokenTrust",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar for Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select a page:", ("Home", "Project Overview", "Data Analysis", "Disclaimer"))

    if page == "Home":
        # Header Section
        st.title("Spam Mail Predictor")
        st.write(
            "This application employs a machine learning algorithm combined with feature extraction to predict whether the content of an email is likely to be spam or ham.")

        # Main Content Area
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Email Content")
            email_content = st.text_area("Paste your email content here:", height=200)

            if st.button("Predict"):
                if email_content:
                    with st.spinner('Predicting...'):
                        time.sleep(2)

                        vectorized_email = vectorizer.transform([email_content])
                        prediction = model.predict(vectorized_email)[0]
                    if prediction == 0:
                        st.success("✅ The email is likely to be **spam**.")
                    else:
                        st.success("✅ The email is likely to be **ham**.")
                else:
                    st.warning("Please enter some email content to get a prediction.")

        with col2:
            st.subheader("About the Model")
            st.write("This model uses logistic regression to classify emails based on their content.")
            st.image("images/model_diagram.png", caption="Model Architecture", use_column_width=True)

    elif page == "Project Overview":
        # Project Overview Section
        st.write("### Project Timeline")
        st.markdown("""
        <div style="border-left: 2px solid #4CAF50; padding-left: 20px;">
            <h5>📥 Data Collection</h5>
            <p>Loading dataset from CSV.</p>
        </div>
        <div style="border-left: 2px solid #4CAF50; padding-left: 20px; margin-top: 20px;">
            <h5>🔍 Data Exploration</h5>
            <p>Checking for duplicates and null values.</p>
        </div>
        <div style="border-left: 2px solid #4CAF50; padding-left: 20px; margin-top: 20px;">
            <h5>⚙️ Data Preprocessing</h5>
            <p>Encoding categories and splitting data.</p>
        </div>
        <div style="border-left: 2px solid #4CAF50; padding-left: 20px; margin-top: 20px;">
            <h5>📊 Feature Extraction</h5>
            <p>Applying TF-IDF vectorization.</p>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Data Analysis":
        st.header("Data Analysis")
        st.write("### Data Preview")
        st.dataframe(df.head())

        # Basic Statistics
        st.write("### Basic Statistics")
        st.write(f"Total number of emails: {len(df)}")
        st.write(f"Number of spam emails: {df[df['Category'] == 'spam'].shape[0]}")
        st.write(f"Number of ham emails: {df[df['Category'] == 'ham'].shape[0]}")

        col1, col2 = st.columns(2)

        with col1:
            category_counts = df['Category'].value_counts()

            # Create the Plotly pie chart
            fig = px.pie(
                names=category_counts.index,
                values=category_counts.values,
                color_discrete_sequence=['#ff9999', '#66b3ff'],
                hole=0.4,  # This creates a donut chart
            )

            # Customize layout for transparency
            fig.update_layout(
                title_text='Distribution of Ham vs Spam Emails',
                title_font_size=20,
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot background
            )

            # Display the figure in Streamlit
            st.plotly_chart(fig)

        with col2:
            st.markdown("""
                <style>
                .scrollable-container {
                    height: 400px;
                    overflow-y: scroll;
                    border: 1px solid #ccc;
                    padding: 10px;
                    border-radius: 5px;
                }
                </style>
                """, unsafe_allow_html=True)

            sample_size = 5
            # Initialize session state for the sampled data
            if "samples" not in st.session_state:
                st.session_state.samples = df.sample(sample_size)

            # HTML for scrollable container
            scrollable_content = "<div class='scrollable-container'>"
            for _, row in st.session_state.samples.iterrows():
                scrollable_content += f"<p><strong>Category:</strong> {row['Category']}</p>"
                scrollable_content += f"<p><strong>Message:</strong> {row['Message'][:100]}...</p>"  # Display first 100 characters
                scrollable_content += "<hr>"
            scrollable_content += "</div>"

            if st.button("Refresh"):
                    st.session_state.samples = df.sample(sample_size)

            # Render the HTML inside Streamlit
            st.markdown(scrollable_content, unsafe_allow_html=True)


    elif page == "Disclaimer":
        # Disclaimer Section
        st.header("⚠️ Disclaimer")
        st.write(
            "This application is intended for educational purposes only and does not provide real-time spam detection capabilities.")
        st.write(
            "While machine learning models can assist in classifying emails as spam or ham based on historical data and patterns, please keep in mind the following:")

        st.markdown("""
        - **Accuracy Limitations**: The model may not always accurately reflect individual user preferences or contexts.
        - **User Judgment**: Users should exercise their judgment when interpreting classifications.
        - **Consider Personal Experience**: Always consider your own experiences with specific senders before acting on classifications.
        """)

if __name__ == '__main__':
    main()