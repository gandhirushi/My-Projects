import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="FoodConnect", page_icon="🍴")

st.title("🍽️ FoodConnect")
st.subheader("Restaurant Rating Prediction & Smart Recommendation System")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ZomatoData.csv", encoding='utf-8')
    except:
        df = pd.read_csv("ZomatoData.csv", encoding='latin1')

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

df = load_data()

# -----------------------------
# Required Columns
# -----------------------------
required_cols = ['city','restaurant_name','cuisines','average_cost_for_two','votes','price_range','rating']

if not all(col in df.columns for col in required_cols):
    st.error("❌ Required columns missing")
    st.stop()

model_df = df[required_cols].dropna()

# -----------------------------
# Train Model
# -----------------------------
X = model_df[['average_cost_for_two', 'votes', 'price_range']]
y = model_df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# -----------------------------
# SIDEBAR (INPUTS + MODEL PERFORMANCE)
# -----------------------------
st.sidebar.header("🔍 Select Preferences")

# Inputs
city_list = sorted(model_df['city'].unique())
selected_city = st.sidebar.selectbox("City", city_list)

cuisine_df = model_df.copy()
cuisine_df['cuisines'] = cuisine_df['cuisines'].str.split(',')
cuisine_df = cuisine_df.explode('cuisines')
cuisine_df['cuisines'] = cuisine_df['cuisines'].str.strip()

cuisine_list = sorted(cuisine_df['cuisines'].unique())
selected_cuisine = st.sidebar.selectbox("Cuisine", cuisine_list)

cost = st.sidebar.slider("Cost for Two (₹)", 100, 5000, 1000)
votes = st.sidebar.slider("Votes", 0, 5000, 500)
price_range = st.sidebar.selectbox("Price Range", [1, 2, 3, 4])

predict_btn = st.sidebar.button("🔮 Predict")

st.sidebar.markdown("---")

# ✅ MODEL PERFORMANCE IN SIDEBAR
st.sidebar.header("📊 Model Performance")

y_pred = model.predict(X_test)

st.sidebar.write("R² Score:", round(r2_score(y_test, y_pred), 3))
st.sidebar.write("MAE:", round(mean_absolute_error(y_test, y_pred), 3))

# -----------------------------
# FILTER DATA
# -----------------------------
filtered_data = cuisine_df[
    (cuisine_df['city'] == selected_city) &
    (cuisine_df['cuisines'] == selected_cuisine)
]

# -----------------------------
# OUTPUT (MAIN SCREEN)
# -----------------------------
if predict_btn:

    if len(filtered_data) == 0:
        st.error("❌ No service available for selected city and cuisine")

    else:
        input_data = np.array([[cost, votes, price_range]])
        prediction = model.predict(input_data)[0]

        st.success(f"⭐ Predicted Rating: {round(prediction, 2)}")

        if prediction >= 4:
            st.info("🔥 Excellent choice!")
        elif prediction >= 3:
            st.info("👍 Good choice")
        else:
            st.warning("⚠️ Consider better options")

        st.divider()

        filtered_data = filtered_data.copy()
        filtered_data['difference'] = abs(filtered_data['rating'] - prediction)

        closest_matches = filtered_data.sort_values(by='difference').head(5)

        st.write("### 🎯 Best Matching Restaurants")
        st.dataframe(closest_matches[['restaurant_name', 'rating', 'average_cost_for_two']])

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("👨‍💻 Internship Project | FoodConnect")

