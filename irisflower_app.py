
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Load dataset
# ----------------------------
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classification")
st.write("Enter flower measurements and classify the species.")

# User input sliders
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Prediction
if st.button("Classify"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = clf.predict(input_data)[0]
    st.success(f"🌼 Predicted species: **{target_names[prediction]}**")