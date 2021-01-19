"""
    Description : This Program detect diabetes using Machine Learning and Python
"""
# run block of code and catch warnings
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.write("""
# Diabetes Detection
Application of Python in Real-World Situations
""")

data = pd.read_csv("E:/Study Material/Python Programs/Diabetes Detection Using ML/diabetes.csv")

st.subheader("Data Information")

st.dataframe(data)

st.write(data.describe())

chart = st.bar_chart(data)

X = data.iloc[: , 0:8].values
Y = data.iloc[:, -1].values

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)


def get_user_input():
    pregnancies = st.sidebar.slider("Pregnancies",0,17,3)
    glucose = st.sidebar.slider("Glucose",0,199,117)
    blood_pressure = st.sidebar.slider("Blood Pressure",0,122,72)
    skin_thickness = st.sidebar.slider("Skin Thickness",0,99,23)
    insulin = st.sidebar.slider("Insulin",0.0,846.0,30.0)
    bmi = st.sidebar.slider("BMI",0.0,67.1,32.0)
    diabetes_pedigree_function = st.sidebar.slider("Diabetes Pedigree Function",0.078,2.42,0.3725)
    age = st.sidebar.slider("Age",0,200,29)

    user_data = {"pregnancies": pregnancies,
                 "glucose" : glucose,
                 "blood_pressure" : blood_pressure,
                 "skin_thickness" : skin_thickness,
                 "insulin" : insulin,
                 "bmi" : bmi,
                 "diabetes_pedigree_function" : diabetes_pedigree_function,
                 "age" : age
                 }
    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()

st.subheader("User Input :")
st.write(user_input)

model = RandomForestClassifier(n_estimators = 100)
model.fit(x_train,y_train)

#st.subheader("Model Test Accuracy Score:")
#st.write(str(accuracy_score(y_test) , model.predict(x_test))*100 + "%")

predictions = model.predict(user_input)

# final_predictions = str(predictions)
# if final_predictions == "0":
#     st.write("""
#             # Predictions :
#             Your test for diabetes is NEGATIVE.
#             """)
# if final_predictions == "1":
#     st.write("""
#             # Predictions :
#             Your test for diabetes is POSITIVE.
#             """)

st.subheader("Predictions :")
st.write(predictions)
