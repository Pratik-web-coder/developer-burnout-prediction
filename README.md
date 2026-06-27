# Developer Burnout Prediction System

## Overview

The **Developer Burnout Prediction System** is a Machine Learning web application that predicts the burnout risk level of software developers based on their work habits and lifestyle factors.

The application is built using **Python**, **Scikit-learn**, and **Streamlit**, providing an interactive interface where users can enter their details and instantly receive a burnout prediction.

---

## Features

* Predicts burnout risk as **Low**, **Medium**, or **High**
* Interactive Streamlit web interface
* User-friendly input sliders
* Displays prediction confidence for each burnout category
* Visual summary of user inputs
* Uses a trained Machine Learning model for prediction

---

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Joblib

---

## Project Structure

```
ml_project/
│── app.py
│── model.pkl
│── scaler.pkl
│── imputer.pkl
│── encoder.pkl
│── requirements.txt
│── README.md
```

---

## Input Features

The model predicts burnout using the following inputs:

* Age
* Experience Years
* Daily Work Hours
* Sleep Hours
* Caffeine Intake
* Bugs Per Day
* Commits Per Day
* Meetings Per Day
* Screen Time
* Exercise Hours
* Stress Level

---

## Machine Learning Pipeline

1. Data Collection
2. Data Preprocessing
3. Missing Value Handling using KNN Imputer
4. Feature Scaling using StandardScaler
5. Logistic Regression Model Training
6. Model Evaluation
7. Deployment using Streamlit

---

## How to Run the Project

### Install the required libraries

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
python -m streamlit run app.py
```

---

## Model Output

The application predicts one of the following burnout levels:

* Low Burnout Risk
* Medium Burnout Risk
* High Burnout Risk

It also displays the prediction confidence for each class.

---

## Future Improvements

* Improve model accuracy using ensemble learning techniques.
* Add more visualization dashboards.
* Deploy the application on Streamlit Cloud.
* Enable CSV file upload for batch predictions.

---

## Author

**Pratik**

This project was developed as a Machine Learning project to demonstrate predictive analytics using Python and Streamlit.
