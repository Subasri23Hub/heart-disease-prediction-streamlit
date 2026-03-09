❤️ Heart Disease Prediction using Machine Learning

An end-to-end Machine Learning application that predicts the risk of heart disease based on patient health parameters.
The project includes a Logistic Regression ML pipeline and an interactive Streamlit dashboard for real-time prediction and visualization.

Features:

Machine Learning pipeline using Scikit-learn

Logistic Regression model with hyperparameter tuning

Interactive Streamlit web application

Patient health input via sidebar

Heart disease risk prediction

Risk probability visualization

Correlation heatmap and data insights

Downloadable patient report

Machine Learning Workflow:

Data loading and preprocessing

Encoding categorical variables

Train-test split

Pipeline creation using:

StandardScaler

LogisticRegression

Hyperparameter tuning using GridSearchCV

Model evaluation using:

Accuracy score

Classification report

Confusion matrix

Model deployment using Streamlit

Dataset:

The dataset contains patient health parameters used to predict heart disease.

Features

Age

Sex

ChestPainType

RestingBP

Cholesterol

FastingBS

RestingECG

MaxHR

ExerciseAngina

Oldpeak

ST_Slope

Target

HeartDisease

1 → Presence of heart disease

0 → No heart disease

Streamlit Dashboard:

The interactive dashboard allows users to:

Input patient health parameters

Predict heart disease risk

View prediction probability

Analyze dataset visualizations

Download patient health report

Tech Stack:

Python

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn

Plotly

Streamlit

Project Structure:
heart-disease-prediction-streamlit
│
├── app.py                # Streamlit dashboard
├── miniproject.py        # ML pipeline training script
├── heart.csv             # Dataset
├── heart_model.pkl       # Trained ML model
├── requirements.txt      # Dependencies
├── .gitignore
└── README.md
▶️ Running the Application
1️⃣ Clone the repository
git clone https://github.com/Subasri23Hub/heart-disease-prediction-streamlit.git
2️⃣ Navigate to the project folder
cd heart-disease-prediction-streamlit
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the Streamlit app
streamlit run app.py

Future Improvements:

Deploy the application using Streamlit Cloud

Add model comparison (Random Forest, XGBoost)

Implement feature importance visualization

Add user authentication for medical dashboards

Author

Subasri

GitHub:
https://github.com/Subasri23Hub

⭐ If you like this project

Consider giving it a star ⭐ on GitHub.
