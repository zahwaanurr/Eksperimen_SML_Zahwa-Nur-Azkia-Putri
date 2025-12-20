#Mengimport Library
import os
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

#Men-setting MLFlow 
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Students Performance Classification")

#Load file yang sudah di preprocessing
base_dir = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(base_dir, "StudentsPerformance_preprocessing.joblib")

file_splitdata = load(database_path)

X_train = file_splitdata["X_train"]
X_test = file_splitdata["X_test"]
y_train = file_splitdata["y_train"]
y_test = file_splitdata["y_test"]

#Mengambil input data untuk Model Registry
input_data = X_train[0:5]            

#Membuat MLFlow 
with mlflow.start_run():
    n_estimators = 200
    max_depth = 10
    random_state = 42

    #Autolog
    mlflow.sklearn.autolog()

    #Train model Random Forest
    rforest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    rforest.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(
    sk_model=rforest,
    artifact_path="model",
    input_example=input_data
    )

    #Menentukan log metrics
    y_pred = rforest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
