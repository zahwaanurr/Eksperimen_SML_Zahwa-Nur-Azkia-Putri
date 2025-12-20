import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    mlflow.autolog()

    df = pd.read_csv("heart_preprocessing.csv")

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("Accuracy:", acc)

if __name__ == "__main__":
    main()
