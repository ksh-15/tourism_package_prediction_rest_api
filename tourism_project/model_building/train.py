import os
import joblib
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def main():
    # 1. Load the prepared splits
    train_path = "tourism_project/model_building/train.csv"
    test_path = "tourism_project/model_building/test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Prepared train/test CSV files not found. Please run prep.py first!")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = 'ProdTaken'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # 2. Setup Preprocessing Pipeline
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # Set up MLflow experiment tracking
    mlflow.set_experiment("Tourism_Package_Prediction")

    with mlflow.start_run() as run:
        print("Starting Model Training and Tuning with MLflow tracking...")

        # 3. Create a parameter grid for XGBoost tuning
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb_model)
        ])

        param_grid = {
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.05, 0.1, 0.2],
            'classifier__n_estimators': [50, 100, 150]
        }

        # Hyperparameter search
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimatorfg = grid_search.best_estimator_

        # 4. Logging parameters & performance metrics to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score_auc", grid_search.best_score_)

        # Evaluate model performance on validation set
        preds = best_pipeline.predict(X_test)
        probs = best_pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        print("\n--- Optimized Model Evaluation Results ---")
        print(classification_report(y_test, preds))
        print(f"Test ROC AUC Score: {roc_auc:.4f}")

        # 5. Save the trained pipeline locally as joblib artifact
        model_dir = "tourism_project/model_building"
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(best_pipeline, model_path)
        print(f"\nSaved optimized model pipeline to: {model_path}")

        # Log final joblib artifact and sklearn model object within MLflow (using pickle fallback to avoid skops trust errors)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(best_pipeline, name="model", serialization_format="pickle")

        print("MLflow experiment logging completed.")

if __name__ == '__main__':
    main()
