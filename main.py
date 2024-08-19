from data.data_preprocessing import load_data, preprocess_data
from data.model_training import train_random_forest, train_gradient_boosting, train_xgboost
from data.model_evaluation import evaluate_model

def main():
    df = load_data('employee.csv')
    smote_train, smote_target, X_test, y_test = preprocess_data(df)
    
    rf_model = train_random_forest(smote_train, smote_target)
    gb_model = train_gradient_boosting(smote_train, smote_target)
    xgb_model = train_xgboost(smote_train, smote_target)
    
    print("Random Forest Model Evaluation:")
    evaluate_model(rf_model, X_test, y_test)
    
    print("Gradient Boosting Model Evaluation:")
    evaluate_model(gb_model, X_test, y_test)
    
    print("XGBoost Model Evaluation:")
    evaluate_model(xgb_model, X_test, y_test)

if __name__ == "__main__":
    main()
