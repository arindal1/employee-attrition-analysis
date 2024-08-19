from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_models(smote_train, smote_target, test, target_val):
    rf, gb, xgb_model = train_models(smote_train, smote_target, test, target_val)
    
    rf_predictions = rf.predict(test)
    print("Random Forest Accuracy score: {}".format(accuracy_score(target_val, rf_predictions)))
    print("="*80)
    print(classification_report(target_val, rf_predictions))
    
    gb_predictions = gb.predict(test)
    print("Gradient Boosting Accuracy score: {}".format(accuracy_score(target_val, gb_predictions)))
    print("="*80)
    print(classification_report(target_val, gb_predictions))
    
    xgb_predictions = xgb_model.predict(test)
    print("XGBoost Accuracy score: {}".format(accuracy_score(target_val, xgb_predictions)))
    print("="*80)
    print(classification_report(target_val, xgb_predictions))
    
    def plot_roc_curve(y_true, y_scores, label=None):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.figure(figsize=(10, 8))
    plot_roc_curve(target_val, rf.predict_proba(test)[:, 1], 'Random Forest')
    plot_roc_curve(target_val, gb.predict_proba(test)[:, 1], 'Gradient Boosting')
    plot_roc_curve(target_val, xgb_model.predict_proba(test)[:, 1], 'XGBoost')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
