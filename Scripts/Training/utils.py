import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score

def evaluate_model(model, X_t, y_t, name, reports_folder, is_nn=False, is_scaled=False):
    """Evaluates the model and logs performance metrics and confusion matrix."""
    if is_nn:
        probs = model.predict(X_t, verbose=0)
        preds = np.argmax(probs, axis=1)
    else:
        preds = model.predict(X_t)
        probs = model.predict_proba(X_t)
    
    print(f"\n--- {name} Evaluation ---")
    print(classification_report(y_t, preds))
    auc = roc_auc_score(y_t, probs, multi_class='ovr')
    print(f"ROC-AUC (OVR): {auc:.4f}")
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_t, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(reports_folder, exist_ok=True)
    plt.savefig(os.path.join(reports_folder, f"cm_{name.replace(' ', '_')}.png"))
    plt.close()
    
    return {'f1_macro': f1_score(y_t, preds, average='macro'), 'auc': auc}
