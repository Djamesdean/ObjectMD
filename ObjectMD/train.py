# train_xgboost_model.py
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load feature dataset
DATA_PATH = "data/features/features.csv"
df = pd.read_csv(DATA_PATH)

# Define features and label
FEATURE_COLS = [
    "box_velocity",
    "avg_hand_box_distance"]
    
LABEL_COL = "is_moving"

# Clean dataset
df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
X = df[FEATURE_COLS]
y = df[LABEL_COL]

# Compute scale_pos_weight to balance classes manually
pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"Calculated scale_pos_weight: {pos_weight:.2f}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define and train model
clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=pos_weight,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🧱 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, "models/box_movement_xgb.pkl")
print("\n✅ Model saved to models/box_movement_xgb.pkl")

# Plot feature importance
plt.figure(figsize=(6, 3))
plt.barh(FEATURE_COLS, clf.feature_importances_, color='gold')
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("models/xgb_feature_importance.png")
plt.show()