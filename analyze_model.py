import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditriskmodelling.utils.main_utils.utils import load_object, load_numpy_array_data


# ===============================
# 1️⃣ Load trained model
# ===============================
model = load_object(
    "D:\\Credit Risk Modelling\\final_model\\model.pkl"
)


# ===============================
# 2️⃣ Load transformed training data
# ===============================
train_arr = load_numpy_array_data(
    "D:\\Credit Risk Modelling\\Artifacts\\26_03_14_21_16_25\\data_transformation\\transformed\\train.npy"
)

X_train = train_arr[:, :-1]
y_train = train_arr[:, -1]


# ===============================
# 3️⃣ Load feature names
# ===============================
try:
    feature_names = load_object(
    "D:\\Credit Risk Modelling\\final_model\\feature_name.pkl"
)

    print("\nSelected Features:")
    for f in feature_names:
        print(f)

except Exception:
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    print("\nFeature names not saved, using indices.")


# ===============================
# 4️⃣ Feature Importance
# ===============================
if hasattr(model.model, "feature_importances_"):
    importance = model.model.feature_importances_

elif hasattr(model.model, "coef_"):
    importance = abs(model.model.coef_[0])

else:
    raise Exception("Model type does not support feature importance")


importance = pd.Series(
    importance,
    index=feature_names
).sort_values(ascending=False)


print("\nTop Important Features:")
print(importance.head(10))


# ===============================
# 5️⃣ Single Feature Leakage Check
# ===============================
print("\nChecking if any single feature predicts the target...")

for i, col in enumerate(feature_names):
    try:
        score = roc_auc_score(y_train, X_train[:, i])

        if score > 0.9:
            print(f"{col} alone predicts the target with AUC {score:.4f}")

    except Exception:
        pass


# ===============================
# 6️⃣ Final Leakage Test
# ===============================
print("\nRunning final leakage test...")

y_shuffled = np.random.permutation(y_train)

preds = model.model.predict_proba(X_train)[:, 1]

auc = roc_auc_score(y_shuffled, preds)

print("AUC after shuffling target:", auc)
