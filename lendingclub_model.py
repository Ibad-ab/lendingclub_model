
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from pdpbox import pdp, info_plots
import warnings
from sklearn.inspection import PartialDependenceDisplay


warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('credit_card_2015_2016.csv')
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df['Outcome'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
df_2015 = df[df['issue_y'] == 15].copy()
df_2016 = df[df['issue_y'] == 16].copy()

# Features to use
features = ['term', 'sub_grade', 'grade', 'emp_length', 'revol_util_n', 'int_rate_n']
df_2015 = df_2015[features + ['Outcome']].dropna()
df_2016 = df_2016[features + ['Outcome']].dropna()

cat_cols = ['term', 'sub_grade', 'grade', 'emp_length']


# Encode categoricals
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([df_2015[col], df_2016[col]], axis=0)
    le.fit(all_values)
    df_2015[col] = le.transform(df_2015[col])
    df_2016[col] = le.transform(df_2016[col])

# Split features and target
X_train, y_train = df_2015.drop('Outcome', axis=1), df_2015['Outcome']
X_oot, y_oot = df_2016.drop('Outcome', axis=1), df_2016['Outcome']

# Gini function
def gini_score(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1

# Train model with repeated CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
gini_scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv)
gini_scores = 2 * gini_scores - 1  # Convert AUC to Gini

# Final model training
model.fit(X_train, y_train)
y_train_pred = model.predict_proba(X_train)[:, 1]
y_oot_pred = model.predict_proba(X_oot)[:, 1]

# ROC Curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
fpr_oot, tpr_oot, _ = roc_curve(y_oot, y_oot_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label='Train ROC')
plt.plot(fpr_oot, tpr_oot, label='OOT ROC')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()


# PDP Plot for top feature
top_feat = X_train.columns[np.argmax(model.feature_importances_)]

from sklearn.inspection import PartialDependenceDisplay

# Choose top feature based on importance
top_feat = X_train.columns[np.argmax(model.feature_importances_)]

# Plot PDP + ICE using sklearn
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=[top_feat],
    kind="both",  # PDP + ICE
    ax=ax
)
plt.title(f'PDP + ICE Plot for {top_feat}')
plt.savefig('pdp_plot.png')
plt.close()


# Business-based threshold analysis
thresholds = np.linspace(0, 1, 100)
profits = []
for t in thresholds:
    preds = y_oot_pred > t
    TP = ((preds == 0) & (y_oot == 1)).sum()  # False negatives (missed bad loans)
    TN = ((preds == 0) & (y_oot == 0)).sum()  # True negatives (approved good loans)
    profit = TN * 300 - TP * 1000
    profits.append(profit)

opt_threshold = thresholds[np.argmax(profits)]

# Save profit plot
plt.figure(figsize=(8, 6))
plt.plot(thresholds, profits)
plt.axvline(opt_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {opt_threshold:.2f}')
plt.title('Profit Curve')
plt.xlabel('Threshold')
plt.ylabel('Expected Profit')
plt.legend()
plt.savefig('profit_curve.png')
plt.close()

# Save Gini stats
with open("gini_stats.txt", "w") as f:
    f.write(f"Average Gini: {np.mean(gini_scores):.4f}\n")
    f.write(f"Std Dev Gini: {np.std(gini_scores):.4f}\n")
    f.write(f"Optimal Threshold: {opt_threshold:.2f}\n")
