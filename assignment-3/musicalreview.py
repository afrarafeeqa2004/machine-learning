import nltk
nltk.download('punkt_tab')
import pandas as pd
import numpy as np
import re
import string
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
df = pd.read_csv("Musical_instruments_reviews.csv")
df.dropna(subset=['reviewText', 'overall'], inplace=True)


stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

start_time = time.time()
df['cleaned_review'] = df['reviewText'].apply(clean_text)
print(f"Cleaning complete in {time.time()-start_time:.2f}s")
#TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['cleaned_review'])
print("TF-IDF shape:", X_tfidf.shape)

df['overall_int'] = df['overall'].astype(int)
y_clf = df['overall_int'] - 1
y_reg = df['overall']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_tfidf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_tfidf, y_reg, test_size=0.2, random_state=42
)
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=200, n_jobs=-1),
    "Naive Bayes": MultinomialNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
}

results_clf = []
best_acc = 0

print("\nClassification")
for name, model in classifiers.items():
    start = time.time()
    model.fit(X_train_clf, y_train_clf)
    y_pred = model.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred)
    prec = precision_score(y_test_clf, y_pred, average='macro')
    rec = recall_score(y_test_clf, y_pred, average='macro')
    f1 = f1_score(y_test_clf, y_pred, average='macro')
    results_clf.append([name, acc, prec, rec, f1])
    print(f"{name}: Acc={acc:.3f}, F1={f1:.3f}, Time={time.time()-start:.2f}s")
    if acc > best_acc:
        best_acc = acc
        best_clf_model = model
        best_clf_name = name

results_df_clf = pd.DataFrame(results_clf, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
print("\nClassification Results:\n", results_df_clf)
cm = confusion_matrix(y_test_clf, best_clf_model.predict(X_test_clf))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_clf_name}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.close()

mnb_model = MultinomialNB()
mnb_model.fit(X_train_clf, y_train_clf)
y_score = mnb_model.predict_proba(X_test_clf)
y_test_dummies = pd.get_dummies(y_test_clf).values

plt.figure(figsize=(10,8))
for i in range(y_test_dummies.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i+1} (AUC={roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Multi-class)')
plt.legend()
plt.show()
plt.close()
regressors = {
    "Linear Regression": LinearRegression(n_jobs=-1),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
    "XGBRegressor": XGBRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
}

results_reg = []
best_r2 = -float('inf')

print("\nRegression")
for name, model in regressors.items():
    start = time.time()
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred)
    results_reg.append([name, mse, rmse, r2])
    print(f"{name}: R2={r2:.3f}, RMSE={rmse:.3f}, Time={time.time()-start:.2f}s")
    if r2 > best_r2:
        best_r2 = r2
        best_reg_model = model
        best_reg_name = name

results_df_reg = pd.DataFrame(results_reg, columns=["Model", "MSE", "RMSE", "R2"])
print("\nRegression Results:\n", results_df_reg)

y_pred_best = best_reg_model.predict(X_test_reg)
residuals = y_test_reg - y_pred_best
plt.figure(figsize=(8,6))
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.hlines(0, y_pred_best.min(), y_pred_best.max(), colors='r', linestyles='--')
plt.title(f'Residual Plot - {best_reg_name}')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()
plt.close()
