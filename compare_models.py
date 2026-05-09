# compare_models.py
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_data
from model import OptimizedELM


X, y = load_data('./CRWU')

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# ELM
# =========================
start = time.time()

elm = OptimizedELM(n_hidden=1000)
elm.fit(X_train, y_train)

pred = elm.predict(X_test)

end = time.time()

print(
    f'ELM: Acc={accuracy_score(y_test,pred):.4f}, '
    f'F1={f1_score(y_test,pred,average="weighted"):.4f}, '
    f'Time={end-start:.4f}s'
)


# =========================
# SVM
# =========================
start = time.time()

svm = SVC()
svm.fit(X_train, y_train)

pred = svm.predict(X_test)

end = time.time()

print(
    f'SVM: Acc={accuracy_score(y_test,pred):.4f}, '
    f'F1={f1_score(y_test,pred,average="weighted"):.4f}, '
    f'Time={end-start:.4f}s'
)


# =========================
# RF
# =========================
start = time.time()

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

end = time.time()

print(
    f'RF: Acc={accuracy_score(y_test,pred):.4f}, '
    f'F1={f1_score(y_test,pred,average="weighted"):.4f}, '
    f'Time={end-start:.4f}s'
)