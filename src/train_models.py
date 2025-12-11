import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sentence_transformers import SentenceTransformer


# Dossier src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

QCM_CSV_PATH = os.path.join(DATA_DIR, "16P_converted.csv")
TEXT_CSV_PATH = os.path.join(DATA_DIR, "MBTI_500.csv")

QCM_MODEL_PATH = os.path.join(MODELS_DIR, "qcm_model_rf.joblib")
QCM_COLUMNS_PATH = os.path.join(MODELS_DIR, "qcm_columns.joblib")

TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_model.joblib")

SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"

# Entraînement modèle TEXTE (MBTI_500.csv)
def train_text_model():
    print(f"🔹 Chargement de la base TEXTE : {TEXT_CSV_PATH}")
    df = pd.read_csv(TEXT_CSV_PATH)
    text_col = "posts"
    target_col = "type"

    if text_col not in df.columns:
        raise ValueError(
            f"La colonne texte '{text_col}' n'existe pas dans MBTI_500.csv.\n"
            f"Colonnes trouvées : {list(df.columns)}"
        )
    if target_col not in df.columns:
        raise ValueError(
            f"La colonne cible '{target_col}' n'existe pas dans MBTI_500.csv.\n"
            f"Colonnes trouvées : {list(df.columns)}"
        )

    print(f"[TEXTE] Colonne texte : {text_col}")
    print(f"[TEXTE] Colonne cible MBTI : {target_col}")
    X_text = df[text_col].astype(str).fillna("")
    y = df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Chargement du SentenceTransformer : {SENTENCE_MODEL_NAME}")
    st_model = SentenceTransformer(SENTENCE_MODEL_NAME)

    print(" Encodage des textes (train)...")
    X_train_emb = st_model.encode(
        X_train.tolist(),
        batch_size=64,
        show_progress_bar=True
    )

    print(" Encodage des textes (test)...")
    X_test_emb = st_model.encode(
        X_test.tolist(),
        batch_size=64,
        show_progress_bar=True
    )

    # Classifieur sur embeddings : LogisticRegression
    clf = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        solver="lbfgs"
    )

    print(" Entraînement du modèle TEXTE (LogisticRegression)...")
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)
    print(" Rapport de classification (TEXTE) :")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, TEXT_MODEL_PATH)
    print(f" text_model.joblib sauvegardé → {TEXT_MODEL_PATH}")


# Entraînement modèle QCM (16P_converted.csv) avec RandomForest
def train_qcm_model_rf():
    print(f" Chargement de la base QCM : {QCM_CSV_PATH}")
    df = pd.read_csv(QCM_CSV_PATH)
    target_col = "Personality"
    if target_col not in df.columns:
        raise ValueError(
            f"La colonne '{target_col}' n'existe pas dans 16P_converted.csv.\n"
            f"Colonnes trouvées : {list(df.columns)}"
        )

    drop_cols = ["Response Id", target_col]
    qcm_cols = [c for c in df.columns if c not in drop_cols]
    if len(qcm_cols) == 0:
        raise ValueError(
            "Aucune colonne QCM trouvée après exclusion de 'Response Id' et 'Personality'."
        )
    print(f"[QCM-RF] Colonnes QCM ({len(qcm_cols)}) : {qcm_cols[:10]}{' ...' if len(qcm_cols) > 10 else ''}")
    print(f"[QCM-RF] Colonne cible MBTI : {target_col}")
    X = df[qcm_cols].copy()
    y = df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    print(" Entraînement du modèle QCM (RandomForest)...")
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(" Rapport de classification (QCM - RandomForest) :")
    print(classification_report(y_test, y_pred))

    joblib.dump(rf, QCM_MODEL_PATH)
    joblib.dump(qcm_cols, QCM_COLUMNS_PATH)

    print(f" qcm_model_rf.joblib sauvegardé → {QCM_MODEL_PATH}")
    print(f" qcm_columns.joblib sauvegardé → {QCM_COLUMNS_PATH}")


if __name__ == "__main__":
    train_text_model()
    train_qcm_model_rf()
