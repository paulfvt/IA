import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.json")

QCM_MODEL_PATH = os.path.join(MODELS_DIR, "qcm_model_rf.joblib")
QCM_COLUMNS_PATH = os.path.join(MODELS_DIR, "qcm_columns.joblib")
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_model.joblib")


# Chargement des ressources

@st.cache_data
def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]

@st.cache_resource
def load_models():
    # RandomForest pour le QCM
    qcm_model = joblib.load(QCM_MODEL_PATH)
    qcm_columns = joblib.load(QCM_COLUMNS_PATH)

    # Modèle texte (LogisticRegression sur embeddings)
    text_model = joblib.load(TEXT_MODEL_PATH)

    # SentenceTransformer pour encoder les réponses ouvertes
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return qcm_model, qcm_columns, text_model, text_encoder

questions = load_questions()
qcm_model, qcm_columns, text_model, text_encoder = load_models()

MBTI_TYPES = [
    "ENFJ", "ENFP", "ENTJ", "ENTP",
    "ESFJ", "ESFP", "ESTJ", "ESTP",
    "INFJ", "INFP", "INTJ", "INTP",
    "ISFJ", "ISFP", "ISTJ", "ISTP",
]


def mbti_to_axes_probs(probas_per_type, mbti_labels):
    """
    proba par type (array de taille 16) -> pourcentage par axe :
    E/I, S/N, T/F, J/P
    """
    letter_indices = {t: i for i, t in enumerate(mbti_labels)}

    totals = {
        "E": 0.0, "I": 0.0,
        "S": 0.0, "N": 0.0,
        "T": 0.0, "F": 0.0,
        "J": 0.0, "P": 0.0,
    }

    # Somme des probas par lettre
    for t, p in zip(mbti_labels, probas_per_type):
        if len(t) != 4:
            continue
        totals[t[0]] += p  # E/I
        totals[t[1]] += p  # S/N
        totals[t[2]] += p  # T/F
        totals[t[3]] += p  # J/P

    axes = {}
    ei_sum = totals["E"] + totals["I"]
    if ei_sum > 0:
        axes["E"] = totals["E"] / ei_sum
        axes["I"] = totals["I"] / ei_sum
    else:
        axes["E"] = axes["I"] = 0.5


    sn_sum = totals["S"] + totals["N"]
    if sn_sum > 0:
        axes["S"] = totals["S"] / sn_sum
        axes["N"] = totals["N"] / sn_sum
    else:
        axes["S"] = axes["N"] = 0.5


    tf_sum = totals["T"] + totals["F"]
    if tf_sum > 0:
        axes["T"] = totals["T"] / tf_sum
        axes["F"] = totals["F"] / tf_sum
    else:
        axes["T"] = axes["F"] = 0.5


    jp_sum = totals["J"] + totals["P"]
    if jp_sum > 0:
        axes["J"] = totals["J"] / jp_sum
        axes["P"] = totals["P"] / jp_sum
    else:
        axes["J"] = axes["P"] = 0.5

    return axes


# Config Streamlit

st.set_page_config(page_title="MBTI Fusion – QCM + Texte", layout="wide")
st.title(" MBTI Fusion – QCM + Texte")


st.sidebar.header("⚙️ Paramètres")

weight_qcm = st.sidebar.slider(
    "Poids du QCM",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05
)
weight_text = 1.0 - weight_qcm
st.sidebar.markdown(f"**Poids du texte :** {weight_text:.2f}")

nb_qcm = sum(1 for q in questions if q.get("type") == "qcm")
st.sidebar.markdown(f"**Nombre de questions QCM :** {nb_qcm}")


st.header(" Questionnaire")

qcm_answers = {}
open_texts = []

for q in questions:
    q_id = q.get("id")
    q_text = q.get("text", "")
    q_type = q.get("type", "qcm")

    if q_type == "qcm":
        value = st.slider(
            q_text,
            min_value=1,
            max_value=5,
            value=3,
            key=f"qcm_{q_id}",
        )
        qcm_answers[q_id] = value

    elif q_type == "text":
        st.markdown(f"**{q_text}**")
        txt = st.text_area(
            label="Your answer:",
            key=f"text_{q_id}",
            height=80,
            placeholder="Write your answer here..."
        )
        open_texts.append(txt)

    st.markdown("---")


# Bouton de prédiction

if st.button(" Analyse my MBTI profile"):
    qcm_dict = {}
    id_to_text = {q["id"]: q["text"] for q in questions if q.get("type") == "qcm"}
    for col in qcm_columns:

        matched_id = None
        for qid, txt in id_to_text.items():
            if txt == col:
                matched_id = qid
                break
        if matched_id is not None and matched_id in qcm_answers:
            qcm_dict[col] = int(qcm_answers[matched_id])
        else:
            # Si jamais on ne trouve pas → valeur par défaut neutre
            qcm_dict[col] = 3

    X_qcm_df = pd.DataFrame([qcm_dict], columns=qcm_columns)
    prob_qcm = qcm_model.predict_proba(X_qcm_df)[0]


    full_text = "\n\n".join([t for t in open_texts if t.strip()])
    if full_text.strip():
        emb = text_encoder.encode([full_text])
        prob_text = text_model.predict_proba(emb)[0]
    else:
        prob_text = np.zeros_like(prob_qcm)

    fused_probas = weight_qcm * prob_qcm + weight_text * prob_text
    fused_probas = fused_probas / (fused_probas.sum() + 1e-9)

    idx = int(np.argmax(fused_probas))
    type_final = MBTI_TYPES[idx]
    conf_final = float(fused_probas[idx])

# result
    st.header(" MBTI Result")
    st.subheader(f"Predicted MBTI type: **{type_final}**")
    st.markdown(f"**Global confidence:** {conf_final:.2f}")

    axes = mbti_to_axes_probs(fused_probas, MBTI_TYPES)

    st.subheader(" MBTI axes (relative percentages)")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**E vs I**")
        st.write(f"- E : {axes['E']*100:.1f}%")
        st.write(f"- I : {axes['I']*100:.1f}%")

        st.write(f"**S vs N**")
        st.write(f"- S : {axes['S']*100:.1f}%")
        st.write(f"- N : {axes['N']*100:.1f}%")

    with col2:
        st.write(f"**T vs F**")
        st.write(f"- T : {axes['T']*100:.1f}%")
        st.write(f"- F : {axes['F']*100:.1f}%")

        st.write(f"**J vs P**")
        st.write(f"- J : {axes['J']*100:.1f}%")
        st.write(f"- P : {axes['P']*100:.1f}%")

    st.subheader(" Top 5 most likely types")
    sorted_indices = np.argsort(fused_probas)[::-1]
    for rank in range(5):
        i = sorted_indices[rank]
        st.write(f"{rank+1}. **{MBTI_TYPES[i]}** – {fused_probas[i]:.2f}")

else:
    st.info("Fill in the questionnaire, then click **“Analyse my MBTI profile”** to see your result.")
