# synonym_checker_extended.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
import json
import tempfile
import os
import shutil
import zipfile
import tarfile
from typing import List, Tuple, Dict, Any
from datetime import datetime

from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
import torch

import altair as alt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --------------------
# Утилиты
# --------------------

def preprocess_text(t: Any) -> str:
    if pd.isna(t):
        return ""
    return " ".join(str(t).lower().strip().split())

def file_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def read_uploaded_file_bytes(uploaded) -> Tuple[pd.DataFrame, str]:
    """Прочитать CSV или Excel из streamlit uploader и вернуть DataFrame + md5 хэш содержимого."""
    raw = uploaded.read()
    h = file_md5(raw)
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            raise ValueError("Файл должен быть CSV или Excel. Ошибка: " + str(e))
    return df, h

def parse_topics_field(val) -> List[str]:
    """Преобразует поле topics в список строк."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in [";", "|", ","]:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s] if s else []

def jaccard_tokens(a: str, b: str) -> float:
    sa = set([t for t in a.split() if t])
    sb = set([t for t in b.split() if t])
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0

def style_low_score_rows(df, threshold=0.75):
    def highlight(row):
        score_val = row.get('score', None)
        try:
            cond = pd.notna(score_val) and float(score_val) < threshold
        except Exception:
            cond = False
        return ['background-color: #ffcccc' if cond else '' for _ in row]
    return df.style.apply(highlight, axis=1)

def style_suspicious_and_low(df, sem_thresh: float, lex_thresh: float, low_score_thresh: float):
    def highlight(row):
        out = []
        try:
            score = float(row.get('score', 0))
        except Exception:
            score = 0.0
        try:
            lex = float(row.get('lexical_score', 0))
        except Exception:
            lex = 0.0

        is_low_score = (score < low_score_thresh)
        is_suspicious = (score >= sem_thresh and lex <= lex_thresh)
        for _ in row:
            if is_suspicious:
                out.append('background-color: #fff2b8')
            elif is_low_score:
                out.append('background-color: #ffcccc')
            else:
                out.append('')
        return out
    return df.style.apply(highlight, axis=1)

# --------------------
# Google Drive model downloader with extraction
# --------------------
def _looks_like_model_dir(path: str) -> bool:
    expected = {"config.json", "pytorch_model.bin", "sentence_bert_config.json", "tokenizer_config.json", "vocab.txt", "tokenizer.json", "special_tokens_map.json"}
    try:
        entries = set(os.listdir(path))
    except Exception:
        return False
    if expected & entries:
        return True
    if "modules" in entries:
        return True
    return False

def _find_model_dir_candidate(base_dir: str) -> str:
    if _looks_like_model_dir(base_dir):
        return base_dir
    try:
        children = [p for p in os.listdir(base_dir) if not p.startswith(".")]
    except Exception:
        return base_dir
    if len(children) == 1:
        child_path = os.path.join(base_dir, children[0])
        if os.path.isdir(child_path) and _looks_like_model_dir(child_path):
            return child_path
    for child in children:
        child_path = os.path.join(base_dir, child)
        if os.path.isdir(child_path) and _looks_like_model_dir(child_path):
            return child_path
    return base_dir

def download_file_from_gdrive(file_id: str) -> str:
    import gdown
    tmp_dir = tempfile.gettempdir()
    archive_path = os.path.join(tmp_dir, f"model_gdrive_{file_id}")
    extract_dir = os.path.join(tmp_dir, f"model_gdrive_extracted_{file_id}")

    if not os.path.exists(archive_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, archive_path, quiet=True)

    if os.path.exists(extract_dir) and os.path.isdir(extract_dir):
        return _find_model_dir_candidate(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(extract_dir)
        else:
            try:
                shutil.copy(archive_path, extract_dir)
            except Exception:
                pass
    except Exception:
        return extract_dir

    candidate = _find_model_dir_candidate(extract_dir)
    return candidate

@st.cache_resource(show_spinner=False)
def load_model_from_source(source: str, identifier: str) -> SentenceTransformer:
    if source == "huggingface":
        model_path = identifier
    elif source == "google_drive":
        model_path = download_file_from_gdrive(identifier)
    else:
        raise ValueError("Unknown model source")
    try:
        model = SentenceTransformer(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Не удалось загрузить модель из '{model_path}'. "
            f"Убедитесь, что это корректное имя HF или zip/tar архива модели. Ошибка: {e}"
        )
    return model

def encode_texts_in_batches(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.array([])
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embs)

# --------------------
# Metrics helpers
# --------------------
def compute_classification_metrics(y_true: List[int], y_score: List[float], threshold: float = 0.75):
    # binary prediction using threshold
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

def compute_topic_metrics(df_with_labels: pd.DataFrame, threshold: float):
    """
    df_with_labels must contain columns: 'topics_list' (list), 'label' (0/1), 'score' (float)
    returns dict topic -> metrics
    """
    topic_metrics = {}
    # collect all topics
    all_topics = sorted({t for topics in df_with_labels['topics_list'] for t in topics})
    for t in all_topics:
        subset = df_with_labels[df_with_labels['topics_list'].apply(lambda lst: t in lst)]
        if len(subset) == 0:
            continue
        metrics = compute_classification_metrics(subset['label'].tolist(), subset['score'].tolist(), threshold)
        topic_metrics[t] = {"n": len(subset), **metrics}
    return topic_metrics

# --------------------
# UI init
# --------------------
st.set_page_config(page_title="Synonym Checker — Extended", layout="wide")
st.title("🔎 Synonym Checker — Extended (QA, model DB, editor, training)")

# Sidebar model settings
st.sidebar.header("Model settings")
model_source = st.sidebar.selectbox("Model source", ["huggingface", "google_drive"], index=0)
if model_source == "huggingface":
    model_id = st.sidebar.text_input("HuggingFace model id", value="sentence-transformers/all-MiniLM-L6-v2")
else:
    model_id = st.sidebar.text_input("Google Drive file id (zip/tar with model)", value="1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf")

enable_ab_test = st.sidebar.checkbox("Enable A/B test", value=False)
if enable_ab_test:
    ab_model_source = st.sidebar.selectbox("Model B source", ["huggingface", "google_drive"], index=0, key="ab_src")
    if ab_model_source == "huggingface":
        ab_model_id = st.sidebar.text_input("HF model id (B)", value="sentence-transformers/all-mpnet-base-v2", key="ab_id")
    else:
        ab_model_id = st.sidebar.text_input("GDrive file id (B)", value="", key="ab_id")
else:
    ab_model_id = ""

batch_size = st.sidebar.number_input("Encode batch size", min_value=8, max_value=1024, value=64, step=8)

st.sidebar.markdown("---")
st.sidebar.header("Detector (high sem, low lex)")
enable_detector = st.sidebar.checkbox("Enable detector", value=True)
semantic_threshold = st.sidebar.slider("Semantic threshold (>=)", 0.0, 1.0, 0.80, 0.01)
lexical_threshold = st.sidebar.slider("Lexical threshold (<=)", 0.0, 1.0, 0.30, 0.01)
low_score_threshold = st.sidebar.slider("Low score highlight (<)", 0.0, 1.0, 0.75, 0.01)

# Load model A
try:
    with st.spinner("Loading main model..."):
        model_a = load_model_from_source(model_source, model_id)
    st.sidebar.success("Main model loaded")
except Exception as e:
    st.sidebar.error(f"Can't load main model: {e}")
    st.stop()

model_b = None
if enable_ab_test and ab_model_id.strip():
    try:
        with st.spinner("Loading model B..."):
            model_b = load_model_from_source(ab_model_source, ab_model_id)
        st.sidebar.success("Model B loaded")
    except Exception as e:
        st.sidebar.error(f"Can't load model B: {e}")
        st.stop()

# Session storage for models DB, history and suggestions
if "models_db" not in st.session_state:
    st.session_state["models_db"] = []  # list of dicts {name, source, id, added_at, last_metrics}
if "history" not in st.session_state:
    st.session_state["history"] = []
if "suggestions" not in st.session_state:
    st.session_state["suggestions"] = []

# helpers
def save_model_record(name: str, source: str, identifier: str, metrics: Dict = None):
    rec = {"name": name, "source": source, "id": identifier, "added_at": datetime.utcnow().isoformat(), "metrics": metrics or {}}
    st.session_state["models_db"].append(rec)
    return rec

def add_to_history(r: dict):
    st.session_state["history"].append(r)

def clear_history():
    st.session_state["history"] = []

def add_suggestions(phrases: List[str]):
    s = [p for p in phrases if p and isinstance(p, str)]
    for p in reversed(s):
        if p not in st.session_state["suggestions"]:
            st.session_state["suggestions"].insert(0, p)
    st.session_state["suggestions"] = st.session_state["suggestions"][:500]

# --------------------
# Top: Model DB UI
# --------------------
st.sidebar.header("Models DB")
st.sidebar.write(f"Saved models: {len(st.session_state['models_db'])}")
with st.sidebar.expander("View / save current model"):
    name = st.text_input("Model friendly name", value=f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    if st.button("Save current model to DB"):
        rec = save_model_record(name, model_source, model_id, metrics=None)
        st.sidebar.success(f"Saved model {name}")

if st.sidebar.button("Download models DB JSON"):
    db_json = json.dumps(st.session_state["models_db"], ensure_ascii=False, indent=2)
    st.sidebar.download_button("Download models DB", data=db_json, file_name="models_db.json", mime="application/json")

st.markdown("## Controls")
st.markdown("Use either **File** mode (bulk file with pairs) or **Manual** mode to test and label pairs. You can run evaluations if you have a labeled test set (column `label` with 0/1).")

# Mode selection
mode = st.radio("Mode", ["File (CSV/XLSX)", "Manual"], index=0, horizontal=True)

# --------------------
# Manual mode — single pair + small batch editor
# --------------------
if mode == "Manual":
    st.header("Manual checks")
    # single pair
    t1 = st.text_input("Phrase A", key="man_a")
    t2 = st.text_input("Phrase B", key="man_b")
    if st.button("Check pair (Manual)"):
        if not t1 or not t2:
            st.warning("Enter both phrases")
        else:
            a = preprocess_text(t1); b = preprocess_text(t2)
            add_suggestions([a,b])
            emb_a = encode_texts_in_batches(model_a, [a], batch_size)
            emb_b = encode_texts_in_batches(model_a, [b], batch_size)
            score_a = float(util.cos_sim(emb_a[0], emb_b[0]).item())
            lex = jaccard_tokens(a,b)
            st.metric("Score A", f"{score_a:.4f}")
            st.metric("Jaccard (lexical)", f"{lex:.4f}")
            if enable_detector and score_a >= semantic_threshold and lex <= lexical_threshold:
                st.warning("Suspicious: high semantic, low lexical")
            if model_b:
                emb_ab = encode_texts_in_batches(model_b, [a], batch_size); emb_bb = encode_texts_in_batches(model_b, [b], batch_size)
                score_b = float(util.cos_sim(emb_ab[0], emb_bb[0]).item())
                st.metric("Score B", f"{score_b:.4f}", delta=f"{(score_b-score_a):+.4f}")
            if st.button("Save manual pair to history"):
                rec = {"source":"manual_single", "pair":{"a":a,"b":b}, "score":score_a, "score_b": (score_b if model_b else None), "lexical_score":lex, "timestamp":datetime.utcnow().isoformat(), "model":model_id}
                add_to_history(rec)
                st.success("Saved to history")
    # small inline editor: user can paste CSV and edit
    st.subheader("Quick dataset editor")
    st.markdown("Paste CSV (columns: phrase_1,phrase_2,label optional, topics optional). After editing you can download it.")
    sample_text = st.text_area("Paste CSV here", height=160, key="quick_csv")
    if st.button("Load pasted CSV"):
        if not sample_text.strip():
            st.warning("Nothing pasted")
        else:
            try:
                df_quick = pd.read_csv(io.StringIO(sample_text))
            except Exception:
                try:
                    df_quick = pd.read_excel(io.BytesIO(sample_text.encode()))
                except Exception as e:
                    st.error(f"Can't parse pasted content: {e}")
                    df_quick = None
            if df_quick is not None:
                # show editable table
                editor = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
                if editor:
                    edited = editor(df_quick, num_rows="dynamic")
                else:
                    st.write("Streamlit version does not support interactive editor on this instance. Showing static table.")
                    st.dataframe(df_quick)
                    edited = df_quick
                st.session_state["_edited_quick"] = edited
                st.download_button("Download edited CSV", data=edited.to_csv(index=False).encode('utf-8'), file_name="edited_quick.csv", mime="text/csv")

# --------------------
# File mode — main features: evaluation, topic reports, editor, training
# --------------------
if mode == "File (CSV/XLSX)":
    st.header("File mode: upload labeled or unlabeled data")
    uploaded = st.file_uploader("Upload CSV/XLSX with columns 'phrase_1','phrase_2' (optional: 'label' 0/1 and 'topics')", type=["csv","xlsx","xls"])
    if uploaded is not None:
        try:
            df_raw, file_hash = read_uploaded_file_bytes(uploaded)
        except Exception as e:
            st.error(f"Read error: {e}")
            st.stop()

        # normalize columns (lowercase names)
        cols_map = {c.lower(): c for c in df_raw.columns}
        if "phrase_1" not in cols_map or "phrase_2" not in cols_map:
            st.error("File must contain columns 'phrase_1' and 'phrase_2' (case-insensitive).")
            st.stop()
        phrase1_col = cols_map["phrase_1"]; phrase2_col = cols_map["phrase_2"]
        topics_col = cols_map.get("topics")
        label_col = cols_map.get("label")

        df = df_raw.copy()
        df["phrase_1_proc"] = df[phrase1_col].apply(preprocess_text)
        df["phrase_2_proc"] = df[phrase2_col].apply(preprocess_text)
        if topics_col:
            df["topics_list"] = df[topics_col].apply(parse_topics_field)
        else:
            df["topics_list"] = [[] for _ in range(len(df))]
        if label_col:
            df["label"] = df[label_col].apply(lambda x: int(x) if pd.notna(x) else None)
        else:
            df["label"] = None

        add_suggestions(list(set(df["phrase_1_proc"].tolist() + df["phrase_2_proc"].tolist())))

        # allow editing dataset in-place
        st.subheader("Dataset editor")
        editor = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
        if editor:
            editable_cols = ["phrase_1", "phrase_2"]
            show_df = df[[phrase1_col, phrase2_col] + (["label"] if label_col else []) + ["topics_list"]].copy()
            st.markdown("You can edit phrases/labels/topics below. After editing press 'Apply edits'.")
            edited = editor(show_df, num_rows="dynamic")
            if st.button("Apply edits to working dataframe"):
                # reflect edits back to df
                for i, row in edited.iterrows():
                    df.at[i, phrase1_col] = row.get(phrase1_col, df.at[i, phrase1_col])
                    df.at[i, phrase2_col] = row.get(phrase2_col, df.at[i, phrase2_col])
                    if "label" in row:
                        df.at[i, "label"] = int(row.get("label")) if pd.notna(row.get("label")) else None
                    if "topics_list" in row:
                        # try to parse topics_list from edited cell: if string, keep as parse; if list, preserve
                        val = row.get("topics_list")
                        if isinstance(val, list):
                            df.at[i, "topics_list"] = val
                        else:
                            df.at[i, "topics_list"] = parse_topics_field(val)
                # update processed text after edits
                df["phrase_1_proc"] = df[phrase1_col].apply(preprocess_text)
                df["phrase_2_proc"] = df[phrase2_col].apply(preprocess_text)
                st.success("Edits applied.")
        else:
            st.info("Interactive data editor not available in this Streamlit version; you can still download and edit offline.")
            st.dataframe(df.head(200))

        # Encode unique texts
        texts = list({t for t in pd.concat([df["phrase_1_proc"], df["phrase_2_proc"]]) if pd.notna(t) and t})
        st.info(f"Unique texts to encode: {len(texts)}")
        with st.spinner("Encoding texts with model A..."):
            embs = encode_texts_in_batches(model_a, texts, batch_size)
        text2idx = {t:i for i,t in enumerate(texts)}

        embs_b = None
        if model_b:
            with st.spinner("Encoding texts with model B..."):
                embs_b = encode_texts_in_batches(model_b, texts, batch_size)

        # compute scores per row
        scores = []
        scores_b = []
        lex_scores = []
        for _, r in df.iterrows():
            a = r["phrase_1_proc"]; b = r["phrase_2_proc"]
            if a in text2idx and b in text2idx:
                sa = embs[text2idx[a]]; sb = embs[text2idx[b]]
                sc = float(util.cos_sim(sa, sb).item())
            else:
                sc = np.nan
            scores.append(sc)
            if embs_b is not None:
                sb_a = embs_b[text2idx[a]]; sb_b = embs_b[text2idx[b]]
                scb = float(util.cos_sim(sb_a, sb_b).item())
                scores_b.append(scb)
            lex_scores.append(jaccard_tokens(a,b))
        df["score"] = scores
        if embs_b is not None:
            df["score_b"] = scores_b
        df["lexical_score"] = lex_scores

        # show results table with highlighting
        st.subheader("Results")
        styled = style_suspicious_and_low(df[[phrase1_col, phrase2_col, "score", "lexical_score", "label", "topics_list"]].rename(columns={phrase1_col:"phrase_1", phrase2_col:"phrase_2"}), semantic_threshold, lexical_threshold, low_score_threshold)
        st.dataframe(styled, use_container_width=True)

        # Download full results
        if st.button("Download results CSV"):
            st.download_button("Download results", data=df.to_csv(index=False).encode('utf-8'), file_name="results_full.csv", mime="text/csv")

        # If labeled, provide QA metrics
        if df["label"].notna().any():
            st.subheader("Quality metrics (using column 'label' as ground truth)")
            # drop rows without label
            df_lab = df[df["label"].notna()].copy()
            y_true = df_lab["label"].astype(int).tolist()
            y_score = df_lab["score"].fillna(0).tolist()
            threshold = st.slider("Prediction threshold for semantic score -> positive", 0.0, 1.0, 0.75, 0.01, key="eval_thresh")
            metrics = compute_classification_metrics(y_true, y_score, threshold)
            st.markdown(f"**Overall metrics (threshold={threshold})**")
            st.write(metrics)
            # confusion matrix
            y_pred = [1 if s >= threshold else 0 for s in y_score]
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])
            st.write("Confusion matrix:")
            st.dataframe(cm_df)

            # Topic-level metrics + chart of worst topics
            topic_metrics = compute_topic_metrics(df_lab, threshold)
            if topic_metrics:
                tdf = pd.DataFrame.from_dict(topic_metrics, orient='index').reset_index().rename(columns={'index':'topic'})
                st.markdown("**Per-topic metrics**")
                st.dataframe(tdf.sort_values("f1").head(50))
                # chart top N worst by f1
                chart_df = tdf.copy()
                if not chart_df.empty:
                    chart_df = chart_df.sort_values("f1")
                    c = alt.Chart(chart_df.head(30)).mark_bar().encode(
                        y=alt.Y("topic:N", sort=alt.EncodingSortField(field="f1", order="ascending")),
                        x=alt.X("f1:Q", title="F1 score"),
                        tooltip=["topic","n","accuracy","precision","recall","f1"]
                    )
                    st.altair_chart(c, use_container_width=True)

            # Save evaluation to model DB
            if st.button("Save evaluation to models DB"):
                model_rec = {"model": model_id, "source": model_source, "evaluated_at": datetime.utcnow().isoformat(), "metrics": metrics, "threshold": threshold, "n": len(df_lab)}
                st.session_state["models_db"].append(model_rec)
                st.success("Saved evaluation to models DB")

        else:
            st.info("No labeled rows found (column 'label' missing). Upload dataset with 'label' column for evaluation metrics.")

        # Suspicious subset
        if enable_detector:
            susp = df[(df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)]
            st.subheader("Suspicious pairs (high semantic, low lexical)")
            if susp.empty:
                st.write("No suspicious pairs with current thresholds.")
            else:
                st.write(f"Found {len(susp)} suspicious pairs")
                st.dataframe(susp[[phrase1_col, phrase2_col, "score", "lexical_score", "topics_list"]].rename(columns={phrase1_col:"phrase_1", phrase2_col:"phrase_2"}), use_container_width=True)
                if st.button("Download suspicious CSV"):
                    st.download_button("Download suspicious", data=susp.to_csv(index=False).encode('utf-8'), file_name="suspicious.csv", mime="text/csv")

        # Training UI — allow user to create train set (from suspicious or from labeled positives) and fine-tune
        st.subheader("Fine-tune model (local)")
        st.markdown("You can create a small train dataset and fine-tune the **loaded** model. **Warning**: training may consume CPU/RAM and is slow on free Streamlit. Keep epochs small and batch sizes small.")
        # assemble train candidates
        make_train_from = st.selectbox("Create train data from", ["no", "positive labeled pairs (label==1)", "suspicious pairs (detector)", "select nothing / upload json"], index=0)
        train_df = None
        if make_train_from == "positive labeled pairs (label==1)":
            if 'label' in df.columns and df['label'].notna().any():
                train_df = df[df['label']==1][[phrase1_col, phrase2_col]].dropna().drop_duplicates()
                st.write(f"Using {len(train_df)} positive labeled pairs")
            else:
                st.warning("No labeled positive rows available.")
        elif make_train_from == "suspicious pairs (detector)":
            train_df = df[(df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)][[phrase1_col, phrase2_col]]
            st.write(f"Using {len(train_df)} suspicious pairs")
        elif make_train_from == "select nothing / upload json":
            uploaded_train = st.file_uploader("Or upload train json (list of {texts:[a,b], label:1.0})", type=["json"], key="trainjson")
            if uploaded_train is not None:
                try:
                    j = json.load(uploaded_train)
                    # transform to df
                    rows = []
                    for item in j:
                        texts = item.get("texts") or item.get("pair") or []
                        if isinstance(texts, list) and len(texts)>=2:
                            rows.append({phrase1_col: preprocess_text(texts[0]), phrase2_col: preprocess_text(texts[1])})
                    train_df = pd.DataFrame(rows)
                    st.write(f"Loaded {len(train_df)} train pairs from json")
                except Exception as e:
                    st.error(f"Can't parse train json: {e}")

        if train_df is not None and not train_df.empty:
            st.dataframe(train_df.head(200))
            train_epochs = st.number_input("Train epochs", min_value=1, max_value=10, value=1)
            train_batch = st.number_input("Train batch size", min_value=2, max_value=64, value=8)
            warmup = st.number_input("Warmup steps", min_value=0, max_value=1000, value=10)
            out_dir = st.text_input("Output model folder (will be saved locally to /tmp)", value=f"/tmp/fine_tuned_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            if st.button("Start fine-tuning (local)"):
                # prepare examples
                examples = []
                for _, r in train_df.iterrows():
                    a = preprocess_text(r[phrase1_col]); b = preprocess_text(r[phrase2_col])
                    examples.append(InputExample(texts=[a,b], label=1.0))
                if not examples:
                    st.warning("No train examples")
                else:
                    st.info("Training started — may take time and CPU.")
                    train_dataloader = DataLoader(examples, shuffle=True, batch_size=int(train_batch))
                    train_loss = losses.CosineSimilarityLoss(model_a)
                    try:
                        model_a.fit(train_objectives=[(train_dataloader, train_loss)], epochs=int(train_epochs), warmup_steps=int(warmup))
                        # save
                        try:
                            model_a.save(out_dir)
                            st.success(f"Model fine-tuned and saved to {out_dir}")
                            # archive and provide download
                            zip_name = out_dir.rstrip("/").split("/")[-1] + ".zip"
                            zip_path = os.path.join(tempfile.gettempdir(), zip_name)
                            shutil.make_archive(base_name=zip_path.replace(".zip",""), format="zip", root_dir=out_dir)
                            with open(zip_path, "rb") as f:
                                data = f.read()
                            st.download_button("Download zipped fine-tuned model", data=data, file_name=zip_name, mime="application/zip")
                        except Exception as e:
                            st.error(f"Model saved failed: {e}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        # allow saving dataset edits / annotated dataset
        if st.button("Download processed dataset (with scores & lexical)"):
            st.download_button("Download processed", data=df.to_csv(index=False).encode('utf-8'), file_name="processed_with_scores.csv", mime="text/csv")

# --------------------
# History viewer
# --------------------
st.markdown("---")
st.header("History & Models DB")
st.write("Recent history entries:")
if st.session_state["history"]:
    for i, rec in enumerate(reversed(st.session_state["history"])):
        st.markdown(f"**#{len(st.session_state['history'])-i}**  |  {rec.get('timestamp','-')}  | source: {rec.get('source','-')}")
        st.write(rec)
        st.markdown("---")
else:
    st.write("No history yet.")

st.write("Models DB:")
if st.session_state["models_db"]:
    st.dataframe(pd.DataFrame(st.session_state["models_db"]))
else:
    st.write("No saved models yet.")
