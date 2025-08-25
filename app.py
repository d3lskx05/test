# app.py
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import io
import os
import json
import requests
from datetime import datetime

# -----------------------------
# Настройки
# -----------------------------
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800
TOP_K = 3
CONTEXT_LIMIT = 2000  # символов для HF API
DATA_DIR = "data_storage"
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Загрузка embedder
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedder()

# -----------------------------
# Вспомогательные функции
# -----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf_bytes(file_bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, max_chars: int = CHUNK_SIZE):
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= max_chars:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def build_embeddings(chunks):
    return np.array(embedder.encode(chunks, show_progress_bar=False))

def save_embeddings(doc_name, chunks, embeddings):
    path = os.path.join(DATA_DIR, f"{doc_name}_chunks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "embeddings": embeddings.tolist()}, f)

def load_embeddings(doc_name):
    path = os.path.join(DATA_DIR, f"{doc_name}_chunks.json")
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["chunks"], np.array(data["embeddings"])

def retrieve_top_k(question, all_chunks, all_embeddings, top_k=TOP_K):
    if not all_chunks or not all_embeddings:
        return []
    q_emb = embedder.encode([question], show_progress_bar=False)[0]
    best = []
    for chunks, emb in zip(all_chunks, all_embeddings):
        if emb.size == 0:
            continue
        norms = np.linalg.norm(emb, axis=1) * (np.linalg.norm(q_emb) + 1e-9)
        sims = (emb @ q_emb) / norms
        idxs = np.argsort(sims)[::-1][:top_k]
        for i in idxs:
            best.append((sims[i], chunks[i]))
    best_sorted = sorted(best, key=lambda x: x[0], reverse=True)[:top_k]
    return [chunk for (_, chunk) in best_sorted]

def highlight_terms(snippet: str, question: str):
    terms = set([w.lower() for w in re.findall(r"\w{3,}", question)])
    escaped = re.escape
    out = snippet
    for t in terms:
        out = re.sub(r'(?i)(' + escaped(t) + r')', r'<mark>\1</mark>', out)
    return out

def extract_money_percent_dates(text: str):
    items = {}
    perc = re.findall(r'(\d+(?:[.,]\d+)?\s?%|\d+(?:[.,]\d+)?\s?проц)', text, flags=re.IGNORECASE)
    items['percents'] = list(set(perc))
    sums = re.findall(r'(\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d+)?\s?(?:₽|руб|руб\.|RUB)?)', text)
    items['sums'] = list(set(sums))
    dates = re.findall(r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})', text)
    items['dates'] = list(set(dates))
    return items

# -----------------------------
# Backend: HF API генерация и кэширование
# -----------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
CACHE_PATH = os.path.join(DATA_DIR, "hf_cache.json")
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        hf_cache = json.load(f)
else:
    hf_cache = {}

def generate_answer(question: str, context: str):
    key = f"{question}||{context[:CONTEXT_LIMIT]}"
    if key in hf_cache:
        return hf_cache[key]
    prompt = f"""Ты помощник, который отвечает только на основе документа. Отвечай коротко и только фактами.
Вопрос: {question}

Контекст:
{context[:CONTEXT_LIMIT]}

Ответ:"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        output = response.json()
        if isinstance(output, list) and "generated_text" in output[0]:
            answer = output[0]["generated_text"].strip()
        else:
            answer = str(output)
    except Exception as e:
        answer = f"Ошибка API: {e}"
    hf_cache[key] = answer
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(hf_cache, f)
    return answer

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Банковский помощник (MVP)", layout="wide")
st.title("📄 Банковский помощник — анализ договоров (MVP)")

if "docs" not in st.session_state:
    st.session_state.docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Управление")
    uploaded = st.file_uploader("Загрузить PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            raw = f.read()
            txt = clean_text(extract_text_from_pdf_bytes(raw))
            st.session_state.docs.append({"name": f.name, "text": txt})
        st.success(f"{len(uploaded)} файл(ов) загружено.")

    if st.button("Обновить индекс"):
        st.session_state.chunks = []
        st.session_state.embeddings = []
        for doc in st.session_state.docs:
            ch, emb = load_embeddings(doc["name"])
            if ch and emb is not None:
                st.session_state.chunks.append(ch)
                st.session_state.embeddings.append(emb)
            else:
                ch = chunk_text(doc["text"])
                emb = build_embeddings(ch)
                save_embeddings(doc["name"], ch, emb)
                st.session_state.chunks.append(ch)
                st.session_state.embeddings.append(emb)
        st.success("Индекс обновлён ✅")

    if st.button("Очистить все данные"):
        st.session_state.docs = []
        st.session_state.chunks = []
        st.session_state.embeddings = []
        st.session_state.history = []

# Main area
if not st.session_state.docs:
    st.info("Загрузите PDF и обновите индекс.")
    st.stop()

# Показ документов и сравнение
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Загруженные документы")
    for i, doc in enumerate(st.session_state.docs):
        with st.expander(f"{i+1}. {doc['name']}", expanded=False):
            snippet = doc['text'][:1000] + ("..." if len(doc['text'])>1000 else "")
            st.write(snippet)
            items = extract_money_percent_dates(doc['text'])
            st.markdown("**Извлечённые элементы (проценты/суммы/даты):**")
            st.write(items)
with col2:
    st.subheader("Сравнение документов")
    doc_names = [d['name'] for d in st.session_state.docs]
    sel_compare = st.multiselect("Выбрать 2 документа", options=doc_names, max_selections=2)
    if st.button("Сравнить выбранные"):
        if len(sel_compare) != 2:
            st.warning("Выберите ровно 2 документа для сравнения.")
        else:
            idxs = [doc_names.index(n) for n in sel_compare]
            info1 = extract_money_percent_dates(st.session_state.docs[idxs[0]]['text'])
            info2 = extract_money_percent_dates(st.session_state.docs[idxs[1]]['text'])
            st.markdown(f"### {sel_compare[0]} vs {sel_compare[1]}")
            st.write("Проценты:", {sel_compare[0]: info1.get('percents',[]), sel_compare[1]: info2.get('percents',[])})
            st.write("Суммы:", {sel_compare[0]: info1.get('sums',[]), sel_compare[1]: info2.get('sums',[])})
            st.write("Даты:", {sel_compare[0]: info1.get('dates',[]), sel_compare[1]: info2.get('dates',[])})

# Вопрос/ответ
st.subheader("Задайте вопрос по документам")
question = st.text_input("Вопрос:")
if st.button("Отправить"):
    top_chunks = retrieve_top_k(question, st.session_state.chunks, st.session_state.embeddings)
    context = "\n\n".join(top_chunks)[:CONTEXT_LIMIT]
    answer = generate_answer(question, context)
    st.session_state.history.append({"q": question, "a": answer, "ctx": top_chunks})

# История
st.markdown("### 💬 История разговоров")
for item in reversed(st.session_state.history):
    st.markdown(f"**Q:** {item['q']}")
    st.markdown(f"**A:** {item['a']}")
    with st.expander("Показать источники"):
        for ch in item['ctx']:
            st.markdown(highlight_terms(ch, item['q']), unsafe_allow_html=True)

# Резюме документа
st.markdown("---")
st.subheader("📌 Быстрая сводка документа")
doc_select = st.selectbox("Выбрать документ для резюме", options=[d['name'] for d in st.session_state.docs])
if st.button("Сделать краткое резюме"):
    idx = [d['name'] for d in st.session_state.docs].index(doc_select)
    text_for_sum = " ".join(st.session_state.chunks[idx][:6])
    summ = generate_answer("Коротко перескажи основные условия документа (пару предложений).", text_for_sum)
    st.success("Резюме:")
    st.write(summ)
