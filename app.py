# app_faiss_full.py
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import io
import os
import json
import requests
import pickle
from datetime import datetime
import faiss

# -----------------------------
# Настройки
# -----------------------------
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800
TOP_K = 3
CONTEXT_LIMIT = 2000
DATA_DIR = "data_storage"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")
CACHE_PATH = os.path.join(DATA_DIR, "hf_cache.json")
os.makedirs(DATA_DIR, exist_ok=True)

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]

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
    return np.array(embedder.encode(chunks, show_progress_bar=False)).astype("float32")

def save_index(chunks, embeddings):
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None

def retrieve_top_k_faiss(question, index, chunks, top_k=TOP_K):
    if index is None or chunks is None or len(chunks)==0:
        return []
    q_emb = build_embeddings([question])
    scores, idxs = index.search(q_emb, top_k)
    return [chunks[i] for i in idxs[0]]

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
# HF API генерация и кэш
# -----------------------------
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
st.title("📄 Банковский помощник — анализ документов с FAISS")

# Инициализация session_state
if "docs" not in st.session_state:
    st.session_state.docs = []
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Управление")
    uploaded = st.file_uploader("Загрузить PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            raw = f.read()
            txt = clean_text(extract_text_from_pdf_bytes(raw))
            st.session_state.docs.append({"name": f.name, "text": txt})
        # Авто-обновление FAISS
        all_chunks = []
        all_embeddings = []
        for doc in st.session_state.docs:
            ch = chunk_text(doc["text"])
            emb = build_embeddings(ch)
            all_chunks.extend(ch)
            all_embeddings.append(emb)
        embeddings_matrix = np.vstack(all_embeddings)
        save_index(all_chunks, embeddings_matrix)
        st.success("FAISS индекс обновлён автоматически ✅")

    if st.button("Очистить все данные"):
        st.session_state.docs = []
        st.session_state.history = []
        if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
        if os.path.exists(CHUNKS_PATH): os.remove(CHUNKS_PATH)
        st.success("Данные очищены.")

# Main
if not st.session_state.docs:
    st.info("Загрузите PDF.")
    st.stop()

faiss_index, chunks = load_index()

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
            st.warning("Выберите ровно 2 документа.")
        else:
            idxs = [doc_names.index(n) for n in sel_compare]
            info1 = extract_money_percent_dates(st.session_state.docs[idxs[0]]['text'])
            info2 = extract_money_percent_dates(st.session_state.docs[idxs[1]]['text'])
            st.markdown(f"### {sel_compare[0]} vs {sel_compare[1]}")
            st.write("Проценты:", {sel_compare[0]: info1.get('percents',[]), sel_compare[1]: info2.get('percents',[])})
            st.write("Суммы:", {sel_compare[0]: info1.get('sums',[]), sel_compare[1]: info2.get('sums',[])})
            st.write("Даты:", {sel_compare[0]: info1.get('dates',[]), sel_compare[1]: info2.get('dates',[])})

# Вопросы
st.subheader("Задайте вопрос по документам")
question = st.text_input("Вопрос:")
if st.button("Отправить"):
    top_chunks = retrieve_top_k_faiss(question, faiss_index, chunks)
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
    text_for_sum = " ".join(chunks[idx*TOP_K: idx*TOP_K + 6])
    summ = generate_answer("Коротко перескажи основные условия документа (пару предложений).", text_for_sum)
    st.success("Резюме:")
    st.write(summ)
