import streamlit as st
from src.generate import answer
from src.pipeline import build_retrievers
import base64


with open("data/barrel.svg", "rb") as f:
    svg_data = f.read().decode("utf-8")

b64_svg = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')

st.set_page_config(
    page_title="RAG ЭЛОУ-АВТ", 
    page_icon="", 
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    
    .main-header {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #0093D0;
        text-align: left;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #0093D0;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 0.95rem;
        color: #555555;
        margin-bottom: 2rem;
    }
    
    .answer-box {
        background-color: #fafafa;
        border-left: 5px solid #F29400;
        padding: 1.5rem;
        margin: 1rem 0;
        border-top: 1px solid #eeeeee;
        border-right: 1px solid #eeeeee;
        border-bottom: 1px solid #eeeeee;
    }
    .answer-label {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #0093D0;
        margin-bottom: 0.5rem;
    }
    .answer-text {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 1rem;
        color: #1a1a1a;
        line-height: 1.6;
    }
    
    .source-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .source-text {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 0.85rem;
        color: #333333;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    
    .stButton > button {
        font-family: 'Segoe UI', Roboto, sans-serif;
        background-color: #0093D0 !important; 
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: background-color 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #F29400 !important; /* Чуть темнее при наведении */
        color: #ffffff !important;
        border-color: #F29400 !important;
    }
    
    .stTextInput > div > div > input {
        font-family: 'Segoe UI', Roboto, sans-serif;
        border-radius: 4px;
        border: 1px solid #cccccc;
        padding: 0.7rem;
        font-size: 0.95rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0093D0;
        box-shadow: 0 0 0 2px rgba(0,147,208,0.2);
    }
    
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #eeeeee;
    }

    hr {
        border: none;
        border-top: 1px solid #eeeeee;
        margin: 1.5rem 0;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', Roboto, sans-serif;
        color: #0093D0;
    }
    p, label, div {
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    
    .stSelectbox > div > div {
        border-color: #cccccc;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #0093D0;
        box-shadow: 0 0 0 2px rgba(0,147,208,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f'''
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
    <!-- Указываем размер 36px и object-fit: contain -->
    <img src="data:image/svg+xml;base64,{b64_svg}" style="width: 36px; height: 36px; object-fit: contain; flex-shrink: 0;">
    <div style="font-family: 'Segoe UI', Roboto, sans-serif; font-size: 2rem; font-weight: 600; color: #0093D0; padding: 0; margin: 0; line-height: 1.2;">
        Семантический поиск по регламенту ЭЛОУ-АВТ
    </div>
</div>
<div style="border-bottom: 3px solid #0093D0; margin-top: 5px; margin-bottom: 1rem;"></div>
''', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Система автоматического поиска ответов в технологическом регламенте на основе RAG-технологии</div>', unsafe_allow_html=True)

@st.cache_resource
def load_retrievers():
    return build_retrievers(alpha=0.9, use_rerank=False)

retrievers = load_retrievers()

with st.sidebar:
    st.markdown("### Параметры поиска")
    retriever_name = st.selectbox("Метод поиска", ["hybrid", "bm25", "dense"], index=0)
    llm_name = st.selectbox("Метод генерации", ["extractive", "groq"], index=1)
    
    k = 10
    
    expand = st.checkbox("Раскрывать аббревиатуры", value=True)
    
    st.markdown("---")
    st.markdown("### Примеры запросов")
    examples = [
        "Какое давление в колонне К-1?",
        "Какая температура низа К-2?",
        "Что такое тяжёлый остаток вакуумной перегонки?",
        "Какое октановое число у бензина после риформинга?"
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state['question'] = ex

if 'question' not in st.session_state:
    st.session_state['question'] = ""

question = st.text_input("Введите вопрос по регламенту:", value=st.session_state['question'], placeholder="Например: При какой температуре отбирается керосиновая фракция?")

if question:
    with st.spinner("Поиск ответа..."):
        res = answer(question, retrievers[retriever_name], k=k, llm=llm_name, expand=expand)
    
    st.markdown(f'''
    <div class="answer-box">
        <div class="answer-label">Ответ системы</div>
        <div class="answer-text">{res["answer"]}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("### Источники")
    for s, (c, _) in zip(res["sources"], res["context"]):
        with st.expander(f"[{s['n']}] {s['document']} — {s['section']}"):
            st.markdown(f'<div class="source-text">{c["text"]}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center; color:#888888; font-size:0.75rem; font-family: Segoe UI, Roboto, sans-serif;">Курсовая работа • Семантический поиск по регламенту ЭЛОУ-АВТ • RAG-система</p>', unsafe_allow_html=True)
