# ===================================================
# 🎬 CineMate — Akıllı Film Sohbet Asistanı (Geliştirilmiş Arayüz)
# ===================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------
# 🌟 Streamlit Sayfa Ayarları
# ---------------------------------------------------
st.set_page_config(page_title="CineMate 🎬", page_icon="🎥", layout="wide")

st.markdown("""
<style>
    /* CineMate öneriyor kutusu */
    .cinemate-box {
        background: linear-gradient(145deg, #CDA4DE, #B47FD9); /* yumuşak mor degrade */
        border: none;
        border-radius: 16px;
        padding: 18px;
        color: white; /* yazılar beyaz */
        box-shadow: 0 3px 10px rgba(90, 40, 140, 0.2);
    }

    .cinemate-box h4 {
        color: white;
        font-weight: 700;
        font-size: 1.1em;
        margin-bottom: 0.5em;
    }

    .cinemate-box p {
        color: white;
        font-size: 1.05em;
        line-height: 1.6;
        margin-bottom: 0.3em;
    }
</style>
""", unsafe_allow_html=True)


st.title("🎬 CineMate — Akıllı Film Arkadaşın")
st.markdown("<p class='subtitle'>✨ Ruh haline göre film bul, sohbet et, keşfet!</p>", unsafe_allow_html=True)

# ---------------------------------------------------
# 🔑 Ortam Değişkenleri
# ---------------------------------------------------
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = os.getenv("MODEL_ID", "openai/gpt-oss-20b:free")
API_KEY_TMDB = os.getenv("API_KEY_TMDB")

# ---------------------------------------------------
# 🗃️ Veri Yükleme
# ---------------------------------------------------
@st.cache_data(show_spinner=False)
def load_movielens():
    movies = pd.read_csv("data/movies.csv")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)", expand=False)
    movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)
    movies["genres"] = movies["genres"].str.replace("|", ", ", regex=False)
    movies["overview"] = "Özet bilgisi bulunamadı (MovieLens)."

    ratings = pd.read_csv("data/ratings.csv", usecols=["movieId", "rating"])
    avg = ratings.groupby("movieId", as_index=False)["rating"].mean().rename(columns={"rating": "avg_rating"})
    df = movies.merge(avg, on="movieId", how="left")
    df["avg_rating"] = df["avg_rating"].fillna(0).round(2)
    return df

df = load_movielens()

# ---------------------------------------------------
# 🎥 TMDb Bilgisi
# ---------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_info(title: str) -> Tuple[str, str, Optional[str]]:
    try:
        for lang in ("tr-TR", "en-US"):
            url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY_TMDB}&query={title}&include_adult=false&language={lang}"
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            data = r.json()
            results = data.get("results", [])
            if not results:
                continue
            m = max(results, key=lambda x: x.get("popularity", 0))
            tloc = m.get("title") or m.get("original_title") or title
            ov = (m.get("overview") or "").strip() or "Özet bulunamadı."
            pth = m.get("poster_path")
            poster = f"https://image.tmdb.org/t/p/w500{pth}" if pth else None
            return tloc, ov, poster
        return title, "Özet bulunamadı.", None
    except Exception:
        return title, "Veri alınamadı.", None

# ---------------------------------------------------
# 🧠 Embedding Modeli
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_embeddings(data: pd.DataFrame):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text_corpus = (data["title"] + " " + data["genres"]).tolist()
    embeddings = model.encode(text_corpus, show_progress_bar=True, batch_size=64)
    return model, np.array(embeddings)

embed_model, EMBEDDINGS = build_embeddings(df)

# ---------------------------------------------------
# 🔍 RAG Retrieval
# ---------------------------------------------------
def retrieve_movies(query: str, top_k: int = 5):
    q_emb = embed_model.encode([query])
    sims = cosine_similarity(q_emb, EMBEDDINGS).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    return df.iloc[top_idx], sims[top_idx]

# ---------------------------------------------------
# 💬 Yanıt Oluşturma
# ---------------------------------------------------
def generate_response(user_query: str, context_df: pd.DataFrame):
    context = "\n".join(
        [f"- {r['title']} ({r['year']}) | Tür: {r['genres']} | Puan: {r['avg_rating']} | "
         f"Özet: {r['overview'][:250]}..." for _, r in context_df.iterrows()]
    )

    prompt = f"""
Kullanıcı şu türde bir film arıyor: "{user_query}"

Aşağıda önerilen bazı filmler var:
{context}

Kullanıcıya doğal, Türkçe ve samimi bir öneri yaz.
Filmleri *italik* veya **kalın** biçimde vurgula.
Cümleni izleme tavsiyesiyle bitir. 🎬
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": """
Sen CineMate adında bir film öneri ve yorum asistanısın.
Kullanıcıya dostça, eğlenceli ve bilgi dolu öneriler sunarsın.
Cevaplarını Türkçe, sıcak bir tonla ver.
""" },
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Hata: {e}")
        return "Üzgünüm, şu anda yanıt oluşturamıyorum. Lütfen biraz sonra tekrar dene. 🎥"

# ---------------------------------------------------
# 💬 Chat Arayüzü
# ---------------------------------------------------
if "chat" not in st.session_state:
    st.session_state["chat"] = [
        {"role": "assistant", "content": "Selam 🎬 Nasıl bir film arıyorsun? (örn: 'romantik komedi', '90’lar aksiyon', 'yüksek puanlı bilim kurgu')"}
    ]

# Önceki mesajları göster
for msg in st.session_state["chat"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcı girdisi
st.markdown("<p style='color:#bbb;'>🎭 Tür veya ruh halini yaz: (örnek: 'korku', 'romantik', 'macera')</p>", unsafe_allow_html=True)
user_input = st.chat_input("Film tarzını veya modunu yaz...")

if user_input:
    st.session_state["chat"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    retrieved, sims = retrieve_movies(user_input)

    enriched = []
    for _, row in retrieved.iterrows():
        tloc, ov, poster = tmdb_info(row["title"])
        enriched.append({
            "title": tloc,
            "year": row["year"],
            "genres": row["genres"],
            "avg_rating": row["avg_rating"],
            "overview": ov,
            "poster": poster
        })
    enriched_df = pd.DataFrame(enriched)

    # Model yanıtı
    answer = generate_response(user_input, enriched_df)

    # Sohbet çıktısı
    with st.chat_message("assistant"):
        st.markdown(f"""
        <div class='cinemate-box'>
            <h4>🍿 CineMate Öneriyor:</h4>
            <p>{answer}</p>
        </div>
        """, unsafe_allow_html=True)

        # 🎞️ Film Kartları
        cols = st.columns(3)
        for i, (_, r) in enumerate(enriched_df.iterrows()):
            with cols[i % 3]:
                st.markdown(f"**🎞️ {r['title']} ({r['year']})**")
                if r["poster"]:
                    st.image(r["poster"], use_container_width=True)
                st.caption(f"⭐ {r['avg_rating']} | {r['genres']}")
                with st.expander("🎬 Özet"):
                    st.write(r["overview"])

    st.session_state["chat"].append({"role": "assistant", "content": answer})
    st.toast("🎬 CineMate senin için en iyi filmleri buldu!")
    st.balloons()
