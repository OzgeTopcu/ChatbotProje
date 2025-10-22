# 🎬 CineMate — Akıllı Film Sohbet Asistanı (RAG Destekli)

CineMate, kullanıcıların ruh haline, tarzına veya izleme niyetine göre **doğal dilde film önerileri sunan akıllı bir sohbet asistanıdır.**  
Proje, klasik filtreleme temelli öneri sistemlerinden farklı olarak, **anlam tabanlı arama (semantic search)** ve **RAG (Retrieval-Augmented Generation)** mimarisini bir araya getirir.  
Kullanıcı, “romantik ama biraz komik”, “90’lar bilim kurgu”, ya da “yüksek puanlı dram filmi” gibi ifadelerle konuşarak öneri alabilir.

---

## 🎯 Projenin Amacı

Günümüzde film öneri sistemleri çoğunlukla kullanıcı geçmişine ya da basit etiket eşleşmelerine dayanıyor.  
CineMate’in amacı, bu klasik yöntemi geliştirip, kullanıcıların **doğal konuşma diliyle film arayabilmesini** sağlamaktır.  
Yani kullanıcılar artık “şu tarz bir film istiyorum” diyerek duygularını veya niyetlerini ifade edebilir, sistem ise bu ifadeyi **anlam düzeyinde analiz ederek** doğru filmleri önerebilir.

---

## 🗂️ Veri Seti Hakkında

Projede iki temel veri kaynağı kullanılmıştır:

1. **[MovieLens Dataset (ml-latest-small)](https://grouplens.org/datasets/movielens/)**  
   - İçeriğinde `movies.csv` ve `ratings.csv` dosyaları yer alır.  
   - Filmlerle ilgili `movieId`, `title`, `genres` bilgilerini ve kullanıcı puanlarını içerir.  
   - Yaklaşık 9.000 film kaydı ve 100.000’den fazla rating bulunmaktadır.

2. **[TMDb (The Movie Database) API](https://developer.themoviedb.org/)**  
   - MovieLens verilerinde eksik olan **film özetleri ve poster görselleri** bu API üzerinden alınmıştır.  
   - API, hem Türkçe hem İngilizce dil desteğiyle çalışacak şekilde tasarlanmıştır.  
   - Örnek veri: `title`, `overview`, `poster_path`.

Veri seti, `data/movies.csv` ve `data/ratings.csv` dosyaları üzerinden otomatik olarak yüklenmektedir.  
Ek olarak `.env` dosyasında **TMDb API anahtarı** ve **OpenRouter API anahtarı** tutulur.

---

## 🧠 Kullanılan Yöntemler ve Mimariler

CineMate, **RAG (Retrieval-Augmented Generation)** yapısını temel alan bir hibrit sistemdir.

### 🔹 1. Anlam Tabanlı Arama (Semantic Retrieval)
- Filmlerin başlık ve tür bilgileri, **SentenceTransformer (“all-MiniLM-L6-v2”)** modeli ile vektörleştirilmiştir.
- Kullanıcının yazdığı sorgu (“komik ama romantik”) embed edilip **cosine similarity** ile en benzer film vektörleri bulunur.

### 🔹 2. Bilgi Zenginleştirme (Augmentation)
- En benzer 5 film TMDb API’den özet ve poster bilgisiyle zenginleştirilir.
- Bu bilgiler **modelin bağlamı (context)** olarak hazırlanır.

### 🔹 3. Doğal Yanıt Üretimi (Generation)
- OpenRouter API üzerinden **`openai/gpt-oss-20b:free`** modeli kullanılmıştır.
- Model, “sinema eleştirmeni tarzında” Türkçe bir öneri mesajı üretir.
- Sonuç, samimi bir dille yazılmış, film tavsiyesi formatında kullanıcıya sunulur.

### 🔹 4. Görsel Arayüz
- Arayüz **Streamlit** ile oluşturulmuştur.
- Sohbet geçmişi, film kartları, posterler ve özetler dinamik olarak görüntülenir.
- Her öneri sonrasında CineMate kullanıcıya özel tavsiyeler sunar. 🍿

---

## 🏗️ Genel Mimarisi

```text
Kullanıcı Girdisi (Doğal Dil)
          │
          ▼
  [SentenceTransformer]
     (Embedding Üretimi)
          │
          ▼
   [Cosine Similarity]
     (Benzer Filmleri Bul)
          │
          ▼
  [TMDb API ile Zenginleştir]
     (Özet + Poster Bilgisi)
          │
          ▼
 [OpenAI / RAG Model (GPT)]
     (Türkçe Öneri Üretimi)
          │
          ▼
     🎬 Sonuçların Gösterimi
