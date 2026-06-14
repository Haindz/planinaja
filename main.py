from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Sistem Rekomendasi Pariwisata Tanpa Login")

# Konfigurasi CORS agar bisa diakses oleh website HTML/Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variabel Global
tourism_id = None
cosine_sim = None

# 1. Mount folder 'assets' agar gambar JPG bisa dibaca browser
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# 2. Mount folder utama (titik '.') agar style.css bisa dibaca
app.mount("/static", StaticFiles(directory="."), name="static")

# 3. Setup Jinja2 (Membaca folder utama karena 'templates' dan 'partisi' terpisah)
templates = Jinja2Templates(directory=".")

# 4. Update Path Dataset ke folder 'datasets'
tourism_id = pd.read_csv('datasets/tourism_with_id.csv')
# (Pastikan dataset lain seperti rating dll juga diubah path-nya menjadi 'datasets/...')

print("Memuat dataset dan merakit algoritma... Mohon tunggu.")

# 1. Load Data
t_id = pd.read_csv('datasets/tourism_with_id.csv')
rating = pd.read_csv('datasets/tourism_rating.csv')

# 2. Data Cleaning
t_id = t_id.drop(['Unnamed: 11', 'Unnamed: 12'], axis=1, errors='ignore')
t_id['Time_Minutes'] = t_id['Time_Minutes'].fillna(t_id['Time_Minutes'].median())

# 3. Feature Engineering: Mood
def tentukan_mood(kategori):
    if kategori == 'Bahari': return 'Santai, Alam'
    elif kategori == 'Budaya': return 'Edukasi, Tenang'
    elif kategori == 'Taman Hiburan': return 'Seru, Keluarga'
    elif kategori == 'Cagar Alam': return 'Petualangan, Alam'
    elif kategori == 'Pusat Perbelanjaan': return 'Santai, Belanja'
    elif kategori == 'Tempat Ibadah': return 'Tenang, Religi'
    else: return 'Umum'

t_id['Mood'] = t_id['Category'].apply(tentukan_mood)
t_id['Tags'] = t_id['Category'] + " " + t_id['City'] + " " + t_id['Mood']

# 4. Feature Engineering: Popularity (Weighted Rating)
jumlah_review = rating.groupby('Place_Id').size().reset_index(name='Jumlah_Ulasan')
t_id = pd.merge(t_id, jumlah_review, on='Place_Id', how='left')
t_id['Jumlah_Ulasan'] = t_id['Jumlah_Ulasan'].fillna(0)

v = t_id['Jumlah_Ulasan']
R = t_id['Rating']
C = t_id['Rating'].mean()
m = t_id['Jumlah_Ulasan'].quantile(0.75)
t_id['Skor_Popularitas'] = ((v / (v + m)) * R) + ((m / (v + m)) * C)

# 5. Algoritma Content-Based (TF-IDF)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(t_id['Tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Simpan ke variabel global
tourism_id = t_id
print("Mesin Rekomendasi Siap Digunakan!")

# --- ROUTING HALAMAN WEB ---
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="templates/index.html")

@app.get("/recommender")
def recommender(request: Request):
    return templates.TemplateResponse(request=request, name="templates/recommender.html")

@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="templates/dashboard.html")

# --- ENDPOINT 2: Wisata Populer (Untuk Landing Page) ---
@app.get("/popular")
def get_popular(kota: str = None, top_n: int = 5):
    df_pop = tourism_id.copy()
    if kota:
        df_pop = df_pop[df_pop['City'].str.lower() == kota.lower()]
    
    df_pop = df_pop.sort_values(by='Skor_Popularitas', ascending=False).head(top_n)
    return {"rekomendasi": df_pop[['Place_Name', 'Category', 'City', 'Rating', 'Jumlah_Ulasan']].to_dict(orient="records")}

# --- ENDPOINT 3: Tempat Serupa (Content-Based) ---
@app.get("/similar/{place_name}")
def get_similar(place_name: str, k: int = 5):
    try:
        idx = tourism_id[tourism_id['Place_Name'].str.lower() == place_name.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores[1:k+1]]
        
        res = tourism_id.iloc[top_indices][['Place_Name', 'Category', 'City', 'Rating']]
        return {"target": place_name, "rekomendasi": res.to_dict(orient="records")}
    except IndexError:
        raise HTTPException(status_code=404, detail="Tempat wisata tidak ditemukan")

# --- ENDPOINT 4: Rencana Perjalanan (Itinerary Generator) ---
@app.get("/itinerary")
def generate_itinerary(kota: str, mood: str, budget: int, pax: int, waktu: int):
    df_filtered = tourism_id[tourism_id['City'].str.lower() == kota.lower()].copy()
    
    if mood:
        df_filtered = df_filtered[df_filtered['Mood'].str.contains(mood, case=False, na=False)]
        
    df_filtered['Total_Biaya'] = df_filtered['Price'] * pax
    df_filtered = df_filtered.sort_values(by='Skor_Popularitas', ascending=False)
    
    itinerary = []
    current_time = 0
    current_cost = 0
    
    for index, row in df_filtered.iterrows():
        if (current_time + row['Time_Minutes'] <= waktu) and (current_cost + row['Total_Biaya'] <= budget):
            itinerary.append({
                'Urutan': len(itinerary) + 1,
                'Tempat': row['Place_Name'],
                'Kategori': row['Category'],
                'Durasi_Menit': row['Time_Minutes'],
                'Biaya_Rp': row['Total_Biaya']
            })
            current_time += row['Time_Minutes']
            current_cost += row['Total_Biaya']
            
    if not itinerary:
        raise HTTPException(status_code=404, detail="Maaf, budget atau waktu terlalu sedikit untuk kota/mood ini.")
        
    return {
        'Ringkasan': {
            'Kota_Tujuan': kota,
            'Mood': mood,
            'Sisa_Budget_Rp': budget - current_cost,
            'Sisa_Waktu_Menit': waktu - current_time
        },
        'Rencana_Perjalanan': itinerary
    }

# --- ENDPOINT 5: Analytics & Model Evaluation (Dinamis - REVISI) ---
@app.get("/analytics")
def get_analytics():
    # 1. Statistik Dasar (5 Scorecard)
    total_dest = len(tourism_id)
    avg_rating = round(tourism_id['Rating'].mean(), 2)
    max_price = int(tourism_id['Price'].max())
    avg_time = int(tourism_id['Time_Minutes'].mean())
    max_reviews = int(tourism_id['Jumlah_Ulasan'].max())
    
    # 2. Distribusi Kota & Kategori
    city_counts = tourism_id['City'].value_counts().to_dict()
    cat_counts = tourism_id['Category'].value_counts().to_dict()
    
    # 3. Rata-rata Rating per Kategori
    cat_ratings = tourism_id.groupby('Category')['Rating'].mean().round(2).to_dict()
    
    # 4. Evaluasi Model: Distribusi Cosine Similarity TF-IDF
    sim_values = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(sim_values, bins=bins)
    
    sim_distribution = {
        "Sangat Beda": int(hist[0]),
        "Kurang Mirip": int(hist[1]),
        "Cukup Mirip": int(hist[2]),
        "Mirip": int(hist[3]),
        "Sangat Mirip": int(hist[4])
    }

    # 5. Data untuk Tabel (List Wisata & Harga)
    table_data = tourism_id[['Place_Name', 'City', 'Price', 'Rating']].to_dict(orient='records')

    return {
        "summary": {
            "total": total_dest,
            "avg_rating": avg_rating,
            "max_price": max_price,
            "avg_time": avg_time,
            "max_reviews": max_reviews
        },
        "city_distribution": city_counts,
        "category_distribution": cat_counts,
        "category_ratings": cat_ratings,
        "model_evaluation": sim_distribution,
        "table_data": table_data
    }