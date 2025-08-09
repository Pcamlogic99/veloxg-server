from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import httpx

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veloxg")

# ---------------- Load ENV ----------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Read multiple keys separated by comma
YOUTUBE_API_KEYS = os.getenv("YOUTUBE_API_KEYS", "").split(",")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing required environment variables: SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- App Config ----------------
app = FastAPI(
    title="VeloxG API",
    description="FastAPI backend API for VeloxG search engine using Supabase + YouTube + Dictionary",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Cache ----------------
cached_records = []
cached_texts = []
cached_vectorizer = None

def refresh_cache():
    global cached_records, cached_texts, cached_vectorizer
    try:
        result = supabase.from_("veloxg").select("*").execute()
        cached_records = result.data or []
        cached_texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in cached_records]
        cached_vectorizer = TfidfVectorizer().fit(cached_texts) if cached_texts else None
        logger.info(f"Cached {len(cached_records)} records")
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        cached_records, cached_texts, cached_vectorizer = [], [], None

refresh_cache()

# ---------------- Dictionary Lookup ----------------
async def get_dictionary_meaning(word: str):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    meanings = []
                    for meaning in data[0].get("meanings", []):
                        part_of_speech = meaning.get("partOfSpeech", "")
                        defs = [d.get("definition", "") for d in meaning.get("definitions", [])]
                        for d in defs:
                            meanings.append(f"{part_of_speech}: {d}")
                    return meanings[:3]  # first 3 short meanings
            return None
    except Exception as e:
        logger.error(f"Dictionary API error: {e}")
        return None

# ---------------- Routes ----------------
@app.get("/")
def home():
    return {"message": "Welcome to VeloxG Search API"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "1.2.0",
        "youtube_keys_loaded": bool(YOUTUBE_API_KEYS and YOUTUBE_API_KEYS[0]),
        "supabase_connected": bool(SUPABASE_URL and SUPABASE_KEY)
    }

@app.post("/add")
def add_link(data: dict = Body(...)):
    if not data.get("title") or not data.get("url"):
        raise HTTPException(status_code=400, detail="Title and URL are required")
    data["timestamp"] = datetime.utcnow().isoformat()
    try:
        response = supabase.from_("veloxg").insert(data).execute()
        refresh_cache()
        return {"message": "Link added successfully", "data": response.data}
    except Exception as e:
        logger.error(f"Error adding link: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(q: str = Query(None), queries: str = Query(None)):
    search_query = q or queries
    if not search_query:
        raise HTTPException(status_code=400, detail="Search query is required")

    # -------- Dictionary Lookup for short single words --------
    dictionary_results = None
    if len(search_query.split()) == 1 and len(search_query) <= 20:
        dictionary_results = await get_dictionary_meaning(search_query)

    try:
        result = supabase.from_("veloxg").select("*").execute()
        records = result.data or []
        texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in records]
        if not texts:
            return {"results": [], "dictionary": dictionary_results}

        vectorizer = TfidfVectorizer().fit(texts)
        query_vec = vectorizer.transform([search_query])
        doc_vecs = vectorizer.transform(texts)
        similarities = (doc_vecs * query_vec.T).toarray().flatten()
        fuzzy_scores = [fuzz.token_set_ratio(search_query, text) for text in texts]
        combined = [(rec, sim, fuzzy) for rec, sim, fuzzy in zip(records, similarities, fuzzy_scores)]
        combined.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top_results = [
            {**rec, "image_url": rec.get("image_url", None)}
            for rec, sim, fuzzy in combined if sim > 0.1 or fuzzy > 60
        ][:10]

        return {"results": top_results, "dictionary": dictionary_results}

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/youtube_search")
def youtube_search(query: str = Query(...)):
    if not YOUTUBE_API_KEYS or not YOUTUBE_API_KEYS[0]:
        raise HTTPException(status_code=500, detail="No YouTube API keys configured")

    for key in YOUTUBE_API_KEYS:
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": 5,
                "key": key.strip()
            }
            response = httpx.get(url, params=params)
            if response.status_code == 200:
                return {"results": response.json().get("items", [])}
            elif response.status_code == 403:
                logger.warning(f"Quota exceeded or forbidden for key: {key}")
                continue
        except Exception as e:
            logger.error(f"YouTube API error with key {key}: {e}")
            continue

    return {"message": "All YouTube API keys failed or quota exceeded"}

@app.get("/search_all")
async def search_all(query: str = Query(...)):
    supabase_results, youtube_results, dictionary_results = [], [], None

    # ---- Dictionary Search (if single short word) ----
    if len(query.split()) == 1 and len(query) <= 20:
        dictionary_results = await get_dictionary_meaning(query)

    # ---- Supabase Search ----
    try:
        result = supabase.from_("veloxg").select("*").execute()
        records = result.data or []
        texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in records]
        if texts:
            vectorizer = TfidfVectorizer().fit(texts)
            query_vec = vectorizer.transform([query])
            doc_vecs = vectorizer.transform(texts)
            similarities = (doc_vecs * query_vec.T).toarray().flatten()
            fuzzy_scores = [fuzz.token_set_ratio(query, text) for text in texts]
            combined = [(rec, sim, fuzzy) for rec, sim, fuzzy in zip(records, similarities, fuzzy_scores)]
            combined.sort(key=lambda x: (x[1], x[2]), reverse=True)
            supabase_results = [
                {**rec, "image_url": rec.get("image_url", None)}
                for rec, sim, fuzzy in combined if sim > 0.1 or fuzzy > 60
            ][:10]
    except Exception as e:
        logger.error(f"Supabase search error: {e}")

    # ---- YouTube Search ----
    for key in YOUTUBE_API_KEYS:
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": 5,
                "key": key.strip()
            }
            response = httpx.get(url, params=params)
            if response.status_code == 200:
                youtube_results = response.json().get("items", [])
                break
            elif response.status_code == 403:
                continue
        except Exception as e:
            logger.error(f"YouTube API error: {e}")
            continue

    return {
        "query": query,
        "dictionary": dictionary_results,
        "supabase_results": supabase_results,
        "youtube_results": youtube_results
    }
