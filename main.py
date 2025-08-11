from fastapi import FastAPI, Query, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
import httpx
from typing import List, Optional, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veloxg")

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
YOUTUBE_API_KEYS = os.getenv("YOUTUBE_API_KEYS", "").split(",")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing required environment variables: SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="VeloxG API",
    description="FastAPI backend API for VeloxG search engine using Supabase + YouTube + Dictionary",
    version="1.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Cache and Rate Limiting ------------
cached_records: List[Dict[str, Any]] = []
cached_texts: List[str] = []
cached_vectorizer: Optional[TfidfVectorizer] = None
cached_matrix = None  # Precomputed TF-IDF matrix

search_cache: Dict[str, Dict[str, Any]] = {}
rate_limit_data: Dict[str, List[float]] = {}
RATE_LIMIT = 10
RATE_LIMIT_PERIOD = 60  # seconds


def refresh_cache():
    global cached_records, cached_texts, cached_vectorizer, cached_matrix
    try:
        result = supabase.from_("veloxg").select("*").execute()
        cached_records = result.data or []
        cached_texts = [
            f"{(r.get('title') or '').strip()} {(r.get('meta_description') or '').strip()}"
            for r in cached_records
        ]
        # Fallback kama haina data
        if not cached_texts or all(not t.strip() for t in cached_texts):
            cached_texts = ["VeloxG Search Engine Default Entry"]
            cached_records = [{"title": "VeloxG Default", "meta_description": "Default description"}]

        cached_vectorizer = TfidfVectorizer().fit(cached_texts)
        cached_matrix = cached_vectorizer.transform(cached_texts)  # Precompute for speed
        logger.info(f"Cache refreshed with {len(cached_records)} records. Matrix shape: {cached_matrix.shape}")
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        cached_records, cached_texts, cached_vectorizer, cached_matrix = [], [], None, None


refresh_cache()


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
                        pos = meaning.get("partOfSpeech", "")
                        defs = [d.get("definition", "") for d in meaning.get("definitions", [])]
                        for d in defs:
                            meanings.append(f"{pos}: {d}")
                    return meanings[:3]
            return None
    except Exception as e:
        logger.error(f"Dictionary API error: {e}")
        return None


def rate_limiter(ip: str) -> bool:
    import time
    now = time.time()
    timestamps = rate_limit_data.get(ip, [])
    timestamps = [ts for ts in timestamps if now - ts < RATE_LIMIT_PERIOD]
    if len(timestamps) >= RATE_LIMIT:
        return False
    timestamps.append(now)
    rate_limit_data[ip] = timestamps
    return True


def batch_search_scores(query: str):
    """Compute TF-IDF + fuzzy match scores for all records at once."""
    if not cached_vectorizer or cached_matrix is None:
        return []

    # TF-IDF scores (dot product for all docs at once)
    query_vec = cached_vectorizer.transform([query])
    tfidf_scores = np.array(cached_matrix.dot(query_vec.T).toarray()).flatten()

    # Fuzzy scores (faster with process.extract)
    fuzzy_scores = np.array([
        fuzz.token_set_ratio(query, text) / 100
        for text in cached_texts
    ])

    combined_scores = tfidf_scores + fuzzy_scores
    return combined_scores


@app.middleware("http")
async def check_rate_limit(request: Request, call_next):
    client_ip = request.client.host
    if not rate_limiter(client_ip):
        return JSONResponse(status_code=429, content={"detail": "Too many requests. Please slow down."})
    response = await call_next(request)
    return response


@app.get("/")
def home():
    return {"message": "Welcome to VeloxG Search API"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "1.5.0",
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


@app.get("/refresh_cache")
def manual_refresh_cache():
    refresh_cache()
    return {"message": "Cache refreshed successfully", "total_records": len(cached_records)}


@app.get("/search")
async def search(
    q: Optional[str] = Query(None, min_length=1, max_length=100),
    queries: Optional[str] = Query(None, min_length=1, max_length=100),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50)
):
    search_query = q or queries
    if not search_query:
        raise HTTPException(status_code=400, detail="Search query is required")

    cache_key = f"{search_query.lower()}_{page}_{per_page}"
    cache_entry = search_cache.get(cache_key)
    if cache_entry and (datetime.utcnow() - cache_entry["timestamp"]) < timedelta(minutes=5):
        return cache_entry["results"]

    dictionary_results = None
    if len(search_query.split()) == 1 and len(search_query) <= 20:
        dictionary_results = await get_dictionary_meaning(search_query)

    combined_scores = batch_search_scores(search_query)
    if combined_scores == []:
        return {"results": [], "dictionary": dictionary_results}

    results_with_scores = [
        (rec, combined_scores[i])
        for i, rec in enumerate(cached_records)
        if combined_scores[i] > 0.05  # small threshold for match
    ]

    results_with_scores.sort(key=lambda x: x[1], reverse=True)
    start = (page - 1) * per_page
    end = start + per_page
    page_results = [
        {**rec, "image_url": rec.get("image_url", None)}
        for rec, _ in results_with_scores[start:end]
    ]

    response = {
        "query": search_query,
        "page": page,
        "per_page": per_page,
        "total_results": len(results_with_scores),
        "results": page_results,
        "dictionary": dictionary_results,
    }

    search_cache[cache_key] = {"timestamp": datetime.utcnow(), "results": response}
    return response


@app.get("/youtube_search")
async def youtube_search(query: str = Query(..., min_length=1, max_length=100)):
    if not YOUTUBE_API_KEYS or not YOUTUBE_API_KEYS[0]:
        raise HTTPException(status_code=500, detail="No YouTube API keys configured")

    async with httpx.AsyncClient() as client:
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
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    return {"results": response.json().get("items", [])}
                elif response.status_code == 403:
                    continue
            except Exception as e:
                logger.error(f"YouTube API error: {e}")
                continue

    return {"message": "All YouTube API keys failed or quota exceeded"}


@app.get("/search_all")
async def search_all(
    query: str = Query(..., min_length=1, max_length=100),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50)
):
    dictionary_results = None
    if len(query.split()) == 1 and len(query) <= 20:
        dictionary_results = await get_dictionary_meaning(query)

    combined_scores = batch_search_scores(query)
    if combined_scores == []:
        return {"dictionary": dictionary_results, "supabase_results": [], "youtube_results": []}

    results_with_scores = [
        (rec, combined_scores[i])
        for i, rec in enumerate(cached_records)
        if combined_scores[i] > 0.05
    ]
    results_with_scores.sort(key=lambda x: x[1], reverse=True)
    start = (page - 1) * per_page
    end = start + per_page
    supabase_results = [
        {**rec, "image_url": rec.get("image_url", None)}
        for rec, _ in results_with_scores[start:end]
    ]

    youtube_results = []
    async with httpx.AsyncClient() as client:
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
                response = await client.get(url, params=params)
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
        "page": page,
        "per_page": per_page,
        "dictionary": dictionary_results,
        "supabase_results": supabase_results,
        "youtube_results": youtube_results
    }
