from fastapi import FastAPI, Query, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import httpx
import asyncio
from typing import List, Optional, Dict, Any

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
    version="1.4.0"
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

# Cache search results: query -> (timestamp, results)
search_cache: Dict[str, Dict[str, Any]] = {}

# Simple in-memory rate limiting: ip -> [timestamps of requests]
rate_limit_data: Dict[str, List[float]] = {}
RATE_LIMIT = 10  # max 10 requests
RATE_LIMIT_PERIOD = 60  # per 60 seconds

def refresh_cache():
    global cached_records, cached_texts, cached_vectorizer
    try:
        result = supabase.from_("veloxg").select("*").execute()
        cached_records = result.data or []
        cached_texts = [
            f"{r.get('title', '')} {r.get('meta_description', '')}"
            for r in cached_records
        ]
        cached_vectorizer = TfidfVectorizer().fit(cached_texts) if cached_texts else None
        logger.info(f"Cache refreshed with {len(cached_records)} records.")
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        cached_records, cached_texts, cached_vectorizer = [], [], None

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
    # Remove timestamps older than RATE_LIMIT_PERIOD
    timestamps = [ts for ts in timestamps if now - ts < RATE_LIMIT_PERIOD]
    if len(timestamps) >= RATE_LIMIT:
        return False
    timestamps.append(now)
    rate_limit_data[ip] = timestamps
    return True

def weighted_score(title: str, description: str, query: str, vectorizer: TfidfVectorizer):
    # Weight title higher than description
    texts = [title, description]
    weights = [0.7, 0.3]  # example weights

    tfidf = vectorizer.transform(texts)
    query_vec = vectorizer.transform([query])

    scores = [(tfidf[i].dot(query_vec.T).toarray()[0][0]) * weights[i] for i in range(len(texts))]
    return sum(scores)

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
        "version": "1.4.0",
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

    # Check cache for results
    cache_key = f"{search_query.lower()}_{page}_{per_page}"
    cache_entry = search_cache.get(cache_key)
    if cache_entry and (datetime.utcnow() - cache_entry["timestamp"]) < timedelta(minutes=5):
        logger.info(f"Returning cached results for query: {search_query}")
        return cache_entry["results"]

    dictionary_results = None
    if len(search_query.split()) == 1 and len(search_query) <= 20:
        dictionary_results = await get_dictionary_meaning(search_query)

    if not cached_vectorizer or not cached_texts:
        return {"results": [], "dictionary": dictionary_results}

    # Vectorize query
    query_vec = cached_vectorizer.transform([search_query])
    results_with_scores = []
    for rec, text in zip(cached_records, cached_texts):
        title = rec.get("title", "")
        desc = rec.get("meta_description", "")
        # Weighted TF-IDF score for title and description
        score = weighted_score(title, desc, search_query, cached_vectorizer)
        # Fuzzy matching score
        fuzzy = fuzz.token_set_ratio(search_query, text)
        combined_score = score + (fuzzy / 100)  # combine normalized fuzzy with tf-idf
        if combined_score > 0.1:
            results_with_scores.append((rec, combined_score))

    # Sort by combined score descending
    results_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    page_results = [
        {**rec, "image_url": rec.get("image_url", None)} for rec, _ in results_with_scores[start:end]
    ]

    response = {
        "query": search_query,
        "page": page,
        "per_page": per_page,
        "total_results": len(results_with_scores),
        "results": page_results,
        "dictionary": dictionary_results,
    }

    # Cache results
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
                    logger.warning(f"Quota exceeded or forbidden for key: {key}")
                    continue
            except Exception as e:
                logger.error(f"YouTube API error with key {key}: {e}")
                continue

    return {"message": "All YouTube API keys failed or quota exceeded"}

@app.get("/search_all")
async def search_all(
    query: str = Query(..., min_length=1, max_length=100),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50)
):
    dictionary_results = None
    supabase_results = []
    youtube_results = []

    if len(query.split()) == 1 and len(query) <= 20:
        dictionary_results = await get_dictionary_meaning(query)

    if cached_vectorizer and cached_texts:
        query_vec = cached_vectorizer.transform([query])
        results_with_scores = []
        for rec, text in zip(cached_records, cached_texts):
            title = rec.get("title", "")
            desc = rec.get("meta_description", "")
            score = weighted_score(title, desc, query, cached_vectorizer)
            fuzzy = fuzz.token_set_ratio(query, text)
            combined_score = score + (fuzzy / 100)
            if combined_score > 0.1:
                results_with_scores.append((rec, combined_score))

        results_with_scores.sort(key=lambda x: x[1], reverse=True)

        start = (page - 1) * per_page
        end = start + per_page
        supabase_results = [
            {**rec, "image_url": rec.get("image_url", None)} for rec, _ in results_with_scores[start:end]
        ]

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
