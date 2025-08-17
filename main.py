from fastapi import FastAPI, Query, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import httpx
import re
from typing import Optional
from collections import defaultdict
import time

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veloxg")

# ---------------- Load ENV ----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing required environment variables: SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Load YouTube Keys ----------------
def load_youtube_keys():
    keys = []
    main_key = os.getenv("YOUTUBE_API_KEYS")
    if main_key:
        keys.extend([k.strip() for k in main_key.split(",") if k.strip()])
    for i in range(1, 10):
        extra_key = os.getenv(f"YOUTUBE_API_KEYS{i}")
        if extra_key:
            keys.append(extra_key.strip())
    return keys

YOUTUBE_API_KEYS = load_youtube_keys()

# ---------------- App Config ----------------
app = FastAPI(title="VeloxG API", version="3.2.0")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Security Headers ----------------
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# ---------------- Custom Rate Limiting ----------------
RATE_LIMITS = {
    "/search": (20, 60),
    "/add": (5, 60)
}
rate_limit_store = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path in RATE_LIMITS:
        limit, period = RATE_LIMITS[path]
        ip = request.client.host
        now = time.time()
        request_times = [t for t in rate_limit_store[ip] if now - t < period]
        if len(request_times) >= limit:
            return HTTPException(status_code=429, detail="Rate limit exceeded")
        request_times.append(now)
        rate_limit_store[ip] = request_times
    return await call_next(request)

# ---------------- Cache ----------------
cached_records = []
cached_texts = []
cached_vectorizer = None
last_cache_time = datetime.min
CACHE_TTL = timedelta(minutes=5)

def refresh_cache(force=False):
    global cached_records, cached_texts, cached_vectorizer, last_cache_time
    if not force and datetime.utcnow() - last_cache_time < CACHE_TTL:
        return
    try:
        logger.info("Refreshing cache from Supabase...")
        result = supabase.from_("veloxg").select("*").execute()
        cached_records = result.data or []
        cached_texts = [
            f"{r.get('title', '')} {r.get('meta_description', '')}" 
            for r in cached_records
        ]
        cached_vectorizer = TfidfVectorizer().fit(cached_texts) if cached_texts else None
        last_cache_time = datetime.utcnow()
        logger.info(f"Cached {len(cached_records)} records")
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        cached_records, cached_texts, cached_vectorizer = [], [], None

refresh_cache(force=True)

# ---------------- YouTube Cache ----------------
youtube_cache = {}
YOUTUBE_CACHE_TTL = timedelta(minutes=10)

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
                    return meanings[:3]
            return None
    except Exception as e:
        logger.error(f"Dictionary API error: {e}")
        return None

# ---------------- Input Validation ----------------
def sanitize_query(q: str) -> str:
    q = q.strip()
    if len(q) > 100:
        raise HTTPException(status_code=400, detail="Query too long")
    if not re.match(r"^[\w\s\-\.,!?]+$", q):
        raise HTTPException(status_code=400, detail="Invalid characters in query")
    return q

# ---------------- Routes ----------------
@app.get("/")
def home():
    return {"message": "Welcome to VeloxG Search API"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "3.2.0",
        "youtube_keys_loaded": bool(YOUTUBE_API_KEYS),
        "supabase_connected": bool(SUPABASE_URL and SUPABASE_KEY),
        "cached_records": len(cached_records),
        "youtube_keys_count": len(YOUTUBE_API_KEYS)
    }

# ---------------- Main Search Endpoint ----------------
@app.get("/search")
async def search(
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(15, ge=1, le=50),
):
    if not q:
        raise HTTPException(status_code=400, detail="Search query required")
    q = sanitize_query(q)
    refresh_cache()

    main_results = []

    # Dictionary lookup
    dictionary_results = None
    if len(q.split()) == 1 and len(q) <= 20:
        dictionary_results = await get_dictionary_meaning(q)

    # ---------------- YouTube Videos ----------------
    if YOUTUBE_API_KEYS:
        now = datetime.utcnow()
        cached_videos = youtube_cache.get(q)
        if cached_videos and now - cached_videos["timestamp"] < YOUTUBE_CACHE_TTL:
            videos_data = cached_videos["data"]
        else:
            videos_data = []
            for key in YOUTUBE_API_KEYS:
                try:
                    url = "https://www.googleapis.com/youtube/v3/search"
                    params = {
                        "part": "snippet",
                        "q": q,
                        "type": "video",
                        "maxResults": 6,
                        "key": key
                    }
                    response = httpx.get(url, params=params, timeout=5)
                    if response.status_code == 200:
                        items = response.json().get("items", [])
                        for video in items:
                            videos_data.append({
                                "video_id": video["id"]["videoId"],
                                "title": video["snippet"]["title"],
                                "description": video["snippet"]["description"],
                                "channel": video["snippet"]["channelTitle"],
                                "thumbnail": video["snippet"]["thumbnails"]["high"]["url"],
                                "type": "youtube",
                                "used_key": key
                            })
                        youtube_cache[q] = {"data": videos_data, "timestamp": now}
                        break
                except Exception as e:
                    logger.error(f"YouTube API error with key {key}: {e}")
                    continue
        main_results.extend(videos_data)

    # ---------------- Supabase results ----------------
    if cached_texts and cached_vectorizer:
        query_vec = cached_vectorizer.transform([q])
        doc_vecs = cached_vectorizer.transform(cached_texts)
        similarities = (doc_vecs * query_vec.T).toarray().flatten()
        fuzzy_scores = [fuzz.token_set_ratio(q, text) for text in cached_texts]
        combined = [(rec, sim, fuzzy) for rec, sim, fuzzy in zip(cached_records, similarities, fuzzy_scores)]
        combined.sort(key=lambda x: (x[1], x[2]), reverse=True)
        supabase_results = [
            {**rec, "image_url": rec.get("image_url", None), "type": "supabase"}
            for rec, sim, fuzzy in combined if sim > 0.1 or fuzzy > 60
        ]
        main_results.extend(supabase_results)

    # ---------------- Wikimedia ----------------
    try:
        wikimedia_url = (
            "https://en.wikipedia.org/w/api.php?"
            f"action=query&generator=search&gsrsearch={q}&"
            "prop=pageimages|extracts&exintro&explaintext&format=json&pithumbsize=200"
        )
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(wikimedia_url)
            data = resp.json()
            if "query" in data and "pages" in data["query"]:
                for page_data in data["query"]["pages"].values():
                    main_results.append({
                        "title": page_data.get("title"),
                        "extract": page_data.get("extract", ""),
                        "image_url": page_data.get("thumbnail", {}).get("source"),
                        "page_url": f"https://en.wikipedia.org/?curid={page_data.get('pageid')}",
                        "type": "wiki"
                    })
    except Exception as e:
        logger.error(f"Wikimedia API error: {e}")

    # ---------------- Pagination ----------------
    start = (page - 1) * limit
    end = start + limit
    paginated_results = main_results[start:end]

    return {
        "page": page,
        "limit": limit,
        "total_results": len(main_results),
        "results": paginated_results,
        "dictionary": dictionary_results
    }

# ---------------- Add Link ----------------
@app.post("/add")
def add_link(data: dict = Body(...)):
    if not data.get("title") or not data.get("url"):
        raise HTTPException(status_code=400, detail="Title and URL required")
    data["timestamp"] = datetime.utcnow().isoformat()
    try:
        response = supabase.from_("veloxg").insert(data).execute()
        refresh_cache(force=True)
        return {"message": "Link added", "data": response.data}
    except Exception as e:
        logger.error(f"Error adding link: {e}")
        raise HTTPException(status_code=500, detail="Database insert failed")
