# main.py
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
import urllib.parse
import re
from typing import Optional, List, Dict, Any
from collections import defaultdict
import time
import asyncio
from functools import lru_cache
import signal
import sys
import psutil

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("veloxg_prod")

# ---------------- Load ENV ----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PORT = int(os.getenv("PORT", 3000))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Load YouTube Keys ----------------
def load_youtube_keys():
    keys = []
    main_key = os.getenv("YOUTUBE_API_KEYS")
    if main_key:
        keys.extend([k.strip() for k in main_key.split(",") if k.strip()])
    for i in range(1, 6):
        extra_key = os.getenv(f"YOUTUBE_API_KEYS{i}")
        if extra_key:
            keys.append(extra_key.strip())
    return keys

YOUTUBE_API_KEYS = load_youtube_keys()

# ---------------- App Config ----------------
app = FastAPI(title="VeloxG Search API", version="3.3.1-Prod")

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

# ---------------- Exception Middleware ----------------
@app.middleware("http")
async def exception_handler(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        logger.error(f"Unhandled Exception: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

# ---------------- Rate Limiting ----------------
RATE_LIMITS = {"/search": (20, 60), "/add": (5, 60)}
rate_limit_store = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path in RATE_LIMITS:
        limit, period = RATE_LIMITS[path]
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        times = [t for t in rate_limit_store[ip] if now - t < period]
        if len(times) >= limit:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        times.append(now)
        rate_limit_store[ip] = times
    return await call_next(request)

# ---------------- Async Controls ----------------
GLOBAL_HTTP_SEMAPHORE = asyncio.Semaphore(6)
HTTPX_CLIENT_LIMITS = httpx.Limits(max_keepalive_connections=10, max_connections=20)
HTTPX_TIMEOUT = httpx.Timeout(5.0, connect=3.0)

# ---------------- Cache ----------------
cached_records = []
cached_texts = []
cached_vectorizer = None
cached_doc_vectors = None
last_cache_time = datetime.min
CACHE_TTL = timedelta(minutes=5)
TFIDF_MAX_RECORDS = 800

def refresh_cache(force=False):
    global cached_records, cached_texts, cached_vectorizer, cached_doc_vectors, last_cache_time
    if not force and datetime.utcnow() - last_cache_time < CACHE_TTL:
        return
    try:
        logger.info("Refreshing cache from Supabase...")
        result = supabase.from_("veloxg").select("*").execute()
        records = result.data or []
        texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in records]
        cached_records = records
        cached_texts = texts
        if texts and len(texts) <= TFIDF_MAX_RECORDS:
            cached_vectorizer = TfidfVectorizer().fit(texts)
            cached_doc_vectors = cached_vectorizer.transform(texts)
        else:
            cached_vectorizer = None
            cached_doc_vectors = None
        last_cache_time = datetime.utcnow()
        logger.info(f"Cached {len(cached_records)} records (TFIDF={'on' if cached_vectorizer else 'off'})")
    except Exception as e:
        logger.error(f"Cache refresh error: {e}")
        cached_records, cached_texts, cached_vectorizer, cached_doc_vectors = [], [], None, None

refresh_cache(force=True)

# ---------------- Helper ----------------
async def _fetch_json(client, url, params=None):
    async with GLOBAL_HTTP_SEMAPHORE:
        try:
            r = await client.get(url, params=params, timeout=HTTPX_TIMEOUT)
            return r.json() if r.status_code == 200 else {"error": f"Status {r.status_code}"}
        except Exception as e:
            return {"error": str(e)}

# ---------------- Dictionary ----------------
@lru_cache(maxsize=128)
async def get_dictionary_meaning(word: str):
    try:
        async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS) as client:
            r = await client.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    out = []
                    for meaning in data[0].get("meanings", []):
                        part = meaning.get("partOfSpeech", "")
                        defs = [d.get("definition", "") for d in meaning.get("definitions", [])]
                        out.extend([f"{part}: {d}" for d in defs])
                    return out[:3]
    except Exception:
        pass
    return None

# ---------------- YouTube ----------------
youtube_cache = {}
YOUTUBE_CACHE_TTL = timedelta(minutes=10)

async def fetch_youtube(q: str, max_results: int = 4):
    now = datetime.utcnow()
    cached = youtube_cache.get(q)
    if cached and now - cached["timestamp"] < YOUTUBE_CACHE_TTL:
        return cached["data"]
    results = []
    async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS) as client:
        for key in YOUTUBE_API_KEYS:
            try:
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={"part": "snippet", "q": q, "type": "video", "maxResults": max_results, "key": key},
                    timeout=HTTPX_TIMEOUT,
                )
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    for video in items:
                        vid = video.get("id", {}).get("videoId")
                        snippet = video.get("snippet", {})
                        if vid:
                            results.append({
                                "video_id": vid,
                                "title": snippet.get("title"),
                                "channel": snippet.get("channelTitle"),
                                "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                                "type": "youtube"
                            })
                    youtube_cache[q] = {"data": results, "timestamp": now}
                    break
            except Exception:
                continue
    return results

# ---------------- Wikipedia ----------------
async def fetch_wikipedia(q: str):
    wikires = []
    async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS) as client:
        data = await _fetch_json(client, "https://en.wikipedia.org/w/api.php", params={
            "action": "query", "generator": "search", "gsrsearch": q,
            "gsrlimit": 2, "prop": "pageimages|extracts",
            "exintro": True, "explaintext": True, "format": "json", "pithumbsize": 200
        })
        if "query" in data and "pages" in data["query"]:
            for p in data["query"]["pages"].values():
                wikires.append({
                    "title": p.get("title"),
                    "extract": p.get("extract"),
                    "image": p.get("thumbnail", {}).get("source"),
                    "type": "wiki"
                })
    return wikires

# ---------------- Football ----------------
async def get_football_data_for_query(q: str):
    base = "https://www.thesportsdb.com/api/v1/json/3"
    endpoints = {
        "team_info": f"{base}/searchteams.php",
        "last_events": f"{base}/eventslast.php",
        "next_events": f"{base}/eventsnext.php",
    }
    async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS) as client:
        tasks = [_fetch_json(client, url, params={"t": q}) for url in endpoints.values()]
        results = await asyncio.gather(*tasks)
        return dict(zip(endpoints.keys(), results))

# ---------------- Location ----------------
async def get_location_data_for_query(q: str):
    res = {}
    async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS) as client:
        nom = await _fetch_json(client, "https://nominatim.openstreetmap.org/search", params={"q": q, "format": "json", "limit": 1})
        res["nominatim"] = nom
        if isinstance(nom, list) and nom:
            lat, lon = nom[0].get("lat"), nom[0].get("lon")
            w = await _fetch_json(client, "https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "current_weather": "true"})
            res["weather"] = w
    return res

# ---------------- Routes ----------------
@app.get("/")
def root():
    return {"message": "VeloxG API running successfully", "version": "3.3.1-Prod"}

@app.get("/health")
def health():
    return {"status": "healthy", "records_cached": len(cached_records)}

@app.get("/status")
def status():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    cpu = psutil.cpu_percent(interval=0.1)
    uptime = (datetime.utcnow() - last_cache_time).total_seconds() / 60
    return {"cpu": cpu, "mem_mb": round(mem, 2), "cache": len(cached_records), "uptime_min": round(uptime, 1)}

@app.get("/search")
async def search(q: str = Query(...), include: Optional[str] = Query(None)):
    q = q.strip()
    refresh_cache()
    sources = set(include.split(",")) if include else {"supabase", "wiki"}
    results = []

    # Supabase results
    if cached_vectorizer:
        qv = cached_vectorizer.transform([q])
        sims = (cached_doc_vectors * qv.T).toarray().flatten()
        fuzzy = [fuzz.token_set_ratio(q, t) for t in cached_texts]
        combined = [(r, s, f) for r, s, f in zip(cached_records, sims, fuzzy)]
        combined.sort(key=lambda x: (x[1], x[2]), reverse=True)
        results += [{**r, "type": "supabase"} for r, s, f in combined[:10] if s > 0.05 or f > 60]
    else:
        for rec in cached_records:
            if q.lower() in str(rec.get("title", "")).lower():
                results.append({**rec, "type": "supabase"})

    tasks = []
    if "youtube" in sources: tasks.append(fetch_youtube(q))
    if "wiki" in sources: tasks.append(fetch_wikipedia(q))
    if "football" in sources: tasks.append(get_football_data_for_query(q))
    if "location" in sources: tasks.append(get_location_data_for_query(q))

    external_results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in external_results:
        if isinstance(res, dict):
            results.append(res)
        elif isinstance(res, list):
            results.extend(res)

    return {"query": q, "count": len(results), "results": results}

@app.post("/add")
def add_link(data: dict = Body(...)):
    if not data.get("title") or not data.get("url"):
        raise HTTPException(status_code=400, detail="Title and URL required")
    data["timestamp"] = datetime.utcnow().isoformat()
    try:
        res = supabase.from_("veloxg").insert(data).execute()
        refresh_cache(force=True)
        return {"message": "Link added", "data": res.data}
    except Exception as e:
        logger.error(f"Add link error: {e}")
        raise HTTPException(status_code=500, detail="Database insert failed")

# ---------------- Graceful Shutdown ----------------
def handle_exit(*_):
    logger.info("Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ---------------- Local Run ----------------
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting VeloxG API on port {PORT} | YouTube keys: {len(YOUTUBE_API_KEYS)}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
