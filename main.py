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

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veloxg")

# ---------------- ENV ----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PORT = int(os.getenv("PORT", 3000))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- YouTube Keys ----------------
def load_youtube_keys():
    keys = []
    main_key = os.getenv("YOUTUBE_API_KEYS")
    if main_key:
        keys.extend([k.strip() for k in main_key.split(",") if k.strip()])
    for i in range(1, 5):
        extra_key = os.getenv(f"YOUTUBE_API_KEYS{i}")
        if extra_key:
            keys.append(extra_key.strip())
    return keys

YOUTUBE_API_KEYS = load_youtube_keys()

# ---------------- App Config ----------------
app = FastAPI(title="VeloxG API Optimized", version="3.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Rate Limit ----------------
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

# ---------------- Cache ----------------
cached_records = []
cached_texts = []
cached_vectorizer = None
cached_doc_vectors = None
last_cache_time = datetime.min
CACHE_TTL = timedelta(minutes=5)
TFIDF_MAX = 600

def refresh_cache(force=False):
    global cached_records, cached_texts, cached_vectorizer, cached_doc_vectors, last_cache_time
    if not force and datetime.utcnow() - last_cache_time < CACHE_TTL:
        return
    try:
        logger.info("Refreshing Supabase cache...")
        result = supabase.from_("veloxg").select("*").execute()
        records = result.data or []
        texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in records]
        cached_records, cached_texts = records, texts
        if texts and len(texts) < TFIDF_MAX:
            cached_vectorizer = TfidfVectorizer().fit(texts)
            cached_doc_vectors = cached_vectorizer.transform(texts)
        else:
            cached_vectorizer = None
            cached_doc_vectors = None
        last_cache_time = datetime.utcnow()
        logger.info(f"Cache refreshed: {len(records)} records")
    except Exception as e:
        logger.error(f"Cache refresh error: {e}")

refresh_cache(force=True)

# ---------------- Helper ----------------
HTTPX_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
HTTPX_LIMITS = httpx.Limits(max_keepalive_connections=10, max_connections=20)
SEM = asyncio.Semaphore(5)

async def _fetch_json(client, url, params=None):
    async with SEM:
        try:
            r = await client.get(url, params=params, timeout=HTTPX_TIMEOUT)
            return r.json() if r.status_code == 200 else {"error": r.status_code}
        except Exception as e:
            return {"error": str(e)}

# ---------------- YouTube ----------------
youtube_cache = {}
YOUTUBE_CACHE_TTL = timedelta(minutes=10)

async def fetch_youtube(q: str, limit=4):
    now = datetime.utcnow()
    if q in youtube_cache and now - youtube_cache[q]["time"] < YOUTUBE_CACHE_TTL:
        return youtube_cache[q]["data"]
    data = []
    async with httpx.AsyncClient(limits=HTTPX_LIMITS) as c:
        for key in YOUTUBE_API_KEYS:
            try:
                res = await c.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={"part": "snippet", "q": q, "type": "video", "maxResults": limit, "key": key},
                    timeout=5,
                )
                if res.status_code == 200:
                    for v in res.json().get("items", []):
                        data.append({
                            "video_id": v["id"]["videoId"],
                            "title": v["snippet"]["title"],
                            "channel": v["snippet"]["channelTitle"],
                            "thumbnail": v["snippet"]["thumbnails"]["high"]["url"],
                            "type": "youtube"
                        })
                    break
            except Exception as e:
                logger.debug(f"YouTube error: {e}")
    youtube_cache[q] = {"data": data, "time": now}
    return data

# ---------------- Wikipedia ----------------
async def fetch_wiki(q: str):
    try:
        async with httpx.AsyncClient(limits=HTTPX_LIMITS) as c:
            url = "https://en.wikipedia.org/w/api.php"
            p = {
                "action": "query", "generator": "search", "gsrsearch": q,
                "gsrlimit": 2, "prop": "pageimages|extracts", "exintro": True,
                "explaintext": True, "format": "json", "pithumbsize": 200
            }
            data = await _fetch_json(c, url, params=p)
            res = []
            if "query" in data and "pages" in data["query"]:
                for p in data["query"]["pages"].values():
                    res.append({
                        "title": p.get("title"),
                        "extract": p.get("extract"),
                        "thumb": p.get("thumbnail", {}).get("source"),
                        "type": "wiki"
                    })
            return res
    except Exception as e:
        logger.debug(f"Wiki error: {e}")
        return []

# ---------------- Location ----------------
async def fetch_location(q: str):
    results = {}
    async with httpx.AsyncClient(limits=HTTPX_LIMITS) as c:
        n = await _fetch_json(c, "https://nominatim.openstreetmap.org/search",
                              params={"q": q, "format": "json", "limit": 1})
        results["nominatim"] = n
        if isinstance(n, list) and n:
            lat, lon = n[0]["lat"], n[0]["lon"]
            w = await _fetch_json(c, "https://api.open-meteo.com/v1/forecast",
                                  params={"latitude": lat, "longitude": lon, "current_weather": "true"})
            results["weather"] = w
    return results

# ---------------- Routes ----------------
@app.get("/")
def home():
    return {"message": "VeloxG API is running", "status": "ok"}

@app.get("/health")
def health_check():
    """Used by Koyeb to verify the service is alive"""
    return {
        "status": "healthy",
        "version": "3.2.3",
        "cached_records": len(cached_records),
        "supabase_connected": bool(SUPABASE_URL and SUPABASE_KEY)
    }

@app.get("/search")
async def search(q: str = Query(...), include: Optional[str] = Query(None)):
    q = q.strip()
    refresh_cache()
    sources = ["supabase", "wiki"]
    if include:
        allowed = {"youtube", "wiki", "location", "supabase"}
        s = [x for x in include.split(",") if x in allowed]
        if s:
            sources = s

    results = []
    # Supabase search
    if "supabase" in sources and cached_records:
        try:
            if cached_vectorizer:
                qv = cached_vectorizer.transform([q])
                sims = (cached_doc_vectors * qv.T).toarray().flatten()
                combined = [
                    (rec, sim, fuzz.token_set_ratio(q, t))
                    for rec, sim, t in zip(cached_records, sims, cached_texts)
                ]
                combined.sort(key=lambda x: (x[1], x[2]), reverse=True)
                for rec, sim, fz in combined[:15]:
                    if sim > 0.05 or fz > 60:
                        results.append({**rec, "type": "supabase"})
            else:
                for rec in cached_records:
                    if q.lower() in str(rec.get("title", "")).lower():
                        results.append({**rec, "type": "supabase"})
        except Exception as e:
            logger.error(f"Supabase match error: {e}")

    if "wiki" in sources:
        results += await fetch_wiki(q)

    if "youtube" in sources:
        results += await fetch_youtube(q)

    if "location" in sources:
        loc = await fetch_location(q)
        results.append({"type": "location", "data": loc})

    return {"results": results, "count": len(results), "sources": sources}

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

# ---------------- Run (for local dev) ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
