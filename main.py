
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

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veloxg_optimized")

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
    for i in range(1, 6):  # fewer extras by default
        extra_key = os.getenv(f"YOUTUBE_API_KEYS{i}")
        if extra_key:
            keys.append(extra_key.strip())
    return keys

YOUTUBE_API_KEYS = load_youtube_keys()

# ---------------- App Config ----------------
app = FastAPI(title="VeloxG API (optimized)", version="3.2.1-optimized")

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

# ---------------- Custom Rate Limiting (fixed) ----------------
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
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        request_times = [t for t in rate_limit_store[ip] if now - t < period]
        if len(request_times) >= limit:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        request_times.append(now)
        rate_limit_store[ip] = request_times
    return await call_next(request)

# ---------------- Global Async Controls ----------------
# Limit total concurrent external HTTP calls to avoid resource spike
GLOBAL_HTTP_SEMAPHORE = asyncio.Semaphore(6)  # tune down for small containers
HTTPX_CLIENT_LIMITS = httpx.Limits(max_keepalive_connections=10, max_connections=20)
HTTPX_TIMEOUT = httpx.Timeout(5.0, connect=3.0)

# ---------------- Cache ----------------
cached_records: List[Dict[str, Any]] = []
cached_texts: List[str] = []
cached_vectorizer: Optional[TfidfVectorizer] = None
cached_doc_vectors = None  # precomputed doc vectors matrix
last_cache_time = datetime.min
CACHE_TTL = timedelta(minutes=5)
TFIDF_MAX_RECORDS = 800  # if records exceed this, skip TF-IDF to save memory

# ---------------- YouTube Cache ----------------
youtube_cache = {}
YOUTUBE_CACHE_TTL = timedelta(minutes=10)

# ---------------- Helper: safe http fetch with global semaphore ----------------
async def _fetch_json(client: httpx.AsyncClient, url: str, params: dict = None) -> Any:
    async with GLOBAL_HTTP_SEMAPHORE:
        try:
            resp = await client.get(url, params=params, timeout=HTTPX_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.debug(f"Non-200 from {url}: {resp.status_code}")
                return {"error": f"Status {resp.status_code}"}
        except Exception as e:
            logger.debug(f"Error fetching {url}: {e}")
            return {"error": str(e)}

# ---------------- Refresh cache (precompute doc vectors) ----------------
def refresh_cache(force=False):
    global cached_records, cached_texts, cached_vectorizer, cached_doc_vectors, last_cache_time
    if not force and datetime.utcnow() - last_cache_time < CACHE_TTL:
        return
    try:
        logger.info("Refreshing cache from Supabase (optimized)...")
        result = supabase.from_("veloxg").select("*").execute()
        records = result.data or []
        texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in records]
        cached_records = records
        cached_texts = texts
        if texts and len(texts) <= TFIDF_MAX_RECORDS:
            cached_vectorizer = TfidfVectorizer().fit(texts)
            # precompute doc vectors (sparse)
            cached_doc_vectors = cached_vectorizer.transform(texts)
        else:
            cached_vectorizer = None
            cached_doc_vectors = None
        last_cache_time = datetime.utcnow()
        logger.info(f"Cached {len(cached_records)} records (tfidf_enabled={cached_vectorizer is not None})")
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        cached_records, cached_texts, cached_vectorizer, cached_doc_vectors = [], [], None, None

refresh_cache(force=True)

# ---------------- Dictionary Lookup (unchanged but limited) ----------------
async def get_dictionary_meaning(word: str):
    try:
        async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS, timeout=HTTPX_TIMEOUT) as client:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            async with GLOBAL_HTTP_SEMAPHORE:
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
        logger.debug(f"Dictionary API error: {e}")
        return None

# ---------------- Input Validation ----------------
def sanitize_query(q: str) -> str:
    q = q.strip()
    if len(q) > 100:
        raise HTTPException(status_code=400, detail="Query too long")
    if not re.match(r"^[\w\s\-\.,!?']+$", q):
        raise HTTPException(status_code=400, detail="Invalid characters in query")
    return q

# ---------------- YouTube search (limited attempts) ----------------
async def fetch_youtube(q: str, max_results: int = 4):
    if not YOUTUBE_API_KEYS:
        return []
    now = datetime.utcnow()
    cached = youtube_cache.get(q)
    if cached and now - cached["timestamp"] < YOUTUBE_CACHE_TTL:
        return cached["data"]
    results = []
    async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS, timeout=HTTPX_TIMEOUT) as client:
        for key in YOUTUBE_API_KEYS:
            try:
                async with GLOBAL_HTTP_SEMAPHORE:
                    resp = await client.get(
                        "https://www.googleapis.com/youtube/v3/search",
                        params={"part": "snippet", "q": q, "type": "video", "maxResults": max_results, "key": key},
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
                                "description": snippet.get("description"),
                                "channel": snippet.get("channelTitle"),
                                "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                                "type": "youtube"
                            })
                    # store and break after first working key
                    youtube_cache[q] = {"data": results, "timestamp": now}
                    break
                else:
                    logger.debug(f"YouTube non-200 {resp.status_code} for key {key}")
            except Exception as e:
                logger.debug(f"YouTube error with key {key}: {e}")
                continue
    return results

# ---------------- Football helper (limited) ----------------
async def get_football_data_for_query(q: str):
    base_sportsdb = "https://www.thesportsdb.com/api/v1/json/3"
    endpoints = {
        "team_info": f"{base_sportsdb}/searchteams.php",
        "last_events": f"{base_sportsdb}/eventslast.php",
        "next_events": f"{base_sportsdb}/eventsnext.php",
    }
    results = {}
    async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS, timeout=HTTPX_TIMEOUT) as client:
        tasks = [
            _fetch_json(client, endpoints["team_info"], params={"t": q}),
            _fetch_json(client, endpoints["last_events"], params={"t": q}),
            _fetch_json(client, endpoints["next_events"], params={"t": q}),
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        # filter and attach
        keys = ["team_info", "last_events", "next_events"]
        for k, resp in zip(keys, responses):
            if isinstance(resp, Exception):
                results[k] = {"error": str(resp)}
            else:
                results[k] = resp
    return results

# ---------------- Location helper (reduced requests) ----------------
async def get_location_data_for_query(q: str, max_images: int = 2):
    results = {}
    async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS, timeout=HTTPX_TIMEOUT) as client:
        nominatim_city_url = "https://nominatim.openstreetmap.org/search"
        nom_params = {"city": q, "format": "json", "limit": 1}
        nom_data = await _fetch_json(client, nominatim_city_url, params=nom_params)
        if isinstance(nom_data, dict) and nom_data.get("error"):
            nom_params2 = {"q": q, "format": "json", "limit": 1}
            nom_data = await _fetch_json(client, nominatim_city_url, params=nom_params2)
        results["nominatim"] = nom_data

        lat = lon = None
        if isinstance(nom_data, list) and nom_data:
            lat = nom_data[0].get("lat")
            lon = nom_data[0].get("lon")

        if lat and lon:
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {"latitude": lat, "longitude": lon, "current_weather": "true"}
            weather_data = await _fetch_json(client, weather_url, params=weather_params)
            results["weather"] = weather_data

        # Wiki summary (attempt a single call)
        try:
            wiki_title = q.replace(" ", "_")
            wiki_summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}"
            wiki_summary = await _fetch_json(client, wiki_summary_url)
            results["wikipedia_summary"] = wiki_summary
        except Exception as e:
            logger.debug(f"Wikipedia summary error: {e}")

        # Wikimedia Commons images (one small call)
        try:
            commons_url = "https://commons.wikimedia.org/w/api.php"
            commons_params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": q,
                "gsrlimit": 4,
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json"
            }
            commons_resp = await _fetch_json(client, commons_url, params=commons_params)
            images = []
            if isinstance(commons_resp, dict) and "query" in commons_resp and "pages" in commons_resp["query"]:
                for page in commons_resp["query"]["pages"].values():
                    imageinfo = page.get("imageinfo")
                    if imageinfo and isinstance(imageinfo, list):
                        url = imageinfo[0].get("url")
                        if url and url.lower().endswith((".jpg", ".jpeg", ".png", ".svg")):
                            images.append(url)
                    if len(images) >= max_images:
                        break
            results["images"] = images
        except Exception as e:
            logger.debug(f"Wikimedia images error: {e}")
            results["images"] = []

    return results

# ---------------- Wikipedia quick search (limited pages) ----------------
async def fetch_wikipedia_search(q: str, max_pages: int = 2):
    wikires = []
    try:
        wikimedia_url = (
            "https://en.wikipedia.org/w/api.php"
        )
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": q,
            "gsrlimit": max_pages,
            "prop": "pageimages|extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json",
            "pithumbsize": 200
        }
        async with httpx.AsyncClient(limits=HTTPX_CLIENT_LIMITS, timeout=HTTPX_TIMEOUT) as client:
            data = await _fetch_json(client, wikimedia_url, params=params)
            if isinstance(data, dict) and "query" in data and "pages" in data["query"]:
                for page_data in data["query"]["pages"].values():
                    title = page_data.get("title")
                    extract = page_data.get("extract", "")
                    thumb = page_data.get("thumbnail", {}).get("source")
                    year = None
                    m = re.search(r"\b(18|19|20)\d{2}\b", extract or "")
                    if m:
                        year = m.group(0)
                    images = []
                    if thumb:
                        images.append(thumb)
                    wikires.append({
                        "title": title,
                        "extract": extract,
                        "about": extract.split("\n", 1)[0] if extract else "",
                        "year": year,
                        "images": images,
                        "image_url": thumb,
                        "wikipedia_url": f"https://en.wikipedia.org/?curid={page_data.get('pageid')}",
                        "type": "wiki"
                    })
    except Exception as e:
        logger.debug(f"Wikimedia search error: {e}")
    return wikires

# ---------------- Main Search Endpoint (optimized) ----------------
@app.get("/search")
async def search(
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(15, ge=1, le=50),
    include: Optional[str] = Query(None, description="comma-separated sources: youtube,wiki,football,location,supabase"),
):
    if not q:
        raise HTTPException(status_code=400, detail="Search query required")
    q = sanitize_query(q)
    refresh_cache()

    # parse include param (default to essential small set)
    allowed_sources = {"youtube", "wiki", "football", "location", "supabase"}
    if include:
        requested = set([s.strip().lower() for s in include.split(",") if s.strip()])
        sources = list(requested & allowed_sources)
        if not sources:
            sources = ["supabase", "wiki"]  # fallback
    else:
        # default: keep lightweight to conserve resources
        sources = ["supabase", "wiki"]

    main_results = []

    # dictionary lookup (only for single short words)
    dictionary_results = None
    if len(q.split()) == 1 and len(q) <= 20:
        dictionary_results = await get_dictionary_meaning(q)

    # ---------------- YouTube (optional) ----------------
    if "youtube" in sources and YOUTUBE_API_KEYS:
        try:
            videos_data = await fetch_youtube(q, max_results=4)
            main_results.extend(videos_data)
        except Exception as e:
            logger.debug(f"YouTube overall error: {e}")

    # ---------------- Supabase results (TF-IDF if enabled) ----------------
    try:
        if cached_texts and cached_vectorizer and cached_doc_vectors is not None:
            query_vec = cached_vectorizer.transform([q])
            # compute cosine-ish similarities via dot product (sparse)
            similarities = (cached_doc_vectors * query_vec.T).toarray().flatten()
            fuzzy_scores = [fuzz.token_set_ratio(q, text) for text in cached_texts]
            combined = [(rec, sim, fuzzy) for rec, sim, fuzzy in zip(cached_records, similarities, fuzzy_scores)]
            combined.sort(key=lambda x: (x[1], x[2]), reverse=True)
            supabase_results = [
                {**rec, "image_url": rec.get("image_url", None), "type": "supabase"}
                for rec, sim, fuzzy in combined if sim > 0.05 or fuzzy > 60
            ]
        else:
            # fallback: simple substring match (cheap)
            supabase_results = []
            qlow = q.lower()
            for rec in cached_records:
                t = f"{rec.get('title','')} {rec.get('meta_description','')}".lower()
                if qlow in t:
                    supabase_results.append({**rec, "type": "supabase"})
        main_results.extend(supabase_results)
    except Exception as e:
        logger.debug(f"Supabase scoring error: {e}")

    # ---------------- Wikipedia (limited) ----------------
    if "wiki" in sources:
        try:
            wiki_pages = await fetch_wikipedia_search(q, max_pages=2)
            main_results.extend(wiki_pages)
        except Exception as e:
            logger.debug(f"Wikipedia fetch error: {e}")

    # ---------------- Football (optional, limited) ----------------
    if "football" in sources:
        try:
            football_data = await get_football_data_for_query(q)
            main_results.append({
                "type": "football",
                "title": f"Football data for {q}",
                "football_data": football_data
            })
        except Exception as e:
            logger.debug(f"Football API error: {e}")

    # ---------------- Location / Place Data (optional, limited) ----------------
    if "location" in sources:
        try:
            location_data = await get_location_data_for_query(q, max_images=2)
            if location_data:
                main_results.append({
                    "type": "location",
                    "title": f"Location data for {q}",
                    "location_data": location_data
                })
        except Exception as e:
            logger.debug(f"Location API error: {e}")

    # ---------------- Pagination ----------------
    start = (page - 1) * limit
    end = start + limit
    paginated_results = main_results[start:end]

    return {
        "page": page,
        "limit": limit,
        "total_results": len(main_results),
        "results": paginated_results,
        "dictionary": dictionary_results,
        "used_sources": sources
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
        return {"message": "Li