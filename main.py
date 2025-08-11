from fastapi import FastAPI, Query, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client
import os, asyncio
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import httpx
from typing import List, Optional, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veloxg")
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
YOUTUBE_API_KEYS = [k.strip() for k in os.getenv("YOUTUBE_API_KEYS", "").split(",") if k.strip()]

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing required env vars: SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="VeloxG API", version="1.6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

cached_records: List[Dict[str, Any]] = []
cached_texts: List[str] = []
cached_vectorizer: Optional[TfidfVectorizer] = None
cached_matrix = None
search_cache: Dict[str, Dict[str, Any]] = {}
rate_limit_data: Dict[str, List[float]] = {}
RATE_LIMIT, RATE_LIMIT_PERIOD = 10, 60  # 10 req/minute

def refresh_cache():
    global cached_records, cached_texts, cached_vectorizer, cached_matrix
    try:
        res = supabase.from_("veloxg").select("*").execute()
        cached_records = res.data or []
        cached_texts = [f"{(r.get('title') or '').strip()} {(r.get('meta_description') or '').strip()}"
                        for r in cached_records]
        if not any(t.strip() for t in cached_texts):
            cached_records = [{"title": "VeloxG Default", "meta_description": "Default description"}]
            cached_texts = ["VeloxG Default Default description"]
        cached_vectorizer = TfidfVectorizer().fit(cached_texts)
        cached_matrix = cached_vectorizer.transform(cached_texts)
        logger.info(f"Cache loaded: {len(cached_records)} records")
    except Exception as e:
        logger.error(f"Cache error: {e}")
        cached_records, cached_texts, cached_vectorizer, cached_matrix = [], [], None, None

refresh_cache()

async def get_dictionary_meaning(word: str):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    out = []
                    for m in data[0].get("meanings", []):
                        pos = m.get("partOfSpeech", "")
                        for d in m.get("definitions", []):
                            out.append(f"{pos}: {d.get('definition', '')}")
                    return out[:3]
    except Exception as e:
        logger.error(f"Dictionary error: {e}")
    return None

def rate_limiter(ip: str) -> bool:
    import time
    now = time.time()
    ts = [t for t in rate_limit_data.get(ip, []) if now - t < RATE_LIMIT_PERIOD]
    if len(ts) >= RATE_LIMIT: return False
    ts.append(now); rate_limit_data[ip] = ts
    return True

def batch_scores(query: str):
    if not cached_vectorizer or cached_matrix is None: return []
    q_vec = cached_vectorizer.transform([query])
    tfidf_scores = np.array(cached_matrix.dot(q_vec.T)).flatten()
    fuzzy_scores = np.array([fuzz.token_set_ratio(query, txt)/100 for txt in cached_texts])
    return tfidf_scores + fuzzy_scores

@app.middleware("http")
async def limiter(request: Request, call_next):
    if not rate_limiter(request.client.host):
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})
    return await call_next(request)

@app.get("/") 
def home(): return {"msg": "Welcome to VeloxG API"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.6.0", "records": len(cached_records)}

@app.post("/add")
def add_link(data: dict = Body(...)):
    if not data.get("title") or not data.get("url"):
        raise HTTPException(400, "Title & URL required")
    data["timestamp"] = datetime.utcnow().isoformat()
    try:
        supabase.from_("veloxg").insert(data).execute()
        refresh_cache()
        return {"msg": "Link added"}
    except Exception as e:
        logger.error(e)
        raise HTTPException(500, str(e))

@app.get("/refresh_cache")
def manual_refresh(): refresh_cache(); return {"msg": "Cache refreshed"}

@app.get("/search")
async def search(q: Optional[str] = Query(None), queries: Optional[str] = Query(None),
                 page: int = 1, per_page: int = 10):
    search_query = q or queries
    if not search_query: raise HTTPException(400, "Query required")
    ck = f"{search_query.lower()}_{page}_{per_page}"
    if ck in search_cache and (datetime.utcnow() - search_cache[ck]["ts"]) < timedelta(minutes=5):
        return search_cache[ck]["res"]
    dict_task = None
    if len(search_query.split()) == 1 and len(search_query) <= 20:
        dict_task = asyncio.create_task(get_dictionary_meaning(search_query))
    scores = batch_scores(search_query)
    results = [(rec, scores[i]) for i, rec in enumerate(cached_records) if scores[i] > 0.05]
    results.sort(key=lambda x: x[1], reverse=True)
    page_res = [{**r, "image_url": r.get("image_url")} for r, _ in results[(page-1)*per_page: page*per_page]]
    resp = {"query": search_query, "page": page, "total_results": len(results),
            "results": page_res, "dictionary": await dict_task if dict_task else None}
    search_cache[ck] = {"ts": datetime.utcnow(), "res": resp}
    return resp

@app.get("/youtube_search")
async def youtube_search(query: str = Query(...)):
    if not YOUTUBE_API_KEYS: raise HTTPException(500, "No YouTube keys")
    async with httpx.AsyncClient() as client:
        for key in YOUTUBE_API_KEYS:
            try:
                params = {"part": "snippet", "q": query, "type": "video", "maxResults": 5, "key": key}
                r = await client.get("https://www.googleapis.com/youtube/v3/search", params=params)
                if r.status_code == 200: return {"results": r.json().get("items", [])}
                if r.status_code == 403: continue
            except Exception as e: logger.error(e)
    return {"msg": "YouTube quota exceeded or failed"}

@app.get("/search_all")
async def search_all(query: str = Query(...), page: int = 1, per_page: int = 10):
    dict_task = None
    if len(query.split()) == 1 and len(query) <= 20:
        dict_task = asyncio.create_task(get_dictionary_meaning(query))
    scores = batch_scores(query)
    results = [(rec, scores[i]) for i, rec in enumerate(cached_records) if scores[i] > 0.05]
    results.sort(key=lambda x: x[1], reverse=True)
    supa_page = [{**r, "image_url": r.get("image_url")} for r, _ in results[(page-1)*per_page: page*per_page]]
    yt_task = asyncio.create_task(youtube_search(query))
    return {
        "query": query, "page": page, "per_page": per_page,
        "dictionary": await dict_task if dict_task else None,
        "supabase_results": supa_page,
        "youtube_results": (await yt_task).get("results", [])
    }
