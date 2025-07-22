from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing required environment variables: SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="VeloxG API",
    description="FastAPI backend API for VeloxG search engine using Supabase",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, set this to specific domains!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for /add
class Link(BaseModel):
    title: str
    url: str
    favicon: Optional[str] = None
    meta_description: Optional[str] = None
    content: Optional[str] = None
    image_url: Optional[str] = None
    category: Optional[str] = None

# Cache
cached_records = []
cached_texts = []
cached_vectorizer = None

def refresh_cache():
    global cached_records, cached_texts, cached_vectorizer
    logger.info("Refreshing cache from Supabase...")

    result = supabase.from_("veloxg").select("*").execute()

    if result.error:
        logger.error(f"Cache refresh error: {result.error}")
        cached_records = []
        cached_texts = []
        cached_vectorizer = None
        return

    cached_records = result.data or []
    cached_texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in cached_records]

    if cached_texts:
        cached_vectorizer = TfidfVectorizer().fit(cached_texts)
        logger.info(f"Cached {len(cached_records)} records.")
    else:
        cached_vectorizer = None
        logger.warning("No texts to cache.")

# Refresh cache at startup
refresh_cache()

@app.get("/")
def home():
    return {"message": "Welcome to VeloxG Search API"}

@app.post("/add")
def add_link(data: Link):
    item = data.dict()
    item["timestamp"] = datetime.utcnow().isoformat()

    try:
        response = supabase.from_("veloxg").insert(item).execute()

        if response.error:
            logger.error(f"Error adding link: {response.error}")
            raise HTTPException(status_code=500, detail=str(response.error))

        refresh_cache()
        return {"message": "Link added successfully", "data": response.data}
    except Exception as e:
        logger.error(f"Unexpected error adding link: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search(q: str = Query(..., min_length=1, description="Search query")):
    if not cached_vectorizer or not cached_texts:
        refresh_cache()

    try:
        logger.info(f"Searching for: {q}")

        query_vec = cached_vectorizer.transform([q])
        doc_vecs = cached_vectorizer.transform(cached_texts)
        similarities = (doc_vecs * query_vec.T).toarray().flatten()

        fuzzy_scores = [fuzz.token_set_ratio(q, text)/100 for text in cached_texts]  # Normalize to 0-1

        combined = [
            (rec, max(sim, fuzzy))  # Use max to pick the stronger match
            for rec, sim, fuzzy in zip(cached_records, similarities, fuzzy_scores)
        ]

        combined.sort(key=lambda x: x[1], reverse=True)

        top_results = [
            {**rec, "image_url": rec.get("image_url", None)}
            for rec, score in combined if score > 0.1
        ][:10]

        logger.info(f"Found {len(top_results)} matching results.")

        return {"results": top_results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data")
def get_data():
    try:
        result = supabase.from_("veloxg").select("*").execute()

        if result.error:
            logger.error(f"Data fetch error: {result.error}")
            raise HTTPException(status_code=500, detail=str(result.error))

        return {"data": result.data}
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/refresh-cache")
def manual_refresh():
    try:
        refresh_cache()
        return {"message": "Cache refreshed"}
    except Exception as e:
        logger.error(f"Cache refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.1.0"}

@app.get("/debug")
def debug_info():
    try:
        tables = supabase.from_("veloxg").select("*").execute()

        return {
            "debug_info": {
                "connected": True,
                "record_count": len(tables.data) if tables.data else 0,
                "supabase_url": SUPABASE_URL[:20] + "...",
                "has_api_key": bool(SUPABASE_KEY),
                "first_record": tables.data[0] if tables.data and len(tables.data) > 0 else None
            }
        }
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return {"debug_info": {"error": str(e), "connected": False}}
