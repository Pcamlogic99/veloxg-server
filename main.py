from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
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
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to VeloxG Search API"}

# Cache records and vectorizer
cached_records = []
cached_texts = []
cached_vectorizer = None

def refresh_cache():
    global cached_records, cached_texts, cached_vectorizer
    result = supabase.from_("veloxg").select("*").execute()
    cached_records = result["data"] if "data" in result else []
    cached_texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in cached_records]
    if cached_texts:
        cached_vectorizer = TfidfVectorizer().fit(cached_texts)

# Refresh cache at startup
refresh_cache()

@app.post("/add")
def add_link(
    data: dict = Body(
        ...,
        example={
            "title": "Example Website",
            "url": "https://example.com",
            "favicon": "https://example.com/favicon.ico",
            "meta_description": "This is an example website",
            "content": "Sample content",
            "image_url": "https://example.com/image.jpg",
            "category": "Example"
        }
    )
):
    if not data.get("title") or not data.get("url"):
        raise HTTPException(status_code=400, detail="Title and URL are required fields")
    data["timestamp"] = datetime.utcnow().isoformat()
    try:
        response = supabase.from_("veloxg").insert(data).execute()
        if "error" in response and response["error"]:
            logger.error(f"Error adding link: {response['error']}")
            raise HTTPException(status_code=500, detail=f"Database error: {response['error']}")
        refresh_cache()
        return {"message": "Link added successfully", "data": response["data"]}
    except Exception as e:
        logger.error(f"Error adding link: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search(q: str = Query(..., min_length=1, description="Search query")):
    if not cached_vectorizer or not cached_texts:
        refresh_cache()
    try:
        logger.info(f"Searching for query: {q}")
        query_vec = cached_vectorizer.transform([q])
        doc_vecs = cached_vectorizer.transform(cached_texts)
        similarities = (doc_vecs * query_vec.T).toarray().flatten()
        fuzzy_scores = [fuzz.token_set_ratio(q, text) for text in cached_texts]
        combined = [
            (rec, sim, fuzzy)
            for rec, sim, fuzzy in zip(cached_records, similarities, fuzzy_scores)
        ]
        combined.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top_results = [
            {**rec, "image_url": rec.get("image_url", None)}
            for rec, sim, fuzzy in combined if sim > 0.1 or fuzzy > 60
        ][:10]
        logger.info(f"Found {len(top_results)} fuzzy/NLP matched records")
        return {"results": top_results}
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data")
def get_data():
    try:
        result = supabase.from_("veloxg").select("*").execute()
        if "error" in result and result["error"]:
            logger.error(f"Error fetching data: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Database error: {result['error']}")
        return {"data": result["data"]}
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/debug")
def debug_info():
    try:
        tables = supabase.from_("veloxg").select("*").execute()
        table_info = {
            "connected": True,
            "table_exists": not ("error" in tables and tables["error"]),
            "record_count": len(tables["data"]) if "data" in tables else 0,
            "supabase_url": SUPABASE_URL[:20] + "..." if SUPABASE_URL else None,
            "has_api_key": bool(SUPABASE_KEY),
            "first_record": tables["data"][0] if "data" in tables and len(tables["data"]) > 0 else None
        }
        return {"debug_info": table_info}
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return {"debug_info": {"error": str(e), "connected": False}
