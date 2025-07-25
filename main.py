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
    title="veloxg API",
    description="FastAPI backend API for VeloxG search engine using Supabase",
    version="1.0.0",
root_path="/api"
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
    return {"message": "Welcome to veloxg Search API"}

# Cache
cached_records = []
cached_texts = []
cached_vectorizer = None

def refresh_cache():
    global cached_records, cached_texts, cached_vectorizer
    try:
        result = supabase.from_("veloxg").select("*").execute()
        if hasattr(result, 'error') and result.error:
            logger.error(f"Supabase error: {result.error}")
            cached_records, cached_texts, cached_vectorizer = [], [], None
            return
        cached_records = result.data
        cached_texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in cached_records]
        if cached_texts:
            cached_vectorizer = TfidfVectorizer().fit(cached_texts)
        logger.info(f"Cached {len(cached_records)} records")
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        cached_records, cached_texts, cached_vectorizer = [], [], None

refresh_cache()

@app.post("/add")
def add_link(data: dict = Body(...)):
    if not data.get("title") or not data.get("url"):
        raise HTTPException(status_code=400, detail="Title and URL are required")
    data["timestamp"] = datetime.utcnow().isoformat()
    try:
        response = supabase.from_("veloxg").insert(data).execute()
        if hasattr(response, 'error') and response.error:
            raise HTTPException(status_code=500, detail=f"Database error: {response.error}")
        refresh_cache()
        return {"message": "Link added successfully", "data": response.data}
    except Exception as e:
        logger.error(f"Error adding link: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search(q: str = Query(None), queries: str = Query(None)):
    search_query = q or queries
    if not search_query:
        raise HTTPException(status_code=400, detail="Search query is required")
    try:
        logger.info(f"Searching: {search_query}")
        result = supabase.from_("veloxg").select("*").execute()
        if hasattr(result, 'error') and result.error:
            raise HTTPException(status_code=500, detail=f"Supabase error: {result.error}")
        records = result.data
        texts = [f"{r.get('title', '')} {r.get('meta_description', '')}" for r in records]
        if not texts:
            return {"results": []}
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
        return {"results": top_results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search_data")
async def search_data(q: str = Query(...)):
    try:
        result = supabase.from_("veloxg").select("*").execute()
        records = result.data
        results = []
        for record in records:
            combined = f"{record.get('title', '')} {record.get('meta_description', '')} {record.get('content', '')}"
            score = fuzz.partial_ratio(q.lower(), combined.lower())
            if score >= 60:
                results.append({"record": record, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"query": q, "results": results}
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

@app.get("/data")
def get_data():
    try:
        result = supabase.from_("veloxg").select("*").execute()
        if hasattr(result, 'error') and result.error:
            raise HTTPException(status_code=500, detail=f"Supabase error: {result.error}")
        return {"data": result.data}
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
        return {
            "connected": True,
            "table_exists": not (hasattr(tables, 'error') and tables.error),
            "record_count": len(tables.data) if hasattr(tables, 'data') else 0,
            "supabase_url": SUPABASE_URL[:20] + "...",
            "has_api_key": bool(SUPABASE_KEY),
            "first_record": tables.data[0] if hasattr(tables, 'data') and tables.data else None
        }
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return {"connected": False, "error": str(e)}
