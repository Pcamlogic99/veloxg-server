from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Confirm Supabase credentials
if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase credentials missing. Please check your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize app
app = FastAPI()

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
def health_check():
    return {"status": "✅ API is running"}

# Data fetching route
@app.get("/data")
def get_data():
    try:
        print("🔍 Supabase URL:", SUPABASE_URL)
        print("🔐 Supabase Key is set:", bool(SUPABASE_KEY))
        result = supabase.table("veloxg").select("*").execute()
        print("📦 Data:", result.data)
        return {"data": result.data}
    except Exception as e:
        print("❌ Supabase Fetch Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
