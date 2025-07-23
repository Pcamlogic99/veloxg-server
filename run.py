from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to veloxg Search API"}

@app.get("/search")
def search(query: str = Query(...)):
    return {"result": f"You searched for: {query}"}
