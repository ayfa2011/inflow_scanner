from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from inflow_scanner import run_inflow_scanner

app = FastAPI(title="Inflow Scanner Live API")

# Google Apps Script uthama samparkakagi CORS enable madide
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "online", "message": "Inflow Scanner Live API is Running!"}

@app.get("/api/scan")
def get_scan_results(capital: float = Query(default=30000.0, gt=0, le=100000000)):
    results = run_inflow_scanner(capital_per_trade=capital)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
