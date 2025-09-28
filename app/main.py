from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from app.middleware.logging_middleware import LoggingMiddleware
from app.utils.handler import run_diarization

app = FastAPI(
    title="Template",
    swagger_ui_parameters={"syntaxHighlight": False}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/diarize")
async def diarize(audio: UploadFile = File(...)):
    # Accept only certain audio types if needed
    content = await audio.read()
    try:
        result = run_diarization(content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)