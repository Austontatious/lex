from fastapi import FastAPI

app = FastAPI(title="Lexi API", version="alpha")


@app.get("/lex/health")
def health():
    return {"status": "ok", "service": "lexi-api", "mode": "alpha"}
