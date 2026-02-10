from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.retriever import get_retriever

app = FastAPI()

# No static folder required

# Templates folder (your HTML file is here)
templates = Jinja2Templates(directory="app/web/templates")

retriever = get_retriever()  # Your RAG retriever

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return {"answer": "Please ask a valid question."}

    # Retrieve answer from your RAG vectorstore
    docs = retriever.get_relevant_documents(question)
    answer = docs[0].page_content if docs else "Sorry, I could not find an answer."
    return {"answer": answer}
