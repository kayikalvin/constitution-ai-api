from fastapi import FastAPI
from pydantic import BaseModel
from qa_engine import ConstitutionQASystem

app = FastAPI()
qa_system = ConstitutionQASystem()

# Load or build vectorstore
if qa_system.load_vectorstore():
    qa_system.setup_llm()
    qa_system.create_qa_chain()
else:
    qa_system.setup_system()
    qa_system.save_vectorstore()

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(q: Question):
    try:
        result = qa_system.ask_question(q.query)
        return {
            "answer": result["answer"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    except Exception as e:
        return {"error": str(e)}
