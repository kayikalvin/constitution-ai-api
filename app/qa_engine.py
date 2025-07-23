import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

class ConstitutionQASystem:
    def __init__(self, pdf_paths=None, model_name="google/flan-t5-base"):
        if pdf_paths is None:
            self.pdf_paths = ["app/TheConstitutionOfKenya.pdf"]
        else:
            self.pdf_paths = pdf_paths if isinstance(pdf_paths, list) else [pdf_paths]

        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None
        self.llm = None

    def load_and_process_pdfs(self):
        all_documents = []
        for pdf_path in self.pdf_paths:
            if os.path.exists(pdf_path):
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                all_documents.extend(documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        return text_splitter.split_documents(all_documents)

    def create_embeddings_and_vectorstore(self, texts):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = FAISS.from_documents(texts, embeddings)

    def setup_llm(self):
        try:
            pipe = pipeline(
                "text2text-generation",
                model=self.model_name,
                max_length=300,
                temperature=0.6,
                do_sample=False,
                device=-1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print("⚠️ Primary model failed, trying fallback...")
            self.setup_fallback_llm()

    def setup_fallback_llm(self):
        try:
            pipe = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=200,
                temperature=0.6,
                do_sample=False,
                device=-1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("✅ Fallback model loaded (flan-t5-small)")
        except Exception as e:
            raise RuntimeError("Both primary and fallback models failed.") from e

    def create_qa_chain(self):
        prompt_template = """Answer the following question based on the given context.

Context: {context}

Question: {question}

Answer:"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def ask_question(self, question: str):
        if not self.qa_chain:
            raise ValueError("System is not set up yet.")
        result = self.qa_chain({"query": question})
        answer = result["result"].strip()
        return {
            "answer": answer,
            "source_documents": result["source_documents"]
        }

    def save_vectorstore(self, path="app/constitution_vectorstore"):
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_vectorstore(self, path="app/constitution_vectorstore"):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vectorstore = FAISS.load_local(
                path, embeddings, allow_dangerous_deserialization=True
            )
            return True
        except:
            return False

    def setup_system(self):
        texts = self.load_and_process_pdfs()
        self.create_embeddings_and_vectorstore(texts)
        self.setup_llm()
        self.create_qa_chain()
