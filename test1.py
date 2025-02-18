import os
import re
import shutil
import logging
import pandas as pd
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Union
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API Key
load_dotenv()
GENAI_API_KEY = "AIzaSyDqgBIeKpq9npG-En5zNk23AU-rY2kKt5s"
if not GENAI_API_KEY:
    raise ValueError("⚠️ ไม่พบ GOOGLE_API_KEY กรุณาตั้งค่า API Key ก่อน!")

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-pro")
embedding_model = SentenceTransformer("sentence-transformers/LaBSE")

class NCTEmbeddingFunction:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        return self.model.encode([input] if isinstance(input, str) else input, convert_to_tensor=False).tolist()

def reset_chroma_db(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    logger.info(f"✨ Reset ChromaDB at {path}")

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df['id'] = df.apply(lambda row: f"{row['members']}_{hash(row['description'])}", axis=1)
    df['search_text'] = df.apply(lambda row: f"{row['members']} {row['description']}", axis=1)
    return df

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip()) if isinstance(text, str) else ""

def load_data_to_chroma(df: pd.DataFrame, collection):
    for _, row in prepare_data(df).iterrows():
        collection.add(ids=[row['id']], documents=[clean_text(row['search_text'])], metadatas=[{"member": row['members'], "description": row['description']}])
    logger.info("✨ ข้อมูลถูกบันทึกใน ChromaDB แล้ว!")

def extract_member_name(query: str, members: List[str]) -> str:
    clean_q = clean_text(query)
    for member in members:
        if clean_text(member) in clean_q:
            return member
    return next((m for m in members if any(w in clean_q for w in clean_text(m).split() if len(w) > 2)), None)

def generate_with_gemini(query: str, member_name: str = None) -> str:
    prompt = f"คำถามเกี่ยวกับ{' ' + member_name if member_name else ''}: \"{query}\"\nโปรดตอบตามความรู้ทั่วไปหรือแจ้งว่าไม่มีข้อมูล"
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if hasattr(response, 'text') else "⚠️ ไม่สามารถสร้างคำตอบได้"
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}")
        return "⚠️ เกิดข้อผิดพลาดในการสร้างคำตอบ"

def format_answer(metadatas: List[dict], member_name: str, query: str) -> str:
    context = ' '.join(meta['description'] for meta in metadatas if clean_text(meta['member']) == clean_text(member_name))
    return generate_with_gemini(query, member_name) if not context else f"{member_name}: {context}"

def search_similar_members(query: str, members: List[str]) -> List[str]:
    clean_q = clean_text(query)
    return [m for m in members if any(w in clean_q for w in clean_text(m).split() if len(w) > 2)]

def rag_search(query: str, df: pd.DataFrame, collection):
    members = df['members'].unique().tolist()
    member_name = extract_member_name(query, members)
    
    if not member_name:
        similar_members = search_similar_members(query, members)
        suggestion = f"⚠️ ไม่พบชื่อสมาชิกที่ชัดเจน คุณหมายถึง: {', '.join(similar_members[:3])} หรือไม่?"
        print(f"🤖 คำตอบ: {suggestion}" if similar_members else f"🔍 คำถาม: {query}\n🤖 คำตอบ: {generate_with_gemini(query)}")
        return
    
    results = collection.query(query_texts=[f"{member_name} {clean_text(query)}"], n_results=5)
    answer = format_answer(results.get("metadatas", [{}])[0], member_name, query) if results.get("ids") else generate_with_gemini(query, member_name)
    print(f"🔍 คำถาม: {query}\n🤖 คำตอบ: {answer}")

def interactive_qa(df: pd.DataFrame, collection):
    print("🎤 ระบบตอบคำถามเกี่ยวกับสมาชิก NCT (พิมพ์ 'exit' เพื่อออก)")
    while (query := input("\n❓ กรุณาป้อนคำถาม: ").strip().lower()) != 'exit':
        rag_search(query, df, collection) if query else print("⚠️ กรุณาป้อนคำถาม")
    print("👋 ขอบคุณที่ใช้บริการ!")

if __name__ == "__main__":
    try:
        df = pd.read_csv("NEO - Sheet1.csv")
        reset_chroma_db("./chroma_nct")
        client = chromadb.PersistentClient(path="./chroma_nct")
        collection = client.create_collection(name="nct_members", embedding_function=NCTEmbeddingFunction(embedding_model))
        load_data_to_chroma(df, collection)
        interactive_qa(df, collection)
    except FileNotFoundError:
        logger.error("❌ ไม่พบไฟล์ข้อมูล CSV")
        print("❌ ไม่พบไฟล์ข้อมูล CSV")
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
