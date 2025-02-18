import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import re
from typing import List, Union
import shutil
import os
from transformers import pipeline

# ใช้โมเดลที่มี dimension 768
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# โหลดโมเดลภาษา Hugging Face สำหรับการสร้างคำตอบ
generator = pipeline("text-generation", model="gpt2")  # สามารถเปลี่ยนโมเดลเป็นอื่นได้ เช่น "facebook/bart-large"

class NCTEmbeddingFunction:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input)
        return embeddings.tolist()

def reset_chroma_db(path: str):
    """ลบและสร้าง ChromaDB ใหม่"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """รวมข้อมูลที่เกี่ยวข้องกันเข้าด้วยกัน"""
    # สร้าง ID ที่ไม่ซ้ำกัน
    df['id'] = df.apply(lambda row: f"{row['members']}_{hash(row['description'])}", axis=1)
    return df

# รีเซ็ต ChromaDB
reset_chroma_db("./chroma_nct")

# เชื่อมต่อ ChromaDB ใหม่
chroma_client = chromadb.PersistentClient(path="./chroma_nct")
embedding_function = NCTEmbeddingFunction(model)

# สร้าง collection ใหม่
collection = chroma_client.create_collection(
    name="nct_members",
    embedding_function=embedding_function
)

def clean_text(text: str) -> str:
    """ทำความสะอาดและเตรียมข้อความ"""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_data_to_chroma(df: pd.DataFrame) -> None:
    """โหลดข้อมูลเข้า ChromaDB"""
    try:
        # เตรียมข้อมูล
        prepared_df = prepare_data(df)
        
        total_records = len(prepared_df)
        print(f"\n📝 กำลังบันทึกข้อมูลทั้งหมด {total_records} รายการ...")
        
        # บันทึกข้อมูลทีละรายการ
        for idx, row in prepared_df.iterrows():
            collection.add(
                ids=[row['id']],
                documents=[clean_text(row['description'])],
                metadatas=[{
                    "member": row['members'],
                    "description": row['description']
                }]
            )
            
            if (idx + 1) % 10 == 0 or (idx + 1) == total_records:
                print(f"✅ บันทึกข้อมูลแล้ว {idx + 1}/{total_records} รายการ")
        
        print("\n✨ บันทึกข้อมูลทั้งหมดเรียบร้อยแล้ว!")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการบันทึกข้อมูล: {str(e)}")

def format_answer(metadatas: List[dict], query: str) -> str:
    """สร้างคำตอบจากข้อมูลที่ค้นพบ"""
    query_lower = query.lower()
    member_name = None
    
    # ค้นหาชื่อสมาชิกจากคำถาม
    for meta in metadatas:
        if meta['member'].lower() in query_lower:
            member_name = meta['member']
            break
    
    if not member_name:
        return f"ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณเกี่ยวกับ {query}"
    
    # ค้นหาข้อมูลที่เกี่ยวข้องกับสมาชิกที่พบ
    relevant_info = [
        meta['description'] for meta in metadatas
        if meta['member'].lower() == member_name.lower()
    ]

    if relevant_info:
        return f"{member_name.title()}: {' '.join(relevant_info)}"
    
    return f"ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณเกี่ยวกับ {member_name}"

def rag_search(query: str) -> None:
    """ค้นหาและสร้างคำตอบ"""
    try:
        clean_query = clean_text(query)
        
        results = collection.query(
            query_texts=[clean_query],
            n_results=5
        )
        
        if results["ids"] and results["ids"][0]:
            # กรองคำตอบที่ตรงกับสมาชิกที่ถามถึง
            answer = format_answer(results["metadatas"][0], query)
            print(f"\n🔍 คำถาม: {query}")
            print(f"🤖 คำตอบ: {answer}")
        else:
            print(f"\n❌ ไม่พบข้อมูลที่เกี่ยวข้องกับคำถาม: {query}")
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการค้นหา: {str(e)}")

def interactive_qa():
    """ฟังก์ชันโต้ตอบกับผู้ใช้"""
    print("\n🎤 ระบบตอบคำถามเกี่ยวกับสมาชิก NCT")
    print("พิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        query = input("\n❓ กรุณาป้อนคำถาม: ").strip()
        
        if query.lower() == 'exit':
            print("👋 ขอบคุณที่ใช้บริการ!")
            break
        
        if query:
            rag_search(query)
        else:
            print("⚠️ กรุณาป้อนคำถาม")

# การทำงานหลัก
if __name__ == "__main__":
    try:
        print("🚀 เริ่มต้นระบบ...")
        df = pd.read_csv("NEO - Sheet1.csv")  # โหลดข้อมูลจากไฟล์ nct.csv
        print("📊 โหลดข้อมูลสำเร็จ")
        load_data_to_chroma(df)
        interactive_qa()
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการทำงาน: {str(e)}")