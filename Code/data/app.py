__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

# 🔹 최신 LangChain 패키지 사용
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ 최신 패키지
from langchain_chroma import Chroma  # ✅ 최신 패키지
from langchain_ibm import WatsonxLLM  # ✅ 최신 WatsonX 패키지

# 🔹 환경 변수 로드
load_dotenv()

# 🔹 FastAPI 인스턴스 생성
app = FastAPI()

# 🔹 IBM WatsonX 설정
project_id = os.getenv("PROJECT_ID", None)
wml_credentials = {
    "apikey": os.getenv("API_KEY", None),
    "url": 'https://us-south.ml.cloud.ibm.com'
}

parameters = { 
    "decoding_method": "greedy",
    "min_new_tokens": 1,
    "max_new_tokens": 500,
    "stop_sequences": ["<|endoftext|>"]
}

model_id = 'ibm/granite-3-8b-instruct'
watsonx_llama2_korean = WatsonxLLM(
    model_id=model_id,
    url=wml_credentials.get("url"),
    apikey=wml_credentials.get("apikey"),
    project_id=project_id, 
    params=parameters
)

# 🔹 데이터 로드
df = pd.read_csv("/home/ibmuser01/src/legal/tala-web/Code/data/accident_datas.csv")

# 🔹 RAG 모델 로드 (ChromaDB)
persist_directory = "./chroma_accidents"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ 최신 패키지 적용
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)  # ✅ 최신 패키지 적용

# 🔹 데이터 모델 정의
class AccidentQuery(BaseModel):
    accident_text: str

class AIQuery(BaseModel):
    accident_text: str
    user_question: str

# 🔹 API: 유사한 사고 사례 검색
@app.post("/search_accidents/")
async def search_accidents(query: AccidentQuery):
    query_embedding = embeddings.embed_query(query.accident_text)
    results = vectorstore.similarity_search_by_vector(query_embedding, k=3)

    if not results:
        raise HTTPException(status_code=404, detail="유사한 사고 사례를 찾을 수 없습니다.")

    # ✅ 사고 데이터에서 URL 포함하여 반환
    response_data = []
    for res in results:
        accident_text = res.page_content
        accident_row = df[df["Accident"] == accident_text].iloc[0]
        response_data.append({
            "accident": accident_text,
            "url": accident_row["URL"]  # ✅ URL 포함
        })

    return response_data


# 🔹 API: AI 질의응답 처리
@app.post("/ask_ai/")
async def ask_ai(query: AIQuery):
    # 사고 사례 상세 정보 조회
    accident_data = df[df["Accident"] == query.accident_text].iloc[0]

    # WatsonX 프롬프트 생성
    prompt_text = f"""
    🚗 사고 사례 분석 🚗

    - 사고 사례: {accident_data['Accident']}
    - 기본 과실 설명: {accident_data['Basic Fault']}
    - 관련 판례: {accident_data['Cases']}
    - 관련 법규: {accident_data['Laws']}

    사용자 질문: {query.user_question}
    """

    # WatsonX AI 호출
    response = watsonx_llama2_korean.generate([prompt_text])

    return {"response": response.generations[0][0].text}

# 🔹 FastAPI 실행 (uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8066)
