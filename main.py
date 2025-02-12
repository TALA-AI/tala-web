import os
import sys
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain.llms import WatsonxLLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

app = FastAPI()

# ============== 1) ENV & CREDENTIALS ==============
# load_dotenv()
API_KEY = os.getenv("API_KEY", "")
PROJECT_ID = os.getenv("PROJECT_ID", "")
LLM_MODEL = "ibm/granite-3-8b-instruct"  # or other

params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY.value,
    GenParams.TOP_K: 50,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
}

wml_credentials = {
    "apikey": API_KEY,
    "url": "https://us-south.ml.cloud.ibm.com"
}

# ============== 2) INIT MODEL & RETRIEVAL (run once at startup) ==============
@app.on_event("startup")
def load_model_and_db():
    global qa_chain

    # 2a) LLM init
    llm = WatsonxLLM(
        model_id=LLM_MODEL,
        url=wml_credentials["url"],
        apikey=wml_credentials["apikey"],
        project_id=PROJECT_ID,
        params=params
    )

    # 2b) Load or create vectorstore
    persist_directory = "chroma_db_krsbert"
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=None  # We can omit or pass the same embedding instance if needed
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k":5})

    # 2c) Create system prompt
    SYS_PROMPT = """당신은 대한민국 법률 지식을 제공하는 고급 상담 모델입니다.
    당신의 역할은 아래와 같이 작동합니다:

    1. RAG(Retrieval Augmented Generation) 방식으로 사용자의 질문에 답변합니다.
    - '검색 결과(context)'로 제공된 텍스트는 실제 대한민국 법률 정보이거나, 법령 해설 등으로 구성됩니다.
    - 당신은 이 '검색 결과'를 참조하되, 그 외에 임의로 사실관계를 상상하거나 지어내지 않습니다.

    2. 출력 형식:
    - 먼저, 사용자의 질문에 대한 간결하고 정확한 답변을 제공합니다.
    - 만약 확실한 근거(검색 결과나 일반적 법률 상식)가 없다면, '모르겠다' 혹은 '데이터가 부족합니다'라고 답변하세요.
    - 그 후, 참고한 검색 결과(출처 or 일부 내용)나 법령 조항을 간략히 제시해줄 수 있습니다.

    3. 제한사항:
    - 법률 정보는 최신성을 완전히 보장하지 않을 수 있으므로, 특정 날짜나 최신 개정 내용이 요청되면 모호하다고 답하십시오.
    - 민감한 개인정보나 변호사 자격 행사를 하지 않습니다. 단지 법률 해설이나 일반적 상담을 제공하는 역할입니다.
    - 불법, 차별적, 폭력적 사용을 조장하는 답변은 거절합니다.

    4. 요약:
    - 당신은 '정직한 에이전트'로서, 법률 상담 목적으로만 답변하세요.
    - 과도한 추측이나 과장 없이, '검색결과(context)' 기반으로 답변하되, 모호하거나 근거 없으면 '모름'을 표기합니다.
    """
    
    system_template = SystemMessagePromptTemplate.from_template(SYS_PROMPT)
    user_template = HumanMessagePromptTemplate.from_template("질문:\n{question}\n\n[검색결과]\n{context}\n---\n답변:")

    chat_prompt = ChatPromptTemplate(
        input_variables=["question","context"],
        messages=[system_template, user_template]
    )

    # 2d) Build RetrievalQA chain
    global chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": chat_prompt
        }
    )


# ============== 3) Pydantic model for user input ==============
class QuestionRequest(BaseModel):
    question: str

# ============== 4) /ask Endpoint ==============
@app.post("/ask")
def ask_question(req: QuestionRequest):
    """
    - Expects JSON: {"question":"횡단보도 사고...?"}
    - Returns {"answer":"...", "source_docs":[...]}
    """
    global qa_chain
    user_q = req.question

    # run chain
    result = qa_chain({"query": user_q})
    answer = result["result"]
    sources = []
    for doc in result["source_documents"]:
        # let's just store doc.metadata["case_id"] or doc.page_content[:50]
        sources.append({"case_id": doc.metadata.get("case_id","?"),
                        "chunk": doc.page_content[:60]})

    # return JSON
    return {
        "answer": answer,
        "sources": sources
    }


# ============== 5) If run directly, launch server ==============
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)