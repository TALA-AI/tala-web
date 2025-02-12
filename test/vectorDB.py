import os
import json
import uuid

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Dict

######################
# 0) 설정
######################
openai.api_key = "sk-..."  # OpenAI API 키 (혹은 환경변수 등)

# Qdrant (로컬 or 클라우드)
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "korean_laws"
EMBED_DIM = 1536  # text-embedding-ada-002
CHUNK_SIZE = 300  # 단어 기준 예시
LAW_DIR = "laws"  # JSON 파일이 들어 있는 폴더

######################
# 1) 임베딩 함수 (OpenAI 예시)
######################
def get_embedding(text: str) -> List[float]:
    """
    text-embedding-ada-002 사용.
    """
    resp = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    vector = resp["data"][0]["embedding"]
    return vector

######################
# 2) chunking
######################
def chunk_text(text: str, chunk_size=CHUNK_SIZE) -> List[str]:
    """
    간단히 '단어' 기준으로 chunk_size만큼씩 분할.
    실제로는 토큰 수 기반, 문단 단위, 항 단위 등으로 조정 권장.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        sub_text = " ".join(words[start:end])
        chunks.append(sub_text)
        start = end
    return chunks

######################
# 3) Qdrant 컬렉션 준비
######################
def setup_collection(collection_name: str, vector_size: int = EMBED_DIM):
    try:
        info = client.get_collection(collection_name)
        print(f"[INFO] Collection '{collection_name}' already exists:", info)
    except:
        print(f"[INFO] Creating collection '{collection_name}'...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

######################
# 4) JSON 파일 -> 텍스트 추출
######################
def parse_law_json(json_data: Dict) -> List[Dict]:
    """
    사용자 예시 JSON 구조에 맞춰 '법령' 내부 필드들을 읽어서
    텍스트 덩어리를 모은다. (개정문내용, 부칙내용 등)
    
    반환 형식: [{'title':..., 'text':...}, ...] 이런 식.
    """
    results = []

    # 최상위 "법령" 키
    law = json_data.get("법령", {})
    law_name = law.get("기본정보", {}).get("법령명_한글", "미상_법령")

    # 1) 개정문 -> 개정문내용 (2차원 배열)
    개정문 = law.get("개정문", {})
    개정문내용 = 개정문.get("개정문내용", [])
    # 예: [["⊙법률 제19841호(2023.12.26)", ... ], ...] 이런 식 2차원
    for paragraph_array in 개정문내용:
        # paragraph_array는 문자열 리스트
        text_joined = "\n".join(paragraph_array)  # 줄바꿈으로 연결
        if text_joined.strip():
            results.append({
                "title": f"{law_name} 개정문",
                "text": text_joined
            })

    # 2) 부칙 -> 부칙단위[].부칙내용[]
    부칙 = law.get("부칙", {})
    부칙단위 = 부칙.get("부칙단위", [])
    for 부칙_item in 부칙단위:
        부칙공포번호 = 부칙_item.get("부칙공포번호", "")
        부칙내용_list = 부칙_item.get("부칙내용", [])
        # 부칙내용_list: 2차원 (리스트 안에 [ ["부칙 ..", "제1조..." ], [...] ])
        for paragraph_array in 부칙내용_list:
            text_joined = "\n".join(paragraph_array)
            if text_joined.strip():
                results.append({
                    "title": f"{law_name}_부칙_{부칙공포번호}",
                    "text": text_joined
                })

    # 필요 시, 법령키, 시행일자, etc.를 메타정보로 추가
    return results


######################
# 5) 실제 업서트
######################
def insert_into_vector_db(law_title: str, content_text: str):
    """
    content_text를 chunk로 쪼개 임베딩 후 Qdrant upsert.
    """
    chunks = chunk_text(content_text, chunk_size=CHUNK_SIZE)
    for idx, chunk_str in enumerate(chunks):
        if not chunk_str.strip():
            continue
        # 임베딩
        embedding = get_embedding(chunk_str)

        # ID
        point_id = str(uuid.uuid4())

        payload = {
            "law_title": law_title,
            "chunk_index": idx,
            "text": chunk_str,
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

        # upsert
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

######################
# 6) 메인
######################
def main():
    setup_collection(COLLECTION_NAME, EMBED_DIM)

    # laws 디렉토리에 있는 *.json 파일들 순회
    for fname in os.listdir(LAW_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(LAW_DIR, fname)

        print(f"[INFO] Processing {fpath}")
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 파싱
        law_segments = parse_law_json(data)
        # law_segments => [{title:..., text:...}, ...]

        for seg in law_segments:
            title = seg["title"]
            body_text = seg["text"]
            insert_into_vector_db(title, body_text)

    print("[DONE] All JSON processed.")


if __name__ == "__main__":
    main()