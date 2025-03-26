import os
import faiss
import numpy as np
import pickle
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 환경 변수 로드
load_dotenv()

# API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)  # CORS 설정

# 메인 페이지 라우트
@app.route('/')
def home():
    return render_template('index.html')

# FAISS 인덱스 로드
def load_faiss_index():
    try:
        index = faiss.read_index("faiss_index/index.faiss")
        with open("faiss_index/index.pkl", "rb") as f:
            embeddings = pickle.load(f)
        return index, embeddings
    except Exception as e:
        print(f"FAISS 인덱스 로드 오류: {str(e)}")
        raise

# 질문을 벡터로 변환하는 함수
def query_to_vector(query):
    try:
        embeddings = OpenAIEmbeddings()
        vector = embeddings.embed_query(query)
        return vector
    except Exception as e:
        print(f"임베딩 생성 오류: {str(e)}")
        raise

# 챗봇 API 라우트
@app.route('/query', methods=['POST'])
def rag_query():
    data = request.json
    query = data.get('query')
    
    try:
        print(f"받은 질문: {query}")
        
        # 간단한 응답으로 테스트
        if query == "테스트":
            return jsonify({
                "query": query,
                "response": "테스트 응답입니다. 시스템이 작동 중입니다."
            })
        
        # FAISS 인덱스 로드
        index, embeddings = load_faiss_index()
        print("FAISS 인덱스 로드 성공")
        
        # 질문을 벡터로 변환
        query_vector = query_to_vector(query)
        print("임베딩 생성 성공")
        
        # FAISS를 사용하여 가장 가까운 문서 검색
        D, I = index.search(np.array([query_vector]).astype('float32'), k=3)
        print(f"검색된 문서 인덱스: {I[0]}")
        
        # 검색된 문서의 내용 가져오기
        contexts = []
        sources = []
        
        for idx, i in enumerate(I[0]):
            if i < len(embeddings) and i >= 0:
                doc_content = embeddings[i]['content']
                contexts.append(doc_content)
                
                # 인덱스 번호를 출처로 사용
                sources.append(f"검색 결과 #{idx+1} (인덱스: {i})")
        
        context = " ".join(contexts)
        source_info = "\n".join(sources)
        print(f"검색된 문맥 길이: {len(context)} 문자")
        print(f"문서 출처: {source_info}")
        
        # LLM에 질문과 문맥 전달
        response = generate_response(query, context)
        
        # 출처 정보 추가
        final_response = f"{response}\n\n참고 문서:\n{source_info}"
        print("응답 생성 성공")
        
        return jsonify({
            "query": query,
            "response": final_response,
            "sources": sources
        })
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"오류 발생: {str(e)}")
        print(error_trace)
        return jsonify({
            "query": query,
            "response": f"오류가 발생했습니다: {str(e)}. 서버 로그를 확인해 주세요."
        })

def generate_response(query, context):
    try:
        # LLM 모델 초기화
        llm = ChatOpenAI(model_name="gpt-4")
        
        # 프롬프트 템플릿 생성
        prompt_template = PromptTemplate.from_template(
            """당신은 질문에 답변하는 도우미입니다.
            다음 정보를 바탕으로 질문에 답변해주세요.
            답변 끝에 출처 정보를 추가하지 마세요. 시스템이 자동으로 추가합니다.
            
            문맥:
            {context}
            
            질문:
            {query}
            
            답변:"""
        )
        
        # 프롬프트 생성
        prompt = prompt_template.format(context=context, query=query)
        
        # LLM 호출하여 응답 생성
        response = llm.invoke(prompt)
        
        return response.content
    except Exception as e:
        print(f"응답 생성 오류: {str(e)}")
        raise

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)