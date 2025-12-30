# 🧪 Scientific Knowledge Question Answering System  
## 한국어 과학 상식 질의응답 시스템 (RAG Optimization)

본 프로젝트는 **"과학 지식 질의 응답 시스템 구축" 대회**를 위해 개발된 **고성능 RAG(Retrieval-Augmented Generation) 파이프라인**입니다.  
초기 베이스라인(**MAP 0.66**)에서 시작하여, **다단계 검색 전략**, **지능형 게이팅**, **합성 데이터 파인튜닝**, 그리고 **정밀한 LLM Judge 전략**을 통해 최종 Private Board 기준 **MAP 0.9136**을 달성했습니다.

---

## 📋 Executive Summary

- **목표**: 한국어 과학 상식 문서(약 **4,200개**) 기반의 정확한 정보 검색 및 답변 생성  
- **최종 성과**: **MAP 0.9136	 / MRR 0.9197	 (최종 Private Board 1위 달성)**  
- **핵심 기술**
  - **Retrieval**: BGE-M3 (Dense + Sparse) Hybrid Search + RRF Fusion
  - **Optimization**: Solar Pro 2 기반 지능형 게이팅 & Query Expansion
  - **Reranking**: BGE-Reranker-v2-m3 Cross-Encoder
  - **Final Strategy**: Synthetic Data Fine-tuning & LLM Judge 기반 Surgical Strike

---

## 📈 Performance Evolution

프로젝트 기간 동안 총 **180회의 실험**을 통해 단계적으로 성능을 향상시켰습니다.

| Phase | 주요 전략 (Milestone) | MAP | MRR | 변화율 | 상태 |
|------:|------------------------|-----:|-----:|------:|------|
| Baseline | BM25 + SBERT (Basic Hybrid) | 0.6629 | 0.6712 | - | 🏁 시작 |
| Phase 1 | Cross-Encoder Reranker 도입 | 0.7742 | 0.7773 | +16.8% | 🚀 급상승 |
| Phase 2 | HyDE (쿼리 확장) + 파라미터 최적화 | 0.8470 | 0.8500 | +9.4% | ⭐ 안정화 |
| Phase 3 | Solar Pro 2 지능형 게이팅 (v2) | 0.8826 | 0.8848 | +4.2% | 📈 개선 |
| Phase 4 | BGE-M3 Hybrid + Multi-Query | 0.9409 | 0.9424 | +6.6% | 🏆 SOTA |
| Final | Surgical Strike (LLM Judge 앙상블) | **0.9470** | **0.9470** | +0.6% | 🥇 Final |

---

## 🏗️ System Architecture

최종 파이프라인(**v9_sota 기반**)은 다음과 같은 4단계 프로세스로 구성됩니다.

```
graph TD
    User[User Query] --> Gating[🕵️ Solar Pro 2 Gating]

    Gating -- "Chitchat (False)" --> Skip[🚫 Return Empty List]
    Gating -- "Search Needed (True)" --> Expansion[✨ Multi-Query + HyDE]

    Expansion --> Retrieval_Dense[🔍 BGE-M3 Dense Search]
    Expansion --> Retrieval_Sparse[📖 BGE-M3 Sparse Search]

    Retrieval_Dense --> Fusion[🔄 RRF Fusion (Top 50)]
    Retrieval_Sparse --> Fusion

    Fusion --> Reranker[⚖️ BGE-Reranker-v2-m3]
    Reranker --> Top5[🏆 Final Top 5 Documents]

    Top5 --> Surgical[🩺 Surgical Strike (LLM Judge)]
    Surgical --> FinalAnswer[📝 Answer Generation]
```

---

## 1. Intelligent Gating & Query Analysis

- **Model**: Solar Pro 2 (Upstage)  
- **Role**: 사용자의 질문 의도를 분석하여  
  - **"검색이 필요한 질문(과학/지식)"** vs  
  - **"일상 대화(인사/잡담)"**  
  를 구분합니다.
- **Effect**
  - 불필요한 검색을 차단하여 감점을 방지(**Empty List 반환**)
  - 검색이 필요한 경우 **Multi-Query(독립 질문, 확장 질문)** 생성

---

## 2. Hybrid Retrieval (BGE-M3)

- **Dense Retrieval**: 의미론적 유사성 검색 (**1024-dim**)  
- **Sparse Retrieval**: 키워드 매칭 (**Lexical Weights**)  
- **Fusion**: **RRF(Reciprocal Rank Fusion, K=60)** 를 사용하여 두 검색 결과의 순위를 조화롭게 결합

---

## 3. Cross-Encoder Reranking

- **Model**: `BAAI/bge-reranker-v2-m3`  
- **Process**
  - 1차 검색된 후보군(**Top 50**)에 대해
  - 원본 쿼리와의 정밀한 관련성 점수를 계산하여 재정렬
- **Insight**
  - 쿼리 변형(HyDE)이 아닌 **원본 쿼리**를 사용할 때 가장 높은 정확도를 보였습니다.

---

## 4. Surgical Strike (Final Optimization)

- **Strategy**
  - Fine-tuned Model(**v3**)과 SOTA Model(**v9**)의 결과가 다를 경우,
  - **Gemini 2.5 Flash**를 '판사(Judge)'로 활용하여 **1:1 비교**를 수행합니다.
- **Impact**
  - 전체 쿼리의 약 **5%**에 해당하는 모호한 케이스를 '수술하듯' 정밀하게 교체하여 최종 **0.9470**을 달성했습니다.

---

## 🔬 Key Strategies & Experiments

### 🧪 1. BGE-M3 최적화 및 감점 방지

기존 SBERT+BM25 조합을 `BAAI/bge-m3` 단일 모델의 **Dense+Sparse 하이브리드 체제**로 전환했습니다.

- **EMPTY_IDS 필터링**  
  과학 상식과 무관한 20개의 질문(일상 대화)을 식별하여 검색 결과 없이 반환하는 전략으로 **+7.02%**의 성능 향상을 이뤘습니다.

- **Multi-Query + HyDE**  
  단일 쿼리의 한계를 극복하기 위해 질문을 다각도로 변형하고 가설 답변을 생성하여 Recall을 극대화했습니다.

---

### 🧪 2. 게이팅(Gating) 전략의 진화

단순히 "과학 vs 비과학"으로 분류하던 초기 시도(Phase 6B)는 실패했습니다.  
이를 **"검색이 필요한 정보성 질문 vs 순수 잡담"**으로 재정의하고 Solar Pro 2를 활용한 정밀 분석기를 도입하여 MAP를 **0.80 → 0.88**까지 끌어올렸습니다.

---

### 🧪 3. 합성 데이터 기반 파인튜닝 (Synthetic Fine-tuning)

도메인 적합성을 높이기 위해 Solar Pro 2를 활용하여 데이터 증강을 수행했습니다.

- **Pipeline**: QA 생성(**12,816쌍**) → Hard Negative Mining (Hybrid) → BGE-M3 Fine-tuning  
- **Result**
  - 파인튜닝 모델 단독 사용 시 **과적합 이슈**가 있었으나,
  - 앙상블의 **보조 모델**로 활용하여 최종 성능 개선에 기여했습니다.

---

## 📂 Repository Structure

```text
.
├── 📁 data/                 # 원본 코퍼스 및 생성된 합성 데이터
├── 📁 models/               # LLM Client (Solar, Gemini) 및 임베딩 모델
├── 📁 retrieval/            # 검색 엔진 (Hybrid Search, Reranker 로직)
├── 📁 finetune/             # 파인튜닝 스크립트 (QA 생성, Negative Mining, 학습)
├── 📁 experiments/          # 실험 로그 및 리더보드 제출 파일
├── 📄 main.py               # 메인 실행 파일
├── 📄 eval_rag.py           # RAG 파이프라인 평가 스크립트
└── 📄 requirements.txt      # 의존성 패키지
```

---

## 🚀 Usage

### 1) 환경 설정

```bash
pip install -r requirements.txt
```

### 2) API 키 설정 (`.env`)

```env
UPSTAGE_API_KEY=sk-***
GOOGLE_API_KEY=AIza***
```

### 3) 실행 (Inference)

```bash
python main.py --config config/prod_sota_v9.yaml
```

---
## 📝 최종 리더보드(IR 3조 1위)

<img width="983" height="564" alt="image" src="https://github.com/user-attachments/assets/6fb5c38f-1fea-4648-b6fa-b375761846c2" />

## 📝 Retrospective

이번 프로젝트를 통해 **"데이터의 품질(Gating/Cleaning)"**과 **"모델의 앙상블(Surgical Strike)"**이 단순한 모델 스케일업보다 중요하다는 것을 확인했습니다.  
특히, 이미 높은 성능(**0.94**) 구간에서 추가 상승을 위해 LLM을 심판(Judge)으로 활용한 전략은 실무적인 RAG 시스템 고도화에도 유효한 접근법이 될 것입니다.

---

## 👥 Authors / Date

- **Authors**: IR 3조 Trinity Team(팀장 문서연, 팀원 박준영, 팀원 김재록, 팀원 이재윤,팀원 임환석)
- **Date**: 2025.12.29
