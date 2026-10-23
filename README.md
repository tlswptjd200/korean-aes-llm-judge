# 한국어 논술형 글쓰기 자동 채점(AES): 인코더 회귀 vs LLM-as-a-Judge, 그리고 프롬프트 주입

자연어처리 수업 프로젝트 1. 데이터: AI Hub 「논술형 글쓰기 평가 데이터」(초5~중3, 89문항, 18,010편).

## 실행 순서

1. AI Hub에서 데이터를 받아 `data/26.논술형 글쓰기 평가 데이터/...` 그대로 둔다 (라벨링 zip만 사용).
2. `pip install -r requirements.txt`
3. Gemini API 키를 `.env`에 `GEMINI_API_KEY=...` 로 저장 (`.gitignore`에 포함됨).
4. 실행 (`aes/` 폴더에서)

```
python load_data.py      # zip(CP949 JSON) → out/essays.jsonl
python baselines.py      # 상수 / 표면특징 / TF-IDF+Ridge, in-prompt & cross-prompt
python embed_ridge.py    # ko-sroberta 임베딩 + Ridge (CPU)
python llm_judge.py      # Gemini 채점 5조건 (zs, fs, 주입, 방어)
python analyze.py        # 표 out/*.csv, 그림 figs/*.png
```

모든 무작위 분할·샘플링은 `SEED=0`으로 고정.
