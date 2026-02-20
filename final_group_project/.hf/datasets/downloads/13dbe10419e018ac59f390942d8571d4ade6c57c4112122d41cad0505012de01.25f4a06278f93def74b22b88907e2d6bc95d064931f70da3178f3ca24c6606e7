---
language:
- ko
task_categories:
- conversational
dataset_info:
  features:
  - name: persona
    struct:
    - name: 나이
      dtype: string
    - name: 비밀
      dtype: string
    - name: 성격
      dtype: string
    - name: 외모
      dtype: string
    - name: 이름
      dtype: string
    - name: 이상
      dtype: string
    - name: 직업
      dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 47910381
    num_examples: 21973
  - name: valid
    num_bytes: 2519850
    num_examples: 1160
  download_size: 25171790
  dataset_size: 50430231
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: valid
    path: data/valid-*
---

gemini 에서 생성한 Persona 와 질문, 답변을 담은 데이터셋입니다.