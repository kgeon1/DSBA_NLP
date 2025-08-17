# IMDB classification: BERT vs RoBERTa
- 본 실험은 IMDB dataset의 sentiment classification task에서 BERT와 RoBERTa의 성능을 비교하는 것을 목적으로 함


## I. BERT vs RoBERTa
  |  모델 	| BERT	| RoBERTa |
  |---------|---------|---------|
  |  **사전 훈련 목적**  |	양방향 문맥 이해 |	BERT 최적화 및 성능 향상 |
  | **마스킹** |	정적 마스킹 (토큰 마스킹 위치 고정, 여러 Epoch동안 동일한 마스킹 패턴) |	동적 마스킹 (문장마다 새로운 마스킹 패턴) |
  | **NSP** |	포함 |	제거 |
  | **훈련 데이터** |	16GB (Wikipedia, BookCorpus) |	160GB (CC-News, OpenWebText, Stories 등 추가) |
  | **배치 크기**	| 작음 (ex 256)	| 큼 (ex 8000) |

## II. Setting
- Dataset: IMDB 50k
  - Num_labels: 2 (긍정/부정)
  - train : validation : test = 8 : 1 : 1
- Epoch: 5
- Learning Rate: 5e-5
- Scheduler: constant (w/o warmup)
- Optimizer: AdamW
- Max len: 128

## III. Model
- bert-base-uncased (110M)
- roberta-base (125M)

## IV. Result
- train loss
  <img width="1432" height="382" alt="Image" src="https://github.com/user-attachments/assets/2dbaae7e-f68f-4534-83d2-a12e252c4603" />
  <img width="1428" height="386" alt="Image" src="https://github.com/user-attachments/assets/90b41fa2-a1c1-432b-b750-2cd556754006" />

- validation loss
  <img width="1429" height="380" alt="Image" src="https://github.com/user-attachments/assets/ffd9ae4c-ff15-4361-8cab-92fdc4bff105" />
  
- validation accuracy
  <img width="1432" height="383" alt="Image" src="https://github.com/user-attachments/assets/503af68a-b0dc-4946-befa-fe0bb7be581d" />

- test loss
  <img width="1430" height="381" alt="Image" src="https://github.com/user-attachments/assets/6f936b70-55ec-4b14-a365-16f2a20d5fc5" />

- test accuracy
  <img width="1431" height="384" alt="Image" src="https://github.com/user-attachments/assets/b0ffd6cf-844c-40b5-bc33-eddd7dda2c37" />
