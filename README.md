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

  ### IV-1. Additional Experiment
  - RoBERTa 모델에서 epoch 3 부근부터 학습이 잘 되지 않는 것처럼 보이는 현상이 나타남.
    (별도의 lr 조절이나 early stopping x)
  - roberta-base에 대해서만 linear scheduler를 사용하여 추가 실험을 진행함.
    - scheduler 외에 다른 조건은 동일

  - train loss
    <img width="1435" height="380" alt="Image" src="https://github.com/user-attachments/assets/baf16377-243c-4b56-b6ed-d7e00129d5e4" />
  - validation loss
    <img width="1434" height="382" alt="Image" src="https://github.com/user-attachments/assets/112eec38-27bd-4e65-9939-9a2cded8d548" />
  - validation accuracy
  - <img width="1432" height="385" alt="Image" src="https://github.com/user-attachments/assets/1ff6b035-9ddc-457b-8f03-42c0c4d69145" />
  - test loss
    <img width="1430" height="381" alt="Image" src="https://github.com/user-attachments/assets/e7ff1b3b-4e64-48aa-896f-f78eb6bdaea9" />
  - test accuracy
    <img width="1429" height="386" alt="Image" src="https://github.com/user-attachments/assets/6e2a6941-2dae-45d1-9b39-1af3f8893415" />

  | model | BERT | RoBERTa | RoBERTa w/ linear scheduler |
  |---|------|---------|-----------------------------|
  | **test loss** | 0.316 | 0.255 | 0.507 |
  | **test accuracy** | 0.885 | 0.892 | 0.905 |

## V. Discussion
