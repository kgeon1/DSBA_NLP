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
- BERT vs RoBERTa
  - 특정 시점 (epoch 3) 이후 RoBERTa의 성능이 더이상 개선되지 않는 것처럼 보임
    - overfitting?
    - 근데 아무리 overfitting이라 하더라도 binary classification에서 accuracy가 0.5 수준이면 그냥 찍는다는 말인데 이게 가능한 현상인지 의문이 생김. (추가 실험에서는 이러한 문제 나타나지 않음)
  - train loss, validation loss, validation accuracy에서는 BERT가 RoBERTa보다 지표가 좋게 나타났지만 test loss, test accuracy에서는 RoBERTa의 성능이 더 좋게 나타남.
    - epoch마다 validation에서 최고 성능을 보인 모델을 저장하여 test를 진행하는 방식으로 실험을 진행하였는데, RoBERTa가 고장나기 이전까지의 최고 성능 기준으로는 RoBERTa의 성능이 더 좋았던 것으로 해석
  - batch size 8000 수준에서 pretraining된 RoBERTa를 BERT와의 조건을 맞추기 위해 batch size 16에서 학습하였는데 이 때문에 문제가 생겼을 수도 있다고 생각됨. 이후 더 큰 batch size에서 추가 실험 필요해보임.

- RoBERTa with linear scheduler
  - linear scheduler를 사용했을 때는 위의 RoBERTa에서의 문제가 나타나지 않았음
  - train loss, valid loss, valid accuracy 모두 세 모델 중 가장 뛰어난 성능을 보였으며 test accuracy 역시 가장 높았음.
  - 하지만, test loss도 함께 크게 증가하였는데 learning rate를 줄이면서 모델이 천천히 수렴하여 스스로의 답에 확신을 갖지 못하는 상태라고 해석됨. 

## VI. Conclusion
- 세 모델의 accuracy가 크게 차이나지 않으므로 가장 안정적인 BERT가 가장 좋은 모델이라고 판단함. scheduler, batch size, epoch, early stopping 사용 여부 등의 하이퍼파라미터를 조절했을 때는 RoBERTa가 BERT보다 좋은 모델이 될 가능성도 충분하다고 생각됨.

## 코드 작성 시 주의사항
- BERT에서는 token_type_ids가 필요하지만 RoBERTa에서는 필요하지 않음
  - 현재 코드에서는 if 문으로 각각 처리해주고 있는데 코드의 직관성이 떨어지는 것 같기도 해서 아예 코드를 분리하는 게 더 좋을 수도 있겠다는 생각이 듦.
- validation step의 logging에서 train step까지 x축에 포함되어 그래프가 끊어지는 형태로 나타나는데 개선이 필요해보임.
