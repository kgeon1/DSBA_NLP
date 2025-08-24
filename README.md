# IMDB classification: BERT with Gradient Accumulation
- 본 실험은 IMDB dataset의 sentiment classification task에서 BERT의 batch size에 따른 성능을 비교하는 것을 목적으로 함. 

## I. Setting
- Dataset: IMDB 50k
  - Num_labels: 2 (긍정/부정)
  - train : validation : test = 8 : 1 : 1
- Epoch: 5
- Learning Rate: 5e-5
- Scheduler: constant (w/o warmup)
- Optimizer: AdamW
- Max len: 128
- Batch Size: 16, 64, 256, 1024

## II. Model
- bert-base-uncased (110M)

## III. Result
- train loss
  <img width="1435" height="381" alt="Image" src="https://github.com/user-attachments/assets/7e4cc586-048a-4c03-bd59-3684a0b0e517" />

- validation loss
  <img width="1437" height="383" alt="Image" src="https://github.com/user-attachments/assets/1996e59e-1a5e-4d49-8ad4-22e050107fd1" />
  
- validation accuracy
  <img width="1437" height="386" alt="Image" src="https://github.com/user-attachments/assets/8e82c32d-4058-46a4-b5d2-1745113e31be" />

- test loss & accuracy
  <img width="892" height="142" alt="Image" src="https://github.com/user-attachments/assets/16c5f197-4b33-4497-8d0b-2292526b7a89" />

## IV. Discussion
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

## V. Conclusion
- 세 모델의 accuracy가 크게 차이나지 않으므로 가장 안정적인 BERT가 가장 좋은 모델이라고 판단함. scheduler, batch size, epoch, early stopping 사용 여부 등의 하이퍼파라미터를 조절했을 때는 RoBERTa가 BERT보다 좋은 모델이 될 가능성도 충분하다고 생각됨.
