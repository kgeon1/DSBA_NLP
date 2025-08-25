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
  *로깅 step마다가 아니라 모델 업데이트마다 할 것

- validation loss
  <img width="1437" height="383" alt="Image" src="https://github.com/user-attachments/assets/1996e59e-1a5e-4d49-8ad4-22e050107fd1" />
  
- validation accuracy
  <img width="1437" height="386" alt="Image" src="https://github.com/user-attachments/assets/8e82c32d-4058-46a4-b5d2-1745113e31be" />

- test loss & accuracy
  <img width="892" height="142" alt="Image" src="https://github.com/user-attachments/assets/16c5f197-4b33-4497-8d0b-2292526b7a89" />

## IV. Discussion
- avg_train_loss
  - batch size가 커질 수록 대체로 빠르게 감소하는 경향을 보였으나 1024에서는 loss 감소 속도가 상대적으로 느렸음
  - batch size가 지나치게 크면 모델 업데이트 빈도를 줄여 학습 효율을 저해했다고 판단됨
- avg_valid_loss
  - 전체적으로 증가하는 경향을 보였으며 이는 overfitting을 시사함
  - 1024에서는 상대적으로 낮게 유지되었으며 overfitting이 덜 발생함
- avg_valid_accuracy
  - 전체적으로 batch size가 커짐에 따라 증가하는 경향을 보임
  - 16에서는 빠르게 증가했다가 이후 overfitting으로 감소했지만 1024에서는 천천히 증가하여 최종적으로는 가장 높았음
- test_loss
  - 256, 64 순으로 높았고 16, 1024는 거의 비슷했음
- test accuracy
  - 거의 비슷했지만 64가 미세하게 가장 높았고 이후 256, 1024, 16 순이었음 
## V. Conclusion
- 세 모델의 test accuracy는 거의 비슷하므로 valid loss, valid accuracy, test loss에서 좋은 지표를 보인 batch size 1024 모델이 가장 좋다고 판단함

## Unfair
- GA를 사용하면 epoch를 고정했을 때 모델이 업데이트되는 횟수가 달라짐
- epoch를 바꿔서 step이 갖도록 해보자!
