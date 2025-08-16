import wandb 
from tqdm import tqdm
import os

import torch
import torch.nn
import omegaconf
from omegaconf import OmegaConf
from transformers import set_seed
from transformers import get_constant_schedule
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW 

from utils import load_config # , set_logger
from model import EncoderForClassification
from data import IMDBDataset, get_dataloader

# torch.cuda.set_per_process_memory_fraction(11/24) -> 김재희 로컬과 신입생 로컬의 vram 맞추기 용도. 과제 수행 시 삭제하셔도 됩니다. 
# model과 data에서 정의된 custom class 및 function을 import합니다.
"""
여기서 import 하시면 됩니다. 
"""

# ======== 시드 설정 부분 ========
set_seed(42)
# ==============================

def train_iter(model, inputs, optimizer, device):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    # 모델이 (logits, loss) 튜플을 반환하므로, 각각의 변수로 받습니다.
    logits, loss = model(**inputs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    wandb.log({'train_loss' : loss.item()})
    return loss

def valid_iter(model, inputs, device):
    model.eval()
    with torch.no_grad():
        inputs = {key : value.to(device) for key, value in inputs.items()}
        # forward args 만들기
        forward_args = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"]
        }
        if "token_type_ids" in inputs:
            forward_args["token_type_ids"] = inputs["token_type_ids"]

        outputs_logits, outputs_loss = model(**forward_args)
        accuracy = calculate_accuracy(outputs_logits, inputs['labels'])
    return outputs_loss.item(), accuracy

def calculate_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def main(configs : omegaconf.DictConfig) :

    if configs.log_config.use_wandb:
        wandb.init(
            project="DSBA_NLP-classification-exp",    # 👈 wandb 프로젝트 이름을 지정하세요.
            # name=f"exp-{wandb.util.generate_id()}", # 👈 실험의 고유한 이름을 지정할 수 있습니다.
            name=f"NLP-exp-roberta-base_with-linear-scheduler", # 👈 실험의 고유한 이름을 지정할 수 있습니다.
            config=OmegaConf.to_container(configs, resolve=True) # 설정값을 wandb에 저장gi
        )
    # =======================================================

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # torch.cuda.set_per_process_memory_fraction(11/24) # 필요시 사용 (주석 해제)

    # Load model
    print("Loading model...")
    model = EncoderForClassification(model_config=configs.model_config).to(device)
    print("Model loaded.")

    # Load data
    print("Loading data...")
    train_dataloader = get_dataloader(data_config=configs.data_config, split='train')
    valid_dataloader = get_dataloader(data_config=configs.data_config, split='valid')
    test_dataloader = get_dataloader(data_config=configs.data_config, split='test') # 테스트 데이터로더도 로드
    print("Data loaded.")

    # Set optimizer
    # AdamW 옵티마이저 사용 (트랜스포머 모델에 권장)
    optimizer = AdamW(model.parameters(), lr=configs.optimizer_config.learning_rate)
    
    # 학습 스케줄러 설정
    # total_steps = len(train_dataloader) * configs.train_config.epochs
    # warmup_steps = int(total_steps * configs.optimizer_config.warmup_ratio)
    # scheduler = get_constant_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps
    # )
    '''
    scheduler = get_constant_schedule(optimizer)
    '''
    # 선형 스케쥴러
    total_steps = len(train_dataloader) * configs.train_config.epochs
    warmup_steps = int(total_steps * 0.1)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # print(f"Optimizer and scheduler set. Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    print(f"Optimizer and scheduler set. Learning rate will be constant after warmup (if any).")

    # Train & validation for each epoch
    print("Starting training...")
    best_valid_accuracy = 0.0 # 최고 검증 정확도를 저장
    for epoch in range(configs.train_config.epochs) :
        # Train Loop
        train_losses = []
        # tqdm으로 학습 진행률 표시
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train")
        for batch in train_pbar:
            loss = train_iter(model, batch, optimizer, device)
            train_losses.append(loss)
            scheduler.step() # 스케줄러 업데이트
            train_pbar.set_postfix({'loss': loss}) # tqdm에 현재 손실 표시
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch+1} Train Avg Loss: {avg_train_loss:.4f}")
        wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch})


        # Validation Loop
        valid_losses = []
        valid_accuracies = []

        valid_pbar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1} Valid")
        for step, batch in enumerate(valid_pbar):
            loss, accuracy = valid_iter(model, batch, device)
            valid_losses.append(loss)
            valid_accuracies.append(accuracy)

            # tqdm 표시
            valid_pbar.set_postfix({'loss': loss, 'accuracy': accuracy})

            # 🔥 매 스텝마다 wandb 로깅
            wandb.log({
                "val_loss_step": loss,
                "val_accuracy_step": accuracy,
                "epoch": epoch,
                "val_step": step
            })

        # Epoch 단위 평균 로깅
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        avg_valid_accuracy = sum(valid_accuracies) / len(valid_accuracies)
        print(f"Epoch {epoch+1} Valid Avg Loss: {avg_valid_loss:.4f}, Avg Accuracy: {avg_valid_accuracy:.4f}")

        wandb.log({
            "avg_valid_loss": avg_valid_loss,
            "avg_valid_accuracy": avg_valid_accuracy,
            "epoch": epoch
        })

        # 최고 정확도 모델 저장 (선택 사항)
        if avg_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = avg_valid_accuracy
            # model.state_dict()는 모델의 학습 가능한 파라미터들을 딕셔너리 형태로 반환합니다.
            torch.save(model.state_dict(), os.path.join(configs.output_dir, "best_model.pt"))
            print(f"Best validation accuracy achieved! Saving model with accuracy: {best_valid_accuracy:.4f}")

    print("Training finished.")
    
    # validation for last epoch
    print("\nStarting final test with the best model...")
    # 최적 모델 로드 (저장했다면)
    if os.path.exists(os.path.join(configs.output_dir, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(configs.output_dir, "best_model.pt")))
        print("Loaded best model for final testing.")
    
    test_losses = []
    test_accuracies = []
    test_pbar = tqdm(test_dataloader, desc="Final Test")
    for batch in test_pbar:
        loss, accuracy = valid_iter(model, batch, device) # valid_iter를 재활용
        test_losses.append(loss)
        test_accuracies.append(accuracy)
        test_pbar.set_postfix({'loss': loss, 'accuracy': accuracy})

    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
    print(f"Final Test Avg Loss: {avg_test_loss:.4f}, Avg Accuracy: {avg_test_accuracy:.4f}")
    if configs.log_config.use_wandb:
        wandb.log({"final_test_loss": avg_test_loss, "final_test_accuracy": avg_test_accuracy})
        wandb.finish() # wandb 세션 종료

    
    
if __name__ == "__main__" :
    configs = load_config(config_path="../configs/config_roberta.yaml")
    os.makedirs(configs.output_dir, exist_ok=True) # output_dir이 없으면 생성
    main(configs)