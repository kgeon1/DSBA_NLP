from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
import omegaconf

class EncoderForClassification(nn.Module):
    def __init__(self, model_config: omegaconf.DictConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_config.model_name)
        classifier_dropout = model_config.get("classifier_dropout", 0.1)  # dropout 값 (기본 0.1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, model_config.num_labels)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None,  # 기본값 None
        labels: Optional[torch.Tensor] = None           # 기본값 None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inputs:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len), only for BERT (optional)
            labels: (batch_size), optional
        Outputs:
            logits: (batch_size, num_labels)
            loss: (1) or None
        """

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,  # None이면 무시됨
            return_dict=True
        )

        # [CLS] 토큰 벡터 (모델 종류 상관없이 범용적으로 사용 가능)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # 드롭아웃 + 분류기
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        # loss 계산 (labels 있을 때만)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss
