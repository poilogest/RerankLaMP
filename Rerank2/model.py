import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer

class BertForClassification(nn.Module):
    def __init__(self, pretrained_name = 'bert-base-uncased', num_labels = 1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name, cache_dir="./cache")
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, num_labels),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels = None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:, 0, :]
        logits = self.classifier(cls_embedding)
        if labels is not None:
            loss = self.loss(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    
    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            logits = self(input_ids, attention_mask)["logits"]
            return torch.softmax(logits, dim=-1)
    
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")  # 无参数时默认返回 CPU