import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer

class BertForClassification(nn.Module):
    def __init__(self, pretrained_name = 'bert-base-uncased', num_labels = 1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, num_labels),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs[0][:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits
    
    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            return torch.softmax(logits, dim=-1)