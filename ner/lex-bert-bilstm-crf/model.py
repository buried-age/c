
import torch
import torch.nn as nn
import torch.functional as F
from TorchCRF import CRF
from torch.autograd import Variable
from transformers import BertModel, BertPreTrainedModel


class LEBertSequenceTagging(BertPreTrainedModel):
    def __init__(self, config, slot_num, token_size, is_crf=True):
        super(LEBertSequenceTagging, self).__init__(config)

        # 编码层
        self.bert = BertModel(config)
        self.bert.resize_token_embeddings(new_num_tokens=token_size)

        dropout_prob = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = config.hidden_size

        self.bilstm = nn.LSTM(input_size=self.hidden_size, hidden_size=int(self.hidden_size / 2), batch_first=True,
                              num_layers=1, bidirectional=True)

        # 解码层
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=slot_num)
        self.is_crf = is_crf
        self.crf = CRF(num_labels=slot_num)

        # focal loss
        self.loss = FocalLoss(class_num=slot_num)

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None, tags=None):
        outputs = self.bert(input_ids=input_ids, position_ids=position_ids)
        last_encoder_layer = self.dropout(outputs["last_hidden_state"])
        emissions, (_, _) = self.bilstm(last_encoder_layer)
        emissions = self.linear(emissions)
        mask = input_ids > 0
        loss = 0.0

        if self.is_crf:
            if tags is not None:
                log_likelihood = self.crf(emissions, tags, mask)
                loss = -torch.mean(log_likelihood)
                sequence_of_tags = self.crf.viterbi_decode(emissions, mask)
            else:
                sequence_of_tags = self.crf.viterbi_decode(emissions, mask)
        else:
            if tags is not None:
                loss = self.loss(logits=emissions, target=tags)
                sequence_of_tags = torch.argmax(torch.softmax(emissions, dim=-1), dim=-1)
            else:
                sequence_of_tags = torch.argmax(torch.softmax(emissions, dim=-1), dim=-1)

        return {"loss": loss, "sequence_of_tags": sequence_of_tags}


class FocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.class_num)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
