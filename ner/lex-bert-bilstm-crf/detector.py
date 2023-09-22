
import json

import torch
from transformers import BertTokenizer
from tqdm import tqdm
from model import LEBertSequenceTagging
from tools.label import LabelProcess, NERLabelling


class BSTDetector(object):
    def __init__(self, model: LEBertSequenceTagging, slots: LabelProcess, tokenizer: BertTokenizer, use_cuda=True):
        self.model = model
        self.tokenizer = tokenizer
        self.slot_label_process = slots
        self.device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def predict(self, item: dict):
        input_ids, position_ids = NERLabelling.encode_lex_bert_inputs(text=item["query"], extend_list=item["extends"],
                                                                      tokenizer=self.tokenizer)
        input_ids, position_ids = [input_ids], [position_ids]
        with torch.no_grad():
            outputs = self.model(input_ids=torch.tensor(input_ids).long().to(self.device),
                                 position_ids=torch.tensor(position_ids).long().to(self.device))

        sequence_of_tags = outputs["sequence_of_tags"]
        seq_labels = self.slot_label_process.decode(labels=sequence_of_tags)
        entity = self.extract_slots_from_labels(input_ids=input_ids[0], slot_labels=seq_labels[0])
        return entity

    def predict_slot_labels(self, tags):
        return self.slot_label_process.decode(tags)

    def extract_slots_from_labels(self, input_ids, slot_labels, delimiter="_"):
        """
        从解码的序列标签中，抽取出相应的槽位信息
        :param input_ids: 文本对应的id
        :param slot_labels: 模型预测的序列标签，BIO序列
        :param mask: 文本中的掩码
        :param delimiter: 实体标签，B与Type之间的分割符
        :return: 实体的结构化数据 [{"type":, "value":}]
        """
        entity_mark = dict()
        entity_pointer = None

        for index, label in enumerate(slot_labels):
            if label.startswith("B"):
                category = label.split(delimiter)[1]
                entity_pointer = (index, category)
                entity_mark.setdefault(entity_pointer, [label])
            elif label.startswith("I"):
                if entity_pointer is None:
                    continue
                if entity_pointer[1] != label.split(delimiter)[1]:
                    continue
                entity_mark[entity_pointer].append(label)
            else:
                entity_pointer = None

        entities = []
        for key, value in entity_mark.items():
            start = key[0]
            e_type = key[1]

            name = ""
            for i in range(start, start + len(value)):
                name += self.tokenizer.decode(input_ids[i])
            entities.append({"start": start, "end": start + len(value), "type": e_type, "value": name})
        return entities

    @classmethod
    def from_pretrained(cls, model_path, tokenizer_path, slot_path, **kwargs):
        slot_dict = LabelProcess.load(slot_path)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        special_tokens_dict = {
            "additional_special_tokens": ["<MCD>", "<AMC>", "<ATP>", "<ECD>", "<AEC>", "<CSE>", "<FAC>", "<SYS>",
                                          "<COM>",
                                          "<ELE>", "<DTD>", "<ELF>", "<RDN>", "<IOE>", "<OMN>", "<MMF>", "<CLE>",
                                          "<MAI>",
                                          "<CAR>", "<IMP>", "<ADJ>", "<INS>", "<ALA>", "<EBN>", "<TER>", "<URG>",
                                          "<EAR>"]}
        tokenizer.add_special_tokens(special_tokens_dict)
        
        model = LEBertSequenceTagging.from_pretrained(model_path, slot_num=55, token_size=len(tokenizer),
                                                      ignore_mismatched_sizes=True)

        return cls(model=model, slots=slot_dict, tokenizer=tokenizer)


if __name__ == '__main__':
    tokenizer_p = "hfl/chinese-roberta-wwm-ext"
    model_p = r"saved_model/model/model_epoch_39"
    slot_p = r"../datasets/industrial/slots.json"

    detector = BSTDetector.from_pretrained(model_path=model_p, tokenizer_path=tokenizer_p, slot_path=slot_p)
    test_data = r"../datasets/industrial/val.json"

    with open(test_data, "r", encoding="utf-8") as reader:
        raw_data = json.load(reader)

    result = []
    for item in tqdm(raw_data):
        data = detector.predict(item)
        item["predict"] = data

        result.append(item)

    with open(r"../datasets/industrial/result.json", "w", encoding="utf-8") as writer:
        json.dump(result, writer)
