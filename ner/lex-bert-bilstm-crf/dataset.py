
import json

import dill
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer
import sys

sys.path.append("/home/ubuntu/sequence_tagging")
from tools.label import LabelProcess, NERLabelling


class LEBertDataset(Dataset):
    def __init__(self, raw_data: list, tags: list, tokenizer: BertTokenizer, max_seq_length=128):
        self.tags = LabelProcess(tags)
        self.tags_num = len(tags)
        self.tokenizer = tokenizer
        self.samples = []
        for a_data in tqdm(raw_data):
            input_ids, position_ids, slot_labels = NERLabelling.encode_lex_bert(text=a_data["query"],
                                                                                slots=a_data["entities"],
                                                                                extend_list=a_data["extends"],
                                                                                tokenizer=self.tokenizer)
            slot_ids = self.tags.encode(slot_labels)

            self.samples.append({
                "input_ids": input_ids,
                "position_ids": position_ids,
                "slot_ids": slot_ids
            })

        def batch_collate_fn(batch_data):
            inputs_ids, slot_ids, position_ids = [], [], []
            for item in batch_data:
                if len(item["input_ids"]) < max_seq_length:
                    inputs_ids.append(item["input_ids"] + ([0] * (max_seq_length - len(item["input_ids"]))))
                    slot_ids.append(item["slot_ids"] + ([0] * (max_seq_length - len(item["slot_ids"]))))

                    position_id = item["position_ids"]
                    extend_ids = []
                    i = position_id[-1]
                    while len(extend_ids) + len(position_id) < max_seq_length:
                        i += 1
                        extend_ids.append(i)
                    position_ids.append(position_id + extend_ids)
                else:
                    inputs_ids.append(item["input_ids"][:max_seq_length])
                    slot_ids.append(item["slot_ids"][:max_seq_length])
                    position_ids.append(item["position_ids"][:max_seq_length])

            return inputs_ids, slot_ids, position_ids

        self.collate_fn = batch_collate_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @classmethod
    def load_from_path(cls, data_path, slot_path, **kwargs):
        with open(data_path, "r", encoding="utf-8") as reader:
            raw_data = json.load(reader)

        with open(slot_path, "r", encoding="utf-8") as reader:
            slot_labels = json.load(reader)

        return cls(raw_data=raw_data, tags=slot_labels, **kwargs)


if __name__ == '__main__':
    data_path = r"../datasets/industrial/train.json"
    slot_path = r"../datasets/industrial/slots.json"

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    special_tokens_dict = {
        "additional_special_tokens": ["<MCD>", "<AMC>", "<ATP>", "<ECD>", "<AEC>", "<CSE>", "<FAC>", "<SYS>", "<COM>",
                                      "<ELE>", "<DTD>", "<ELF>", "<RDN>", "<IOE>", "<OMN>", "<MMF>", "<CLE>", "<MAI>",
                                      "<CAR>", "<IMP>", "<ADJ>", "<INS>", "<ALA>", "<EBN>", "<TER>", "<URG>", "<EAR>"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    datasets = LEBertDataset.load_from_path(data_path=data_path, slot_path=slot_path, tokenizer=tokenizer)

    with open('../datasets/industrial/train.pkl', 'wb') as writer:
        dill.dump(datasets, writer)
