
import json
from transformers import BertTokenizer
import re


class LabelProcess(object):
    def __init__(self, labels):
        assert len(labels) == len(set(labels)), "ERROR: repeated labels appeared!"
        self.label_dict = {k: i for i, k in enumerate(labels)}
        self.reverse_label = {i: k for i, k in enumerate(labels)}

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        elif isinstance(idx, str):
            if idx in self.label_dict:
                return self.label_dict[idx]
            else:
                print("ERROR: unknown idx!")
        elif isinstance(idx, int):
            return self.reverse_label[idx]

        assert True, "Warning: unknown indexing type!"
        return None

    def encode(self, labels):
        return self.__getitem__(idx=labels)

    def decode(self, labels):
        return self.__getitem__(idx=labels)

    @classmethod
    def load(cls, load_path, **kwargs):
        labels = json.load(open(load_path, "r"))
        return cls(labels=labels, **kwargs)


class NERLabelling(object):
    @classmethod
    def encode_lex_bert_inputs(cls, text, extend_list, tokenizer: BertTokenizer):
        text_tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.encode(text=text)
        position_ids = list(range(len(input_ids)))
        last_position = len(position_ids)

        keywords = []
        for extend in extend_list:
            t = extend["type"]
            w = extend["word"]
            words = re.finditer(w, text)
            for word in words:
                start = word.start()
                keywords.append({"start": start, "keyword": w, "type": t})
        keywords.sort(key=lambda x: x["start"])

        i = 0
        while i < len(text_tokens):
            keyword_matched = False

            for item in keywords:
                if "position_start" in item:
                    continue

                pattern_tokens = tokenizer.tokenize(item["keyword"])
                if "".join(text_tokens[i: i + len(pattern_tokens)]) == "".join(pattern_tokens):
                    keyword_matched = True
                    item["position_start"] = i + 1
                    i += len(pattern_tokens)
                    break

            if not keyword_matched:
                i += 1

        for item in keywords:
            if "position_start" in item:
                input_ids.append(tokenizer.encode("".join(["<", item["type"], ">"]))[1])
                position_ids.append(item["position_start"])

        # last <SEP>
        input_ids.append(tokenizer.encode("[SEP]")[1])
        position_ids.append(last_position)

        return input_ids, position_ids

    @classmethod
    def encode_lex_bert(cls, text, slots, extend_list, tokenizer: BertTokenizer):
        text_tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.encode(text=text)
        position_ids = list(range(len(input_ids)))
        last_position = len(position_ids)

        slot_labels = cls.encode(text=text, slots=slots, tokenizer=tokenizer, is_bio=True)
        slot_labels = ["O"] + slot_labels + ["O"]

        keywords = []
        for extend in extend_list:
            t = extend["type"]
            w = extend["word"]
            words = re.finditer(w, text)
            for word in words:
                start = word.start()
                keywords.append({"start": start, "keyword": w, "type": t})
        keywords.sort(key=lambda x: x["start"])

        i = 0
        while i < len(text_tokens):
            keyword_matched = False

            for item in keywords:
                if "position_start" in item:
                    continue

                pattern_tokens = tokenizer.tokenize(item["keyword"])
                if "".join(text_tokens[i: i + len(pattern_tokens)]) == "".join(pattern_tokens):
                    keyword_matched = True
                    item["position_start"] = i + 1
                    i += len(pattern_tokens)
                    break

            if not keyword_matched:
                i += 1

        for item in keywords:
            if "position_start" in item:
                input_ids.append(tokenizer.encode("".join(["<", item["type"], ">"]))[1])
                position_ids.append(item["position_start"])
                slot_labels.append("O")

        # last <SEP>
        input_ids.append(tokenizer.encode("[SEP]")[1])
        position_ids.append(last_position)
        slot_labels.append("O")

        return input_ids, position_ids, slot_labels

    @classmethod
    def encode(cls, text, slots, tokenizer: BertTokenizer, is_bio=False, other="O"):
        text_tokens = tokenizer.tokenize(text)

        slot_labels = []
        i = 0
        while i < len(text_tokens):
            slot_matched = False
            for item in slots:
                if slot_matched:
                    break

                pattern_tokens = tokenizer.tokenize(item["value"])
                if "".join(text_tokens[i: i + len(pattern_tokens)]) == "".join(pattern_tokens):
                    slot_matched = True
                    if is_bio:
                        slot_labels.extend(['B-' + item["type"]] + ['I-' + item["type"]] * (len(pattern_tokens) - 1))
                    else:
                        slot_labels.extend([item["type"]] * len(pattern_tokens))
                    i += len(pattern_tokens)
                    break

            if not slot_matched:
                slot_labels.append(other)
                i += 1

        return slot_labels

    @classmethod
    def simple_encode(cls, text, slots, tokenizer, is_bio=False, other="O"):
        text_tokens = tokenizer.tokenize(text)

        slot_labels = []
        i = 0

        while i < len(text_tokens):
            slot_matched = False

            for slot in slots:
                if slot_matched:
                    break

                slot_tokens = tokenizer.tokenize(slot)
                if "".join(text_tokens[i: i + len(slot_tokens)]) == "".join(slot_tokens):
                    slot_matched = True
                    slot_labels.extend(["B_OBJECT"] + ["I_OBJECT"] * (len(slot_tokens) - 1))
                    i += len(slot_tokens)

            if not slot_matched:
                slot_labels.append(other)
                i += 1

        return slot_labels
