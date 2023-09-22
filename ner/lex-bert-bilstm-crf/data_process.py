
import pandas as pd
import json

MAX_LENGTH = 12
dictionary_p = r"../datasets/industrial/dictionaries.json"
with open(dictionary_p, "r", encoding="utf-8") as reader:
    dictionaries = json.load(reader)
dictionary = list(dictionaries.keys())


def left_max(inputstr):
    results = list()
    while len(inputstr) > 0:
        if len(inputstr) < MAX_LENGTH:
            subSeq = inputstr
        else:
            subSeq = inputstr[:MAX_LENGTH]

        while len(subSeq) > 0:
            if (subSeq in dictionary) or (len(subSeq) == 1):
                if subSeq in dictionary:
                    results.append(subSeq)
                inputstr = inputstr[len(subSeq):]
                break
            else:
                subSeq = subSeq[:len(subSeq) - 1]
    return results


class Format(object):
    @classmethod
    def standardization(cls, source_path: str, json_path: str):
        data_list = []
        single_chars = []
        single_tags = []
        with open(source_path, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.replace("\n", "")
                parts = line.split(" ")
                if len(parts) == 2:
                    char, tag = parts[0], parts[1]
                    single_chars.append(char)
                    single_tags.append(tag)
                else:
                    query, entities = cls.normal(chars=single_chars, tags=single_tags)
                    data_list.append({"query": query, "entities": entities})
                    single_chars, single_tags = [], []

        with open(json_path, "w", encoding="utf-8") as writer:
            json.dump(data_list, writer, ensure_ascii=False)

    @classmethod
    def normal(cls, chars: list, tags: list, delimiter="-"):
        entity_mark = dict()
        entity_pointer = None

        for index, label in enumerate(tags):
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
                name += chars[i]
            entities.append({"start": start, "end": start + len(value), "type": e_type, "value": name})

        query = "".join(chars)
        return query, entities

    @classmethod
    def slots_category(cls, slot_path: str, save_path: str):
        slots = ["O"]

        types = []
        with open(slot_path, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.replace("\n", "")
                if line:
                    types.append(line)
        types = sorted(types)

        for t in types:
            slots.append("-".join(["B", t]))
            slots.append("-".join(["I", t]))

        with open(save_path, "w", encoding="utf-8") as writer:
            json.dump(slots, writer, ensure_ascii=False)

    @classmethod
    def create_dictionary(cls, slot_path: str, save_path: str):
        data_frame = pd.read_excel(slot_path)
        dictionaries = dict()
        for item in data_frame.to_dict(orient="records"):
            dictionaries[item["name"]] = item["type"]

        with open(save_path, "w", encoding="utf-8") as writer:
            json.dump(dictionaries, writer, ensure_ascii=False)

    @classmethod
    def second_retriever(cls, raws: list, dictionaries: dict, save_path: str):
        items = []
        for item in raws:
            query = item["query"]
            words = list(set(left_max(query)))

            words.sort(key=len, reverse=True)
            extends = []
            for word in words:
                extends.append({"type": dictionaries.get(word), "word": word})
            item["extends"] = extends

            items.append(item)

        with open(save_path, "w", encoding="utf-8") as writer:
            json.dump(items, writer, ensure_ascii=False)


if __name__ == '__main__':
    # train_path = r"../datasets/industrial/val.txt"
    # save_p = r"../datasets/industrial/val.json"
    #
    # Format.standardization(source_path=train_path, json_path=save_p)
    #
    # slot_p = r"../datasets/industrial/标签种类.txt"
    # sa_p = r"../datasets/industrial/slots.json"
    # Format.slots_category(slot_path=slot_p, save_path=sa_p)
    #
    # slot_p = r"../datasets/industrial/标注特征词.xlsx"
    # sa_p = r"../datasets/industrial/dictionaries.json"
    # Format.create_dictionary(slot_path=slot_p, save_path=sa_p)

    # 二次召回，找出query中存在的关键词

    with open(r"../datasets/industrial/val.json", "r", encoding="utf-8") as reader:
        raws = json.load(reader)
    Format.second_retriever(raws=raws, dictionaries=dictionaries, save_path=r"../datasets/industrial/val.json")
