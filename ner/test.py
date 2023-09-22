import json

rel_str = r"(.+)"
result_path = r"/mnt/storage/bury_boner/sequence_tagging/datasets/objects/intent.json"
intent_dict = json.load(open(result_path))

result = {}
for k, v in intent_dict.items():
    h = list(set(v))
    h.sort(key=lambda x:len(x.replace(rel_str, "").replace(rel_str, "").replace(rel_str, "")), reverse=True)
    result[k] = h

h_path = r"/mnt/storage/bury_boner/sequence_tagging/datasets/objects/intent_1.json"
with open(h_path, "w", encoding="utf-8") as writer:
    json.dump(result, writer, ensure_ascii=False)