
import json


with open(r"dataset/result2.json", "r", encoding="utf-8") as reader:
    raw_data = json.load(reader)

predict_all = 0
true_all = 0
slot_acc = 0
for item in raw_data:
    true_slot = item["slots"]
    predict_slot = item["predict_slots"]

    predict_all += len(predict_slot)
    true_all += len(true_slot)

    for k, v in predict_slot.items():
        if k in true_slot:
            name = "".join(v)
            name.replace(" ", "")
            name.replace("#", "")

            if name == true_slot[k]:
                slot_acc += 1

p = slot_acc / predict_all
r = slot_acc / true_all
f1 = 2 * p * r / (p + r)

print("slot_precison:", p)
print("slot_recall:", r)
print("slot_f1:", f1)
