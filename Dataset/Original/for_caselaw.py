# -*- coding: UTF-8 -*-
import csv
import json
import os


csv_path = "answer.csv"
new_data = []
with open(csv_path, "r", encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        new_data.append(row)

print("Length of new_data:", len(new_data))
print("new_data[0]:", new_data[0][3])

case_data = []
map = dict()
map["Plaintiff wins"] = 0
map["Defendant wins"] = 0
for data in new_data:
    labels = data[3]
    if labels not in ["Plaintiff wins", "Defendant wins"]:
        continue
    if map["Defendant wins"] > 100 and map["Plaintiff wins"] > 1.75*map["Defendant wins"] and labels == "Plaintiff wins":
        continue
    case_data.append(data)
    if labels not in map:
        map[labels] = 0
    map[labels] += 1

print("Length of case_data:", len(case_data))
print("map:", map)

save_path = "Caselaw/Caselaw_data.csv"
with open(save_path, "w", encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "case_background", "labels"])
    case_background = [data[0]+'\t'+data[1]+'\t'+data[2] for data in case_data]
    labels = [data[3] for data in case_data]
    for i in range(len(case_background)):
        writer.writerow([i, case_background[i], labels[i]])
print("Data saved successfully!")

save_path = "Caselaw/Caselaw_data.json"
with open(save_path, "w", encoding='utf-8') as f:
    for data in case_data:
        json_obj = dict()
        json_obj["case_background"] = data[0]+'\t'+data[1]+'\t'+data[2]
        json_obj["labels"] = data[3]
        json.dump(json_obj, f)
        f.write('\n')

# shuffle data
import random
random.shuffle(case_data)
test_case = case_data[:2000]
train_case = case_data[2000:]
eval_case = train_case[:1000]
print("Length of train_case:", len(train_case))
print("Length of eval_case:", len(eval_case))
print("Length of test_case:", len(test_case))

with open("Caselaw/train.json", "w", encoding='utf-8') as f:
    for data in train_case:
        json_obj = dict()
        json_obj["case_background"] = data[0]+'\t'+data[1]+'\t'+data[2]
        json_obj["labels"] = data[3]
        json.dump(json_obj, f)
        f.write('\n')
print("train.json saved successfully!")
with open("Caselaw/eval.json", "w", encoding='utf-8') as f:
    for data in eval_case:
        json_obj = dict()
        json_obj["case_background"] = data[0]+'\t'+data[1]+'\t'+data[2]
        json_obj["labels"] = data[3]
        json.dump(json_obj, f)
        f.write('\n')
print("eval.json saved successfully!")
with open("Caselaw/test.json", "w", encoding='utf-8') as f:
    for data in test_case:
        json_obj = dict()
        json_obj["case_background"] = data[0]+'\t'+data[1]+'\t'+data[2]
        json_obj["labels"] = data[3]
        json.dump(json_obj, f)
        f.write('\n')
print("test.json saved successfully!")







print("Data saved successfully!")

