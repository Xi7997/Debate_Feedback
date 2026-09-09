import csv
import json
import re

import tqdm

answer_list = []


def parse_block(block):
    block = block.strip('{}').strip()
    data = {}

    lines = re.split(r'\n(?=[^ \t])', block)
    key = None
    value_lines = []

    for line in lines:
        if ':' in line:
            if key:
                data[key.strip(' "')] = ' '.join(value_lines).strip(' "')
                value_lines = []
            k, v = line.split(':', 1)
            key = k.strip(' "')
            value_lines.append(v.strip(' "'))
        else:
            value_lines.append(line.strip(' "'))
    if key:
        data[key.strip(' "')] = ' '.join(value_lines).strip(' "')
    return data


with open('answer.txt', 'r', encoding='utf-8') as file:
    content = file.read()

    json_objects = re.findall(r'\{.*?\}', content, re.DOTALL)

    for json_object in json_objects:
        json_object = parse_block(json_object)
        answer_list.append(json.dumps(json_object))
print("length of answer list: ",len(answer_list))
Judgement_List = []
with open('answer.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["Case Background", "Plaintiff's claim", "Defendant's Defenses","Court's Final Judgment"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for text in answer_list:
        try:
            data = json.loads(text)
            Case_Background = data['Case Background']
            Plaintiff_Claim = data["Plaintiff's Claims"]
            Defendant_Defenses = data["Defendant's Defenses"]
            Judgement = data["Court's Final Judgment"]
            Judgement_List.append(Judgement)

            writer.writerow({"Case Background": Case_Background,
                             "Plaintiff's claim": Plaintiff_Claim,
                             "Defendant's Defenses": Defendant_Defenses,
                             "Court's Final Judgment": Judgement})
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing entry: {e}, skipping...")

print("Done!")
print("length of Judgement_List: ", len(Judgement_List))
with open('../Prediction/judgement.txt', 'w', encoding='utf-8') as file:
    for judgement in Judgement_List:
        file.write(judgement)
        file.write("\n")
print("Judgement saved in judgement.txt")






