import numpy as np
import json
import tqdm
import re
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

path = "present_data.jsonl"


def read_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# generate_prompt_path = "../PromptSet/Generate_Summary.txt"
# test_path = "../PromptSet/test.txt"
# with open(generate_prompt_path, "r") as f:
#     generate_prompt = f.read()
# with open(test_path, "r") as f:
#     test = f.read()
# print("####################################################")

def call_openai_api(case_description):
    prompt = (
        f"I have a legal case description that includes the case background, the plaintiff's claims, the defendant's defenses, and the court's final judgment. Please help me summarize the following information from the description:\n"
        f"1. Case Background: Include the time, location, key individuals involved, and what happened.No more than 500 words\n"
        f"2. Plaintiff's Claims: Summarize the plaintiff's main arguments and legal requests.No more than 200 words\n"
        f"3. Defendant's Defenses: Summarize the defendant's key arguments and their requests.No more than 200 words\n"
        f"4. Court's Final Judgment: Summarize the court's final decision and classify it into one of the following labels: Plaintiff wins, Defendant wins, Settlement, Dismissed.\n"
        f"\n"
        f"Here is the case description:\n"
        f"\n{case_description}\n"
        f"Please extract and summarize the information based on the points mentioned above, and provide the court's final judgment with one of the specified labels."
        f"Your answer should follow the structures below:\n"
        f"{{\n"
        f"Case Background:[YOUR OUTPUT]\n"
        f"Plaintiff's Claims:[YOUR OUTPUT]\n"
        f"Defendant's Defenses:[YOUR OUTPUT]\n"
        f"Court's Final Judgment:[YOUR OUTPUT,only include one of these four labels(Plaintiff wins, Defendant wins, Settlement, Dismissed).]\n"
        f"}}\n"
        )
    client = OpenAI()
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        model="gpt-4o-mini",
    )
    return response.choices[0].message.content


datas = read_jsonl(path)
print("Data loaded successfully!")
print("length of data: ", len(datas))
index_start = 0
index_length = 10
DataSet = datas[index_start:index_start + index_length]
answer_list = []

with ThreadPoolExecutor(max_workers=50) as executor:
    future_to_index = {executor.submit(call_openai_api, DataSet[i]): i for i in range(len(DataSet))}
    with tqdm.tqdm(total=len(DataSet)) as pbar:
        for future in as_completed(future_to_index):
            try:
                answer = future.result()
                index = future_to_index[future]
                pbar.update(1)
                answer_list.append(answer)
            except Exception as e:
                print(f"An error occurred: {e}")

print("Data processed successfully!")
with open("article.txt", "a", encoding='utf-8') as f:
    for answer in answer_list:
        f.write(answer)
        f.write("\n")
print("Answer saved in article.txt")
