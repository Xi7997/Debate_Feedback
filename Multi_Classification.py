import csv
from openai import OpenAI
import json
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed



DEBUG = False




class AdversarialAI:
    def __init__(self, dataset_path, method, model_name, task):
        self.n = 0
        self.dataset_path = dataset_path
        self.case_background = []
        self.Labels = []
        self.Outputs = []
        self.method = method
        self.model_name = model_name
        self.Debater = []
        self.Single_Debate_prompts = []
        self.prediction = []
        self.prompt_with_assistant = ""
        self.Assistant_LLM = False
        self.classifier = None
        self.task = task


    def read_dataset(self, ):
        map = dict()
        count = 0
        with open(self.dataset_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                # data : {"key": "value"}
                self.case_background.append(data["fact"])
                self.Labels.append(data["relevant_articles"])
                self.Outputs.append("null")
                if data["relevant_articles"] not in map:
                    map[data["relevant_articles"]] = 0
                map[data["relevant_articles"]] += 1
                count += 1
                if count > 12000:
                    break

        print("Dataset loaded successfully!")
        print("Number of samples:", len(self.case_background))
        print("Number of labels:", len(map))
        self.n = len(self.case_background)
        self.Debater = list(map.keys())
        print("Debater:", self.Debater)



    def run(self):
        if self.method == "zeroshot":
            self.zeroshot(self.model_name, self.method)
        elif self.method == "test":
            self.test(self.model_name, self.method)
        elif self.method == "debate-feedback":
            self.run_debate(self.model_name, self.method)
        elif self.method == "assistant_debate":
            self.assistant_debate(self.model_name, self.method)
        else:
            print("Method not implemented")

    def test(self, model_name, method):
        self.zeroshot(model_name, method)

    def single_debate(self, model_name, method, idx):
        client = OpenAI()
        initial_prompt = (f"Here is a legal case, your task is to predict its relevant law article, the labels can be"
                  f"one of the following: {self.Debater}. \n"
                  f"These labels represent the Criminal Law of the People's Republic of China. \n"
                  f"ONLY output the one of the labels without anything else.\n"
                  f"Here is the case background:\n")
        Debater_prompt = (f"Here is a legal case, your task is to predict its relevant law article and Debate with"
                          f"other experts, the labels can be one of the following: {self.Debater}. \n"
                          f"These labels represent the Criminal Law of the People's Republic of China. \n"
                          f"Please state your opinion and refer to factual examples whenever possible.\n"
                          f"Your answer no more than 200 words.\n"
                          f"Here is the case background:\n"
                          )
        Exchange_prompt = (f"Now I will give you the opinion of another expert, "
                           f"if their opinions differ from yours, you can support their views or offer a counter-argument.\n"
                           f"This round of conversation will be provided to the judge for reference, so remember to express your stance clearly.\n"
                           f"Your answer no more than 200 words.\n"
                           f"Here is their opinion:\n"
                           )
        final_prompt = (f"Some experts discussed the case in this round and here is the summary, please provide your final opinion.\n"
                        f"Same as above, ONLY output the one of the labels without anything else.\n"
                        f"Here is the debate summary:\n"
                        )
        judge_conversation_history = []

        judge_conversation_history.append({"role": "user", "content": initial_prompt+self.case_background[idx]})
        responses = client.chat.completions.create(
            messages=judge_conversation_history,
            model=model_name,
        )

        initial_predict = responses.choices[0].message.content
        judge_conversation_history.append({"role": "system", "content": initial_predict})

        Debate_round = 1
        Number_of_Debater = 2
        Debater_conversation_history = [[] for _ in range(Number_of_Debater)]

        final_predict = ""
        for i in range(Debate_round):
            Debater_opinion = [""] * Number_of_Debater
            for j in range(Number_of_Debater):
                Debater_conversation_history[j].append({"role": "user", "content": Debater_prompt+self.case_background[idx]})
                responses = client.chat.completions.create(
                    messages=Debater_conversation_history[j],
                    model=model_name,
                )
                initial_predict = responses.choices[0].message.content
                Debater_conversation_history[j].append({"role": "system", "content": initial_predict})
                Debater_opinion[j] = initial_predict
            Debater_comment = [""] * Number_of_Debater
            # print("i am here!")
            for j in range(Number_of_Debater):
                for k in range(Number_of_Debater):
                    if j != k:
                        Debater_conversation_history[j].append({"role": "user", "content": Exchange_prompt+Debater_opinion[k]+f"This is the opinion of the {k}-th expert."})
                        responses = client.chat.completions.create(
                            messages=Debater_conversation_history[j],
                            model=model_name,
                        )
                        response = responses.choices[0].message.content
                        Debater_conversation_history[j].append({"role": "system", "content": response})
                        Debater_comment[j] = initial_predict
            summary_of_debate = [issue for issue in Debater_comment]
            summary_of_debate = str(summary_of_debate)
            # print("Debater history:", Debater_conversation_history[0])
            judge_conversation_history.append({"role": "user", "content": final_prompt+summary_of_debate})
            responses = client.chat.completions.create(
                messages=judge_conversation_history,
                model=model_name,
            )
            final_predict = responses.choices[0].message.content
            judge_conversation_history.append({"role": "system", "content": final_predict})
            # print("Judge history:", judge_conversation_history)
        self.Outputs[idx] = final_predict





    def run_debate(self, model_name, method):
        prediction = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures_to_case = {executor.submit(self.single_debate, model_name, method,idx): idx for idx in range(len(self.case_background))}
            with tqdm.tqdm(total=len(self.case_background)) as pbar:
                for future in as_completed(futures_to_case):
                    try:
                        answer = future.result()
                        if answer == True:
                            prediction.append("Correct")
                        else:
                            prediction.append("Incorrect")
                        index = futures_to_case[future]
                        pbar.update(1)
                    except Exception as e:
                        print(f"An error occurred: {e}")
        self.save(model_name, method)

    def assistant_debate(self, model_name, method):
        model_path = "assistant_model//trained_model.pt"
        self.Assistant_LLM = True
        # self.classifier = TextClassificationModel(model_path)
        print("Assistant Model loaded successfully!")
        self.run_single_debate(model_name, method)

    def zeroshot(self, model_name, method):
        prompt = (f"Here is a legal case, your task is to predict its relevant law article, the labels can be"
                  f"one of the following: {self.Debater}. \n"
                  f"These labels represent the Criminal Law of the People's Republic of China. \n"
                  f"ONLY output the one of the labels without anything else.\n"
                  f"Here is the case background:\n")

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures_to_case = {executor.submit(self.model_request, model_name, prompt+self.case_background[idx], self.Labels[idx], idx): idx for idx in range(len(self.case_background))}
            with tqdm.tqdm(total=len(self.Labels)) as pbar:
                for future in as_completed(futures_to_case):
                    try:
                        answer = future.result()
                        if answer == True:
                            self.prediction.append("Correct")
                        else:
                            self.prediction.append("Incorrect")
                        pbar.update(1)
                    except Exception as e:
                        print(f"An error occurred: {e}")
        self.save(model_name, method)



    def single_model_request(self, model_name, prompt):
        # print("Prompt:", prompt)
        client = OpenAI()
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"{prompt}"
            }],
            model=model_name,
        )
        return response.choices[0].message.content

    def model_request(self, model_name, prompt, ground_truth, idx):
        # print("Prompt:", prompt)
        client = OpenAI()
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"{prompt}"
            }],
            model=model_name,
        )
        predict = response.choices[0].message.content
        self.Outputs[idx] = predict
        # print("Ground Truth:", ground_truth)
        # print("Prediction:", predict)
        if DEBUG:
            print("Ground Truth:", ground_truth,".  Prediction:", predict)

        if ground_truth == predict:
            return True
        return False
        # return response.choices[0].message.content

    def save(self, model_name, method):
        file_name = f"Multi-Result/{self.task} & {method} & {model_name} & data{self.n}.txt"
        with open(file_name, "w", encoding='utf-8') as f:
            for label, predict in zip(self.Labels, self.Outputs):
                str = f"{label}   {predict}"
                f.write(str)
                f.write("\n")
        print("Prediction saved in prediction.txt")



if __name__ == "__main__":
    dataset_path = "data/ceil2018/article_test.json"
    model_name = "gpt-4o-mini"
    method = "zeroshot"
    task = "article-prediction"
    print("Dataset loading..., predict method is:", method, ", model name is:", model_name, ", dataset path is:",
          dataset_path)
    adversarial_ai = AdversarialAI(dataset_path, method, model_name, task)
    adversarial_ai.read_dataset()
    adversarial_ai.run()

