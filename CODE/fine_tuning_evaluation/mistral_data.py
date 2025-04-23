import os
import json
import tqdm
import pandas as pd
import datasets
from datasets import load_dataset

from common import (
    JOB_DATA_PATH,
    MISC_DATA_PATH,
    logger,
    MissionType,
    JobDataPath,
)


class MistralData:
    def __init__(self, mission_type: MissionType):
        func_join_path = lambda x: os.path.join(JOB_DATA_PATH, x)
        match mission_type:
            case MissionType.WA:
                # Load the data
                self.df_dev = pd.read_csv(func_join_path(JobDataPath.WA_DEV))
                self.df_test = pd.read_csv(func_join_path(JobDataPath.WA_TEST))
                # Create train dataset
                train_jsonl_list = self.__output_prompt_jsonl_WA(self.df_dev)
                train_jsonl_path = os.path.join(
                    MISC_DATA_PATH, "mistral", "work_arrangements_train.jsonl"
                )
                self.__write_jsonl(train_jsonl_path, train_jsonl_list)
                self.train_dataset = load_dataset("json", data_files=train_jsonl_path)
                # Create test dataset
                test_jsonl_list = self.__output_prompt_jsonl_WA(self.df_test)
                test_jsonl_path = os.path.join(
                    MISC_DATA_PATH, "mistral", "work_arrangements_test.jsonl"
                )
                self.__write_jsonl(test_jsonl_path, test_jsonl_list)
                self.test_dataset = load_dataset("json", data_files=test_jsonl_path)
            case MissionType.SA:
                # Load the data
                self.df_dev = pd.read_csv(func_join_path(JobDataPath.SA_DEV))
                self.df_test = pd.read_csv(func_join_path(JobDataPath.SA_TEST))
                # Create train dataset
                train_jsonl_list = self.__output_prompt_jsonl_SA(self.df_dev)
                train_jsonl_path = os.path.join(
                    MISC_DATA_PATH, "mistral", "salary_train.jsonl"
                )
                self.__write_jsonl(train_jsonl_path, train_jsonl_list)
                self.train_dataset = load_dataset("json", data_files=train_jsonl_path)
                # Create test dataset
                test_jsonl_list = self.__output_prompt_jsonl_SA(self.df_test)
                test_jsonl_path = os.path.join(
                    MISC_DATA_PATH, "mistral", "salary_test.jsonl"
                )
                self.__write_jsonl(test_jsonl_path, test_jsonl_list)
                self.test_dataset = load_dataset("json", data_files=test_jsonl_path)
            case MissionType.SE:
                # Load the data
                self.df_dev = pd.read_csv(func_join_path(JobDataPath.SE_DEV))
                self.df_test = pd.read_csv(func_join_path(JobDataPath.SE_TEST))
                # Create train dataset
                train_jsonl_list = self.__output_prompt_jsonl_SE(self.df_dev)
                train_jsonl_path = os.path.join(
                    MISC_DATA_PATH, "mistral", "seniority_train.jsonl"
                )
                self.__write_jsonl(train_jsonl_path, train_jsonl_list)
                self.train_dataset = load_dataset("json", data_files=train_jsonl_path)
                # Create test dataset
                test_jsonl_list = self.__output_prompt_jsonl_SE(self.df_test)
                test_jsonl_path = os.path.join(
                    MISC_DATA_PATH, "mistral", "seniority_test.jsonl"
                )
                self.__write_jsonl(test_jsonl_path, test_jsonl_list)
                self.test_dataset = load_dataset("json", data_files=test_jsonl_path)
            case _:
                self.df_dev = None
                self.df_test = None
                raise ValueError(f"Invalid mission type: {mission_type}")
        pass

    def __output_prompt_jsonl_SA(self, df: pd.DataFrame) -> list:
        logger.info(f"Creating jsonl for {df.keys()} rows")
        jsonl_list = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating jsonl"):
            jsonl_list.append(
                {
                    "instruction": (
                        "You are a helpful assistant."
                        "Your task is to classify the salary level of the job advertisement. "
                        "The format will be [NUMBER]-[NUMBER]-[CURRENCY SYMBOL]-[HOURLY|DAILY|MONTHLY|YEARLY|ANNUAL]."
                    ),
                    "input": (  # job_title job_ad_details nation_short_desc salary_additional_text
                        f"The job title is: {row['job_title']}."
                        f"Further details: {row['job_ad_details']}."
                        f"Country Codes: {row['nation_short_desc']}."
                        f"Salary information: {row['salary_additional_text']}."
                    ),
                    "output": f"Answer: {row['y_true']}",
                }
            )
        return jsonl_list

    def __output_prompt_jsonl_SE(self, df: pd.DataFrame) -> list:
        logger.info(f"Creating jsonl for {df.keys()} rows")
        jsonl_list = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating jsonl"):
            jsonl_list.append(
                {
                    "instruction": (
                        "You are a helpful assistant."
                        "Your task is to classify the seniority level of the job advertisement."
                        "The format will be [SENIORITY LEVEL]."
                    ),
                    "input": (
                        f"The job title is: {row['job_title']}."
                        f"To summarize the job: {row['job_summary']}."
                        f"Further details: {row['job_ad_details']}"
                        f"The classification is: {row['classification_name']} / {row['subclassification_name']}."
                    ),
                    "output": f"Answer: {row['y_true']}",
                }
            )
        return jsonl_list

    def __output_prompt_jsonl_WA(self, df: pd.DataFrame) -> list:
        logger.info(f"Creating jsonl for {df.keys()} rows")
        jsonl_list = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating jsonl"):
            jsonl_list.append(
                {
                    "instruction": (
                        "You are a helpful assistant."
                        "Your task is to classify which type of work arrangements the job advertisement belongs to."
                        "There are three types: ['OnSite', 'Remote', 'Hybrid']."
                    ),
                    "input": row["job_ad"],
                    "output": f"Answer: {row['y_true']}",
                }
            )
        return jsonl_list

    def __write_jsonl(self, jsonl_path: str, data: list) -> None:
        # if file exists, remove it
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)
        # create file and write data
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def format_prompt(self, example) -> str:
        return f"<s>[INST] {example['instruction']} {example['input']} [/INST] {example['output']} </s>"

    def get_map_data(self, dataset: datasets.Dataset) -> datasets.Dataset:
        return dataset.map(
            lambda example: {"text": self.format_prompt(example)},
            remove_columns=["instruction", "input", "output"],
        )

    pass


pass
