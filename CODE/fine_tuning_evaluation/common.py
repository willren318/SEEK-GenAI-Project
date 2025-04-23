# -*- coding: utf-8 -*-

import logging
import os
from enum import EnumType
import pandas as pd

EXIT_FAILURE = 1
EXIT_SUCCESS = 0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Mistral Logger")


class MissionType(EnumType):
    WA = 1
    SA = 2
    SE = 3


# MISC_DATA_PATH = os.path.join(os.path.pardir, "MISC")
MISC_DATA_PATH = os.path.join(os.path.pardir, os.path.pardir, "MISC")
# MISC_DATA_PATH = os.path.join(os.path.curdir, "drive/MyDrive/COMP6713/MISC")
JOB_DATA_PATH = os.path.join(MISC_DATA_PATH, "job_data_files")

print(f"MISC_DATA_PATH: {MISC_DATA_PATH}")
print(f"JOB_DATA_PATH: {JOB_DATA_PATH}")


class JobDataPath(EnumType):
    WA_DEV = "work_arrangements_development_set.csv"
    WA_TEST = "work_arrangements_test_set.csv"
    SA_DEV = "salary_labelled_development_set.csv"
    SA_TEST = "salary_labelled_test_set.csv"
    SE_DEV = "seniority_labelled_development_set.csv"
    SE_TEST = "seniority_labelled_test_set.csv"


class JobData:
    def __init__(self, mission_type: MissionType):
        func_join_path = lambda x: os.path.join(JOB_DATA_PATH, x)
        match mission_type:
            case MissionType.WA:
                self.df_dev = pd.read_csv(func_join_path(JobDataPath.WA_DEV))
                self.df_test = pd.read_csv(func_join_path(JobDataPath.WA_TEST))
            case MissionType.SA:
                self.df_dev = pd.read_csv(func_join_path(JobDataPath.SA_DEV))
                self.df_test = pd.read_csv(func_join_path(JobDataPath.SA_TEST))
            case MissionType.SE:
                self.df_dev = pd.read_csv(func_join_path(JobDataPath.SE_DEV))
                self.df_test = pd.read_csv(func_join_path(JobDataPath.SE_TEST))
            case _:
                self.df_dev = None
                self.df_test = None
                raise ValueError(f"Invalid mission type: {mission_type}")

    pass


pass
