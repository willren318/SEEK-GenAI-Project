import re
import pandas as pd
import torch
import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from peft import prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from common import logger, MissionType
from mistral_data import MistralData
from mistral_config import MistralConfig


class MistralModel:
    def __init__(self, MistralConfig: MistralConfig, MistralData: MistralData) -> None:
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.3"

        self.model_config = MistralConfig

        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=MistralConfig.bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, MistralConfig.lora_config)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.gradient_checkpointing_enable()

        self.fine_tuing_data = MistralData.get_map_data(MistralData.train_dataset)

        self.trainer = None  # type: SFTTrainer | None

        self.y_pred = None  # type: list | None

        logger.info("Model and tokenizer loaded successfully.")

    def __build_train_arguments(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            optim="paged_adamw_8bit",
            save_steps=5,
            logging_steps=1,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="linear",
            report_to="wandb",
            seed=42,
        )

    def train(self) -> None:
        trainer_args = self.__build_train_arguments()
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.fine_tuing_data["train"],
            args=trainer_args,
            peft_config=self.model_config.lora_config,
        )
        self.trainer.train()
        self.trainer.save_state()
        logger.info("Training completed successfully.")

    def save(self) -> None:
        if self.trainer is None:
            raise ValueError(
                "Trainer is not initialized. Please train the model first."
            )
        save_model_path = "./mistral-7b-lora"
        match self.model_config.mission_type:
            case MissionType.WA:
                save_model_path = "./mistral-7b-lora-WA"
            case MissionType.SA:
                save_model_path = "./mistral-7b-lora-SA"
            case MissionType.SE:
                save_model_path = "./mistral-7b-lora-SE"
            case _:
                raise ValueError(f"Invalid mission type: {MistralConfig.mission_type}")
        self.trainer.save_model(save_model_path)
        self.tokenizer.save_pretrained(save_model_path)
        self.model.save_pretrained(save_model_path)
        logger.info("Model saved successfully.")

    def __find_answer(self, text: str) -> str:
        regex = r"Answer: (.*)"
        result = re.search(regex, text)
        if result:
            return result.group(1).strip()
        else:
            return "None"

    def predict(self, MistralData: MistralData) -> list:
        if self.trainer is None:
            raise ValueError(
                "Trainer is not initialized. Please train the model first."
            )
        self.y_pred = []
        test_data = MistralData.test_dataset["train"]
        logger.info("Predicting...")
        for i in tqdm.tqdm(range(len(test_data))):
            input_text = test_data["input"][i]
            instruction_text = test_data["instruction"][i]
            format_input = f"<s>[INST] {instruction_text} {input_text} [/INST]"
            input_ids = self.tokenizer(format_input, return_tensors="pt").input_ids.to(
                self.device
            )
            attention_mask = self.tokenizer(
                format_input, return_tensors="pt"
            ).attention_mask.to(self.device)
            self.model.gradient_checkpointing_enable()
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=1,
            )
            self.model.gradient_checkpointing_enable()
            # print(sequences[0]["generated_text"])
            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # print(f"Output: {output_text}")
            a = self.__find_answer(output_text)
            # print(f"Answer: {a}")
            self.y_pred.append(a)
        return self.y_pred

    def report(self, MistralData: MistralData) -> None:
        if self.trainer is None:
            raise ValueError(
                "Trainer is not initialized. Please train the model first."
            )
        if self.y_pred is None:
            raise ValueError(
                "No predictions made. Please run the predict method first."
            )

        y_test = MistralData.df_test["y_true"].tolist()
        y_pred = self.y_pred

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred, zero_division=0)
        logger.info(f"Accuracy: {accuracy:.4f}")

        # Calculate precision
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        logger.info(f"Precision: {precision:.4f}")

        # Calculate recall
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        logger.info(f"Recall: {recall:.4f}")

        # Calculate F1 score
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        logger.info(f"F1 Score: {f1:.4f}")

        # Generate classification report
        report = classification_report(y_test, y_pred, zero_division=0)

        logger.info(f"Classification Report: \n{report}")

        report_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        # Convert to DataFrame for better readability
        report_df = pd.DataFrame(report_dict).transpose()

        report_file_name_template = (
            "classification_report_{mission_type}_{model_version}.csv"
        )
        match self.model_config.mission_type:
            case MissionType.WA:
                report_file_name = report_file_name_template.format(
                    mission_type="work_arrangements", model_version="mistral-7b-lora"
                )
            case MissionType.SA:
                report_file_name = report_file_name_template.format(
                    mission_type="salary", model_version="mistral-7b-lora"
                )
            case MissionType.SE:
                report_file_name = report_file_name_template.format(
                    mission_type="seniority", model_version="mistral-7b-lora"
                )
            case _:
                raise ValueError(
                    f"Invalid mission type: {self.model_config.mission_type}"
                )

        report_df.to_csv(report_file_name, index=False)
        logger.info(f"Classification report saved to {report_file_name}")
        pass

    pass
