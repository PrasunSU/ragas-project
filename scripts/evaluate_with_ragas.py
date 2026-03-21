import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    answer_correctness,
)
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from config.logger_config import get_logger

logger = get_logger(__name__)

logger.info("Debugging details")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
LLM_BINDING_API_KEY =os.getenv("LLM_BINDING_API_KEY")
LLM_BINDING_HOST = os.getenv("LLM_BINDING_HOST")
AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION")

model = os.getenv("model")
azure_endpoint = os.getenv("azure_endpoint")
api_key = os.getenv("api_key")
api_version = os.getenv("api_version")
deployment = os.getenv("deployment")

class RagasEvaluator:
    def __init__(self, dataset_path="data/ragas_dataset.json", output_path="ragas_report/metrics_output.json"):
        load_dotenv()
        logger.info("Ragas Evaluation starts")
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.llm = AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            openai_api_key=LLM_BINDING_API_KEY,
            azure_endpoint=LLM_BINDING_HOST,
            openai_api_version=AZURE_OPENAI_API_VERSION
        )      
        self.embedding_model = AzureOpenAIEmbeddings(
            model=model,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment=deployment
        )
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def load_dataset(self):
        logger.info("Dataset Loading into evaluator")
        with open(self.dataset_path) as f:
            original_data = json.load(f)

        ragas_data = [{
            "question": item["question"],
            "answer": item["answer"],
            "reference": item["ground_truth"],  # renamed key
            "reference_contexts": item.get("reference_contexts", [])
        } for item in original_data]

        return Dataset.from_list(ragas_data)

    def run_evaluation(self):
        dataset = self.load_dataset()
        result = evaluate(
            dataset,
            metrics=[answer_relevancy, answer_similarity, answer_correctness],
            llm=self.llm,
            embeddings=self.embedding_model
        )
       # print(result)
        result.to_pandas().to_json(self.output_path, orient="records", indent=2)
        return result
