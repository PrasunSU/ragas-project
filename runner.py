from scripts.generate_answers import generate_ui_answers
from scripts.ragas_dataset_generator import RagasTestsetCreator
from scripts.evaluate_with_ragas import RagasEvaluator
import os






azure_config = {
    "api_key": os.getenv("AZURE_API_KEY"),
    "base_url": os.getenv("AZURE_BASE_URL"),
    "model_deployment": os.getenv("AZURE_MODEL_DEPLOYMENT"),
    "model_name": os.getenv("AZURE_MODEL_NAME"),
    "embedding_deployment": os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    "embedding_name": os.getenv("AZURE_EMBEDDING_NAME"),
    "doc_path": os.getenv("DOC_PATH"),
    "output_path": os.getenv("OUTPUT_PATH"),
    "save_graph": os.getenv("SAVE_GRAPH", "True").lower() == "true"
}




dataset = RagasTestsetCreator(azure_config)
dataset.run(10)
generate_ui_answers()
ragasEval=RagasEvaluator(dataset_path="data/ragas_dataset.json", output_path="ragas_report/metrics_output.json")
ragasEval.run_evaluation()
