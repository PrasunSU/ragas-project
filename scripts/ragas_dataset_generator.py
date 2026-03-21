import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution
from langchain_community.document_loaders import PyPDFLoader
from config.logger_config import get_logger
logger = get_logger(__name__)



class RagasTestsetCreator:
    def __init__(self, config: dict):
        self.config = config
        self.llm = None
        self.embedding = None
        self.docs = None
        self.knowledge_graph = None

    def setup_env(self):
        os.environ["AZURE_OPENAI_API_KEY"] = self.config["api_key"]

    def load_documents(self):
        logger.info("Loading documents...")
        from langchain_community.document_loaders import PyPDFLoader

        loader = DirectoryLoader(
            self.config["doc_path"],
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        self.docs = loader.load()
        logger.info(f" Loaded {len(self.docs)} documents")

    def init_llm_and_embedding(self):
      #  print(" Initializing LLM and embedding...")
        logger.info(" Initializing LLM and embedding...")
        self.llm = LangchainLLMWrapper(
            AzureChatOpenAI(
                openai_api_version="2025-01-01-preview",
                azure_endpoint=self.config["base_url"],
                azure_deployment=self.config["model_deployment"],
                model=self.config["model_name"],
                validate_base_url=False,
            )
        )
        self.embedding = LangchainEmbeddingsWrapper(
            AzureOpenAIEmbeddings(
                openai_api_version="2023-05-15",
                azure_endpoint=self.config["base_url"],
                azure_deployment=self.config["embedding_deployment"],
                model=self.config["embedding_name"],
            )
        )

    def create_knowledge_graph(self):
      #  print(" Creating knowledge graph...")
        logger.info(" Creating knowledge graph...")
        kg = KnowledgeGraph()
        for doc in self.docs:
            kg.nodes.append(Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata
                }
            ))
        self.knowledge_graph = kg

    def apply_transformations(self):
      #  print(" Applying default transformations...")
        logger.info(" Applying default transformations...")
        trans = default_transforms(
            documents=self.docs,
            llm=self.llm,
            embedding_model=self.embedding
        )
        apply_transforms(self.knowledge_graph, trans)

       # print(f" Transformed KnowledgeGraph: {len(self.knowledge_graph.nodes)} nodes, "
       #       f"{len(self.knowledge_graph.relationships)} relationships")
        logger.info(f" Transformed KnowledgeGraph: {len(self.knowledge_graph.nodes)} nodes, "
              f"{len(self.knowledge_graph.relationships)} relationships")

        if self.config.get("save_graph", False):
            self.knowledge_graph.save("knowledge_graph.json")

    def generate_testset(self, testset_size=10):
       # print(" Generating testset...")
        logger.info(" Generating testset...")
        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embedding,
            knowledge_graph=self.knowledge_graph
        )
        query_distribution = default_query_distribution(self.llm)
        testset = generator.generate(
            testset_size=testset_size,
            query_distribution=query_distribution
        )
        df = testset.to_pandas()
        output_path = self.config.get("output_path", "testset_output.json")
        df.to_json(output_path, orient="records", indent=2)
       # print(f" Testset saved to {output_path}")
        logger.info(f" Testset saved to {output_path}")
        return df

    def run(self, testset_size):
        self.setup_env()
        self.load_documents()
        self.init_llm_and_embedding()
        self.create_knowledge_graph()
        self.apply_transformations()
        return self.generate_testset(testset_size=testset_size)


# ----------------------------
# Configuration & Execution
# ----------------------------
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


if __name__ == "__main__":
    generator = RagasTestsetCreator(config=azure_config)
    testset_df = generator.run(testset_size=10)
    print(testset_df.head())
