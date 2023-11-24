import logging
import os

from phoenix.trace.llama_index import (
    OpenInferenceTraceCallbackHandler,
)
from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    OpenAIEmbedding,
    load_index_from_storage,
    download_loader
)
from llama_index.callbacks import CallbackManager
from llama_index.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import phoenix as px

session = px.launch_app()

# Once you have started a Phoenix server, you can start your LlamaIndex application with the `OpenInferenceTraceCallback` as a callback. To do this, you will have to add the callback to the initialization of your LlamaIndex application:
from phoenix.trace.llama_index import (
    OpenInferenceTraceCallbackHandler,
)

STORAGE_DIR = "./storage"  # directory to cache the generated index
DATA_DIR = "./data"  # directory containing the documents to index

# Initialize the callback handler
callback_handler = OpenInferenceTraceCallbackHandler()

# LlamaIndex application initialization may vary
# depending on your application
service_context = ServiceContext.from_defaults(
#    llm_predictor=LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)),
    llm=OpenAI(model="gpt-3.5-turbo"),
    embed_model=OpenAIEmbedding(model="text-embedding-ada-002"),
    callback_manager=CallbackManager(handlers=[callback_handler]),
)

# View the traces in the Phoenix UI
# px.active_session().url

def get_index():
    logger = logging.getLogger("uvicorn")
    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        logger.info("Creating new index")

        if os.path.exists("./credentials.json"):
            # Grab all emails from the last week
            logger.info("Trying to load Gmail messages")
            GmailReader = download_loader('GmailReader')
            loader = GmailReader(query="newer_than:1d", max_results=100, results_per_page=100, service=None)
            documents = loader.load_data()
            logger.info("Grabbed {} emails".format(len(documents)))
            index = VectorStoreIndex.from_documents(documents,service_context=service_context)
            # store it for later
            index.storage_context.persist(STORAGE_DIR)
            logger.info("Loaded {} emails".format(len(documents)))
        else:
            logger.info("No credentials.json file present.")

        # load the documents and create the index
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents,service_context=service_context)
        # store it for later
        index.storage_context.persist(STORAGE_DIR)
        logger.info(f"Finished creating new index. Stored in {STORAGE_DIR}")
    else:
        # load the existing index
        logger.info(f"Loading index from {STORAGE_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context,service_context=service_context)
        logger.info(f"Finished loading index from {STORAGE_DIR}")
    return index
