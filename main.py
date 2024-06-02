from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OllamaEmbeddings(model="llama3")

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter,
)

db = Chroma.from_documents(
    documents=docs,                  # Data
    embedding=embeddings,     # Embedding model
    persist_directory="./chroma_db", # Directory to save data
)
