from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv


load_dotenv()

embeddings = OllamaEmbeddings(model="llama3")

llm = Ollama(model="llama3")

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
)

result = chain.invoke("What is interesting fact about English language?")

print(result)