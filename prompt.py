from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv


load_dotenv()

embeddings = OllamaEmbeddings(model="llama3")

llm = Ollama(model="llama3")

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
# retriever = db.as_retriever()
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    # chain_type="stuff"; Take some stuff from the vector store and "stuff" it into the prompt
    # chain_type="map_reduce"; Build a summary or each document and then feed each summary into final question
    # chain_type="map_rerank"; Find relevant part of each document and give it a score of how relevant it is
    # chain_type="refine"; Build the initial response and then give LLM the opportunity to update it with further context
    chain_type="stuff",
)

result = chain.invoke("What is interesting fact about English language?")

print(result['result'])