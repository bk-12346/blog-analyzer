# we take an article from Medium, split it into chunks, embed everything and store it in a vector database

from dotenv import load_dotenv
load_dotenv()

# document loaders are class implementations of how to load and process data so it can be used by the llm
# text splitter is a class that takes a document and splits it into chunks
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# chunk overlap parameter specifies the amount of overlap between the chunks when we split the text into smaller parts
# overlap can be helpful to ensure that the text isn't split in a way that disturbs the context or meaning of the text
# length function helps Langchain determine the chunk size


if __name__ == "__main__":
    print("Ingestion script started...")

    # load the file
    # different loaders can be used for different files; see document_loaders documentation Langchain
    loader = TextLoader("C:\\Users\\Lenovo\\Documents\\LangChain_Course\\blog-analyzer\\mediumblog1.txt")
    document = loader.load()
    print("Document loaded successfully.")

    # split the document into chunks
    # chunk size shouldn't be too small or too large, too much info, llm won't be able to process it effectively
    # overlapping chunks are helpful when we don't want to use context between chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # The split_documents method correctly splits the document object and returns a list of new Document objects.
    chunks = text_splitter.split_documents(document)

    # create embeddings for the chunks
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    # chunk_embeddings = embeddings.embed_documents(chunks)
    # print("Chunk embeddings created successfully.")

    # store the embeddings in a vector database
    # it's going to iterate over every chunk, create an embedding for it and store it in the vector database
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("finish")
    # vector_store = PineconeVectorStore()
    # vector_store.add_embeddings(chunk_embeddings)
    # print("Chunk embeddings stored in vector database successfully.")
