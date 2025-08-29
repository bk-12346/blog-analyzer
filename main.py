# implementing retrival

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  
from langchain import hub

# this is going to get the prompt, embed it, send it to vector store, look for similar vectors, retrieve it and send it to the llm
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

if __name__ == "__main__":
    print("Retrieving...")

    query = "What are vector databases? Explain in detail."
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=os.getenv("INDEX_NAME"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    # prompt that we send to llm after we retrieve the info
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # chain
    create_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=create_docs_chain)

    # invoking the retrieval chain
    result = retrieval_chain.invoke({"input": query})
    # print(result)

    ## using only LECL
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say you don't know. Don't try to make up an answer.
    Use a maximum of three sentences and keep the answer as concise as possible.
    Always say "Thanks for asking!" at the end of your response.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
 
    # include RunnablePassthrough so that the question's value remains unchanged and propagates to the final llm call
    # context needs to be filled with relevant documents
    # format_docs takes a document and appends it one to another
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": itemgetter("question") | vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    rag_result = rag_chain.invoke({"question": query})
    print(rag_result)
