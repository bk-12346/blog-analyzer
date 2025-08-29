Blog Analyzer with LangChain
============================

This project is a blog analyzer built using the LangChain framework. It demonstrates how to retrieve and analyze information from a blog post by leveraging a large language model (LLM) and a vector database.

Features
--------

-   **Ingestion Script:** A script to load a blog post, split its content into manageable chunks, create embeddings, and store them in a Pinecone vector database.

-   **Retrieval-Augmented Generation (RAG):** The core of the project uses a RAG pipeline to retrieve relevant information from the vector store and provide it as context to the LLM for accurate, context-aware answers.

-   **LangChain Expression Language (LCEL):** The project showcases two different approaches for building the retrieval pipeline: a basic `create_retrieval_chain` and a more flexible, custom chain built with LCEL.

Prerequisites
-------------

Before running this project, you need to have Python installed and set up your API keys for OpenAI and Pinecone.

-   **Python 3.10+**

-   **OpenAI API Key**: For embedding and LLM calls.

-   **Pinecone API Key & Environment**: For your vector database.

Installation
------------

1.  **Clone the repository:**

    Bash

    ```
    git clone [your-repository-url]
    cd [your-project-directory]

    ```

2.  **Create and activate a virtual environment:**

    Bash

    ```
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate

    ```

3.  **Install the required packages:** You will need to create a `requirements.txt` file based on your project's dependencies (e.g., `langchain`, `langchain-openai`, `langchain-pinecone`, `python-dotenv`).

    Bash

    ```
    pip install -r requirements.txt

    ```

Setup
-----

1.  **Create a `.env` file** in the root of your project.

2.  **Add your API keys and configuration** to the file in the following format:

    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    PINECONE_API_KEY="your_pinecone_api_key_here"
    PINECONE_ENVIRONMENT="your_pinecone_environment_here"
    INDEX_NAME="your_pinecone_index_name"

    ```

    *Note: Your Pinecone `INDEX_NAME` should match the index name you created in your Pinecone dashboard.*

Usage
-----

### Step 1: Ingest the Data

First, run the `ingestion.py` script to process the blog post and store the embeddings in your Pinecone index. Ensure your Pinecone index is created and has the correct dimensions (e.g., 1536 for modern OpenAI embedding models).

Bash

```
python ingestion.py

```

### Step 2: Run the Retrieval Script

Once the ingestion is complete, run the `main.py` script to perform a retrieval query against your vector database using the RAG pipeline.

Bash

```
python main.py

```

The script will output the results from both the `create_retrieval_chain` and the custom LCEL chain.
