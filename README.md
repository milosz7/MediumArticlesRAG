### Medium Articles RAG pipeline

The project was built using Python 3.11.7 on MacOS. All code should also work properly on Linux. I cannot guarantee that it will work on Windows as I do not have any way to check it.

#### 1. How to run the project?

Clone the repository and run the following command in the terminal:

```bash
python3 -m pip install -r requirements.txt
```

After installing the requirements, run all cells in the notebook `rag-creation.ipynb`.

The notebook will generate embeddings and preprocessed dataset in the `data` folder.

After running the notebook change the directory to root and run the following command in the terminal:

```bash
streamlit run app/app.py
```

The command will start the Streamlit app. You can access the app by opening the link in the terminal.\
The app will be available at `http://localhost:8501/`.

#### 2. What is the project about?

The project is about creating a pipeline for generating answers to questions using the RAG model.

The pipeline consists of the following steps:

1. Query embeddings generation
2. Retrieval of relevant documents
3. Answer generation

The project uses `all-mpnet-base-v2` for the embeddings generation and `google/gemma-2b-it` for the answer generation.

The indexing process is done using the same model that was used for the embeddings generation. I use dot product similarity for the retrieval of relevant documents.

In the query I use 5 chunks from the dataset to augment the query.

The chunks are generated using the following code:

```python
n_sentences_in_chunk = 10
CONTEXT_WINDOW = 384
TOKEN_SIZE = 4


def split_text_into_chunks(text: str, n_sentences_in_chunk: int, overlap=0) -> list:
    doc = nlp(text)
    sents = [sent.text for sent in doc.sents]
    chunks = []
    n_sentences_in_chunk = n_sentences_in_chunk - overlap
    remainder = 0
    for i in range(overlap, len(sents), n_sentences_in_chunk):
        if remainder == n_sentences_in_chunk:
            remainder = 0
        chunk = sents[i-overlap-remainder:i + n_sentences_in_chunk-remainder]
        remainder = 0
        while len(" ".join(chunk)) / TOKEN_SIZE > CONTEXT_WINDOW:
            remainder += 1
            chunk = chunk[:-remainder]
            
        chunk = " ".join(chunk)
        chunks.append(chunk)
    return chunks
```

As the model's context window is 384 tokens, I split the text into chunks of max 384 tokens or 10 sentences. If the chunk is bigger than 384 tokens, I remove the last sentence until the chunk is smaller than 384 tokens.
Chunks are made with an overlap of 1 sentence.

#### 3. Challenges encountered

 - Text chunking was a challenging task as I wanted to do it manually without using libraries.

 - Creating an interface for the pipeline was also a challenge as I had to learn how to use Streamlit.

  - Prompt engineering - I had to experiment with different prompts to get the best results.

#### 4. Future improvements

 - Use a more advanced model for the embeddings generation and generation of answers.
 - Improve text chunking. Use libraries like `LangChain` for more precise chunking.
 - Rerank the documents using a more advanced method like BM25. https://www.sbert.net/examples/applications/retrieve_rerank/README.html
 - Improve the interface to add more functionalities, maybe add model selection.
 - Further improvements of prompt engineering.

#### 5. Using the app

The app has a simple interface. You can input a question and get the answer. The app will show the top 5 documents that were retrieved and the answer generated by the model.
![Image demo](https://github.com/milosz7/MediumArticlesRAG/blob/main/preview.png "Preview")

