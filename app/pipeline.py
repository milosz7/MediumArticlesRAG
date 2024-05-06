import torch
import pandas as pd
import os
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM



def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


def load_embeddings():
    device = get_device()
    path = os.path.join("data", f"text_embeddings_{device}.pt")
    try:
        embeddings = torch.load(path)
        return embeddings
    except FileNotFoundError:
        raise FileNotFoundError(
            "Embeddings file not found. Please run the notebook `rag-creation.ipynb` to generate embeddings."
        )


def load_text_chunks():
    path = os.path.join("data", "chunks_embedded.csv")
    try:
        text_chunks_df = pd.read_csv(path)
        return text_chunks_df
    except FileNotFoundError:
        raise FileNotFoundError(
            "Text chunks file not found. Please run the notebook `rag-creation.ipynb` to generate text chunks."
        )


def retrieve_similar_embeddings(query, embeddings, model, device, n=5):
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)
    dot_products = util.dot_score(query_embedding, embeddings)
    top_results = torch.topk(dot_products, k=n)

    scores, indices = top_results

    return indices, scores


def search_text(query, embeddings, chunks_df, model, device, n=5):
    indices, scores = retrieve_similar_embeddings(query, embeddings, model, device, n)
    indices = indices.cpu().numpy().ravel()
    scores = scores.cpu().numpy().ravel()

    results = chunks_df.iloc[indices]
    chunks = results["Chunk"].values
    titles = results["Title"].values

    return chunks, titles, scores


def load_encoder():
    device = get_device()
    embedder = SentenceTransformer("all-mpnet-base-v2", device=device)
    embedder.to(device)
    return embedder


def load_llm():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    llm = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
                                               attn_implementation="sdpa",
                                               torch_dtype=torch.float16,
                                               low_cpu_mem_usage=False)

    llm.to(device)
    return llm, tokenizer


def create_prompt(query, context_chunks, tokenizer):
    query_start = "Answer the question: " + query
    answer_requirements = """
Give yourself room to think by extracting relevant passages from the context before answering.
Return just the answer to the question.
Make sure the answer is as explanatory as possible.
Do not include additional information not related to the question that is present in the context.
Use the following reference questions and answers as a style guideline:
1. What is overfitting in machine learning?
   - Overfitting occurs when a model learns to memorize the training data instead of capturing the underlying patterns, leading to poor generalization on unseen data.

2. What is the purpose of a validation set in machine learning?
   - The validation set is used to evaluate the performance of a model during training and to tune hyperparameters to prevent overfitting.

3. What is the difference between precision and recall in binary classification?
   - Precision measures the proportion of true positives among all predicted positives, while recall measures the proportion of true positives among all actual positives.

4. What is the softmax function used for in neural networks?
   - The softmax function is used to convert the raw output of a neural network into probabilities, enabling it to make multi-class predictions.

5. What is transfer learning in deep learning?
   - Transfer learning involves using pre-trained neural network models as a starting point for training on a new task, often resulting in faster convergence and better performance with less data.

6. What is batch normalization in neural networks?
   - Batch normalization is a technique used to normalize the inputs of each layer in a neural network, stabilizing training and accelerating convergence.

7. What is the purpose of the Adam optimizer in deep learning?
   - The Adam optimizer is an adaptive learning rate optimization algorithm that combines the advantages of both AdaGrad and RMSProp, making it widely used in training deep neural networks.

8. What is the curse of dimensionality in machine learning?
   - The curse of dimensionality refers to the increased difficulty of learning and generalizing from data in high-dimensional spaces, leading to sparsity and increased computational complexity.
    """

    query_end = "Do not answer to the above reference questions. Use the following context."
    context = "- " + "\n- ".join([chunk for chunk in context_chunks])
    prompt = "\n".join([query_start, answer_requirements, context, query_end])

    llm_prompt_template = [{
        "role": "user",
        "content": prompt,
    }]

    prompt = tokenizer.apply_chat_template(llm_prompt_template,
                                           tokenize=False,
                                           add_generation_prompt=True)

    return prompt


def generate_response(query, llm, tokenizer, device, context_chunks, max_length=256):
    prompt = create_prompt(query, context_chunks, tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = llm.generate(input_ids, do_sample=True, max_new_tokens=max_length, temperature=0.7)

    response = tokenizer.decode(output[0])

    return response
