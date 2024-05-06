import torch
import pandas as pd
import os
import textwrap
from sentence_transformers import util

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
