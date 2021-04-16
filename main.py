from data import DocumentCorpusDataset
from pipeline import APISearchPipeline
from model import SentenceTransformerWrapper
from config import ModelParameters, Configuration
import transformers
import torch
import random
import onnx
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

random.seed(43)

model = "model/sencoder-distilbert-multi-quora/"
model_config = ModelParameters(
    model_name = "eval_sentence_mining",
    hidden_size=768
)
configuration = Configuration(
    model_parameters=model_config,
    model = model,
    save_path = "./results",
    batch_size = 16,
    device = torch.device("cpu"),
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=True),
    model_path = "./model/sencoder-distilbert-multi-optim/model-opt.onnx"
)

encoder_model = SentenceTransformerWrapper.load_pretrained(model, params=configuration)

num_queries = 1
corpus_percent = 0.001
#document_dataset = load_file("./corpus/jp-wikipedia-dataset")
document_dataset = DocumentCorpusDataset.from_tsv("corpus/ja.wikipedia_250k.txt")
all_sentences = list(document_dataset.sentences)
random.shuffle(all_sentences)
corpus_portion = all_sentences[:int(len(all_sentences)*corpus_percent)]

pipeline = APISearchPipeline(
    name="semantic_search", 
    params=configuration,
    index_path="./index",
    corpus=corpus_portion, 
    model=encoder_model,
    inference_mode=True,
    max_n_results=40)


class SentenceRequestModel(BaseModel):
    text: str
    n_results: int

app = FastAPI()

@app.post("/search")
async def search(sentence_model: SentenceRequestModel):
	results = pipeline(sentence_model.text, max_num_results=sentence_model.n_results)
	return {"results": results}

@app.put("/add")
async def add_item(text: Union[str, List[str]]):
    pipeline.add_elements(text)
