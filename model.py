from config import Configuration
from sentence_transformers import SentenceTransformer
from modules import *
import torch
from torch import nn
from transformers import AutoConfig
from typing import Union, Dict, List
from transformers import AutoModel
import transformers
import numpy as np
from tqdm import tqdm
import os


class BaseEncoderModel(nn.Module):
    """
    base class for all the encoder models
    """
    def __init__(
        self,  
        params: Configuration,
        context_embedder: nn.Module,
        normalize: bool = False
        ):
            super(BaseEncoderModel, self).__init__()
            self.params = params
            self.normalize = normalize
            self.context_embedder = context_embedder
            
    @classmethod
    def load_pretrained(cls, path, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=config)
        return cls(
            params=params,
            context_embedder=context_embedder
        )

    @property
    def model_name(self):
        return self.params.model_parameters.model_name

    def set_hidden_size(self, paraphrase=True):
        if isinstance(self.context_embedder, SentenceTransformer):
            embedder_size = self.context_embedder.get_sentence_embedding_dimension()
        else:
            embedder_size = self.embedding_size
        pretrained_size = self.params.pretrained_embeddings_dim
        hidden_size = embedder_size if \
                    not paraphrase else \
                    embedder_size * 2
        if self.params.model_parameters.use_pretrained_embeddings:
            if self.params.senses_as_features:
                hidden_size = (embedder_size + pretrained_size) * 2
            else:
                hidden_size = embedder_size + pretrained_size
        self.params.model_parameters.hidden_size = hidden_size

    @property
    def config(self):
        return self.context_embedder.config

    @property
    def embedding_size(self):
        if "distilbert" in self.params.model:
            embed_size = self.config.dim
        else:
            embed_size = self.config.hidden_size
        if self.params.model_parameters.hidden_size is not None and self.params.model_parameters.hidden_size < embed_size:
            return self.params.embedding_Size
        return embed_size

    @property
    def params_num(self):
        return sum(param.numel() for param in self.context_embedder.parameters() if param.requires_grad)

    def forward(self):
        raise NotImplementedError()

    def encode(self):
        raise NotImplementedError()


class OnnxSentenceTransformerWrapper(BaseEncoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, attention_mask):
        token_embeddings = self.context_embedder(input_ids=input_ids, attention_mask=attention_mask)[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        return pooled

    @classmethod
    def load_pretrained(cls, path, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=config)
        return cls(
            params=params,
            context_embedder=context_embedder
        )

    def save_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        config_path = os.path.join(path, "model_config.bin")
        torch.save(self.params, config_path)
        self.context_embedder.save_pretrained(path)
        self.context_embedder.config.save_pretrained(path)
        self.params.tokenizer.save_pretrained(path)


class SentenceTransformerWrapper(BaseEncoderModel):
    def __init__(self, pooler: PoolingStrategy, merge_strategy: MergingStrategy, loss: Loss, *args, projection_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooler = pooler
        if merge_strategy is not None:
            self.merge_strategy = merge_strategy
        if loss is not None:
            self.loss = loss
        hidden_size = self.context_embedder.config.hidden_size if not 'distilbert' in self.params.model else self.context_embedder.config.dim
        if projection_dim is not None:
            self.projection = nn.Linear(hidden_size, projection_dim, bias=False)

    def forward(self, features, parallel_mode=True, return_output=False, head_mask=None):
        if parallel_mode:
            features_1 = features.sentence_1_features.to_dict()
            features_2 = features.sentence_2_features.to_dict()
            if head_mask is not None:
                features_1['head_mask'] = head_mask
                features_2['head_mask'] = head_mask
            embed_features_1 = self.context_embedder(**features_1)[0]
            embed_features_2 = self.context_embedder(**features_2)[0]
            if hasattr(self, "projection"):
                embed_1 = self.projection(self.pooler(embed_features_1, features.sentence_1_features))
                embed_2 = self.projection(self.pooler(embed_features_2, features.sentence_2_features))
            else:
                embed_1 = self.pooler(embed_features_1, features.sentence_1_features)
                embed_2 = self.pooler(embed_features_2, features.sentence_2_features)
            merged = self.merge_strategy(features, embed_1, embed_2)
        else:
            assert isinstance(features, EmbeddingsFeatures)
            input_features = features.to_dict()
            if head_mask is not None:
                input_features['head_mask'] = head_mask
            model_output = self.context_embedder(**input_features, output_attentions=return_output, output_hidden_states=return_output)
            if hasattr(self, "projection"):
                pooled = self.pooler(model_output[0], features)
                print(f"Pooled dimension: {pooled.shape}")
                pooled = self.projection(pooled)
            else:
                pooled = self.pooler(model_output[0], features)
            merged = pooled
        if hasattr(self, "projection"):
            merged = self.projection(pooled)
        output = self.loss(merged, features)
        if not parallel_mode:
            if return_output:
               return output, model_output
        return output

    def encode(self, features: EmbeddingsFeatures, return_output=False, **kwargs) -> torch.Tensor:
        output = self.context_embedder(**features.to_dict(), output_attentions=return_output, output_hidden_states=return_output)
        pooled = self.pooler(output[0], features)
        if hasattr(self, "projection"):
            pooled = self.projection(pooled)
        if return_output:
            return pooled, output
        return pooled

    def encode_text(self, documents: List[str], output_np: bool=False) -> Union[torch.Tensor, np.array]:
        self.to(self.params.device)
        length_sorted_idx = np.argsort([len(sen) for sen in documents])
        documents = [documents[idx] for idx in length_sorted_idx]
        encoded_documents = []
        self.eval()
        for start_index in range(0, len(documents), self.params.batch_size):
            sentences_batch = documents[start_index:start_index+self.params.batch_size]   
            encoded_dict = self.params.tokenizer(
                    text=sentences_batch,
                    add_special_tokens=True,
                    padding='longest',
                    truncation=True,
                    max_length=self.params.sequence_max_len,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors='pt'
            )
            input_ids = encoded_dict["input_ids"].to(self.params.device)
            attention_mask = encoded_dict["attention_mask"].to(self.params.device)
            features = EmbeddingsFeatures(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
            )
            with torch.no_grad():
                embeddings = self.encode(features, parallel_mode=False)
            embeddings = embeddings.detach()

            if output_np:
                embeddings = embeddings.cpu()

            encoded_documents.extend(embeddings)
        encoded_documents = [encoded_documents[idx] for idx in np.argsort(length_sorted_idx)]
        
        if output_np:
            encoded_documents = np.asarray([embedding.numpy() for embedding in encoded_documents])
            return encoded_documents
        return torch.stack(encoded_documents)
    
    def get_sentence_embedding_dimension(self):
        return self.context_embedder.config.hidden_size

    def save_pretrained(self, path):
        assert(path is not None)
        if not os.path.exists(path):
            os.makedirs(path)
        self.context_embedder.save_pretrained(path)
        self.params.tokenizer.save_pretrained(path)
        

    @classmethod
    def load_pretrained(cls, path, merge_strategy=None, loss=None, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        embedder_config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=embedder_config)
        pooler = AvgPoolingStrategy()
        return cls(
            params=params,
            context_embedder=context_embedder,
            merge_strategy=merge_strategy,
            pooler=pooler,
            loss=loss
        )
