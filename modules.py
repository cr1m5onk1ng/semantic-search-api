import torch
from typing import Iterable, Union, Dict, List, Callable
from data import *
from config import *
from torch import nn
import numpy as np

@dataclass
class ModelOutput:
    loss: Union[torch.Tensor, np.array]


@dataclass
class ClassifierOutput(ModelOutput):
    predictions: Union[torch.Tensor, np.array, None]
    attention: Union[torch.Tensor, None] = None


@dataclass
class SimilarityOutput(ModelOutput):
    embeddings: Union[torch.Tensor, np.array, None]
    scores: Union[torch.Tensor, np.array, List[float], None]


class LearningStrategy(nn.Module):
    """
    Base class for tensor combining strategies
    """
    def __init__(self):
        super(LearningStrategy, self).__init__()

    def forward(self):
        raise NotImplementedError()


class PoolingStrategy(LearningStrategy):
    """
    Base class for classes that provide
    a pooling strategy for a tensor, usually
    an embedding
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor):
        raise NotImplementedError()


class WordPoolingStrategy(PoolingStrategy):
    """
    The representation is pooled by extracting
    all the tokens that are part of a word in the sentence
    and taking their avarage
    """
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params

    def forward(self, embeddings: torch.Tensor, features: WordFeatures, **kwargs):
        word_embeddings = []
        for sen_idx, w_idxs in enumerate(features.indexes):
            curr_w_vectors = embeddings[sen_idx][w_idxs] 
            vectors_avg = torch.mean(curr_w_vectors, dim=0)
            word_embeddings.append(vectors_avg)
        return torch.stack(word_embeddings, dim=0)

class SequencePoolingStrategy(WordPoolingStrategy):
    """
    The representation is pooled by extracting
    all the tokens that are part of a word in the sentence
    and taking their avarage
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor, features: WordFeatures, **kwargs):
        word_embeddings = []
        longest_dim = embeddings.shape[1]
        for sen_idx in range(embeddings.shape[0]):
            new_sequence = []
            tokens_indexes = features.indexes[sen_idx]
            for idx in tokens_indexes:
                curr_w_vectors = embeddings[sen_idx][idx]
                curr_w_vectors = torch.mean(curr_w_vectors, dim=0).to(self.params.device)
                new_sequence.append(curr_w_vectors) 
            pad_n = longest_dim - len(new_sequence)
            padding = [torch.zeros(curr_w_vectors.shape[-1]).to(self.params.device)] * pad_n
            new_sequence += padding
            new_sequence = torch.stack(new_sequence, dim=0).to(self.params.device)
            word_embeddings.append(new_sequence)
        stacked = torch.stack(word_embeddings, dim=0).to(self.params.device)
        print(f"Pooled embedding dim: {stacked.shape}")
        return stacked

        

class AvgPoolingStrategy(PoolingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor, features: EmbeddingsFeatures):
        assert len(embeddings.shape) == 3 #batch, seq_len, embed_size
        mask = features.attention_mask
        #we expand the mask to include the embed_size dimension
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        #we zero out the weights corresponding to the zero positions
        # of the mask and we sum over the seq_len dimension
        sum_embeddings = torch.sum(embeddings * mask, 1)
        #we sum the values of the mask on the seq_len dimension
        # obtaining the number of tokens in the sequence
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        #we take the average
        embeddings = sum_embeddings/sum_mask 
        return embeddings


class CLSPoolingStrategy(PoolingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor, features: EmbeddingsFeatures):
        assert len(embeddings.shape) == 3 #batch, seq_len, embed_size
        #the CLS token corresponds to the first token in the seq_len dimension
        return embeddings[:0:]


class MergingStrategy(LearningStrategy):
    """
    Base class for classes that offer functionalities for 
    merging pretrained and contextualized embeddings
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self):
        raise NotImplementedError()


class EmbeddingsSimilarityCombineStrategy(MergingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, features: DataLoaderFeatures, embed_1: torch.Tensor, embed_2: torch.Tensor):
        out = torch.stack([embed_1, embed_2], dim=0)
        return out


class SentenceEncodingCombineStrategy(MergingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, features: DataLoaderFeatures, pooled: torch.Tensor):
        return pooled


class SentenceBertCombineStrategy(MergingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, features, embeddings_1, embeddings_2):
        diff = torch.abs(embeddings_1 - embeddings_2)
        out = torch.cat((embeddings_1, embeddings_2, diff), dim=-1)
        return out
        

class Pooler(nn.Module):
    """Module that pools the output of another module according to different strategies """
    def __init__(
        self, 
        pooling_strategy: PoolingStrategy, 
        normalize=False
        ):
        super(Pooler, self).__init__()
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize

    def forward(self):
        raise NotImplementedError()


class Loss(nn.Module):
    def __init__(self, params: Configuration):
        super(Loss, self).__init__()
        self.params = params

    def forward(self, hidden_state: torch.Tensor, features: EmbeddingsFeatures) -> ModelOutput:
        raise NotImplementedError()


class SoftmaxLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(self.params.model_parameters.hidden_size, self.params.model_parameters.num_classes)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, hidden_state, features):
        labels = features.labels 
        logits = self.classifier(hidden_state)
        loss = self.loss_function(
            logits.view(-1, self.params.model_parameters.num_classes), 
            labels.view(-1)
        )
        return ClassifierOutput(
            loss = loss,
            predictions = logits
        )
        

class SimilarityLoss(Loss):
    def __init__(self, *args, margin=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin

    def forward(self, embeddings, features):
        raise NotImplementedError()


class ContrastiveSimilarityLoss(SimilarityLoss):
    """Ranking loss based on the measure of cosine similarity """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, features):
        assert embeddings.shape[0] == 2
        distances = 1 - F.cosine_similarity(embeddings[0], embeddings[1], dim=-1)
        loss = 0.5 * (features.labels.float() * distances.pow(2) + (1 - features.labels).float() * F.relu(self.margin - distances).pow(2))
        return ClassifierOutput(
            loss = loss,
            predictions = embeddings
        ) 


class OnlineContrastiveSimilarityLoss(SimilarityLoss):
    """Online contrastive loss as defined in SBERT """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, features):
        assert embeddings.shape[0] == 2
        distance_matrix = 1-F.cosine_similarity(embeddings[0], embeddings[1], dim=-1)
        negs = distance_matrix[features.labels == 0]
        poss = distance_matrix[features.labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]
        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return ClassifierOutput(
            loss = loss,
            predictions = embeddings,
        ) 


class CosineSimilarityLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.loss_function = nn.MSELoss()

    def forward(self, embeddings, features):
        scores = self.similarity(embeddings[0], embeddings[1])
        if isinstance(features, dict):
            labels = features["labels"]
        else:
            labels = features.labels
        loss = self.loss_function(scores, labels.view(-1))
        return ClassifierOutput(
            loss = loss,
            predictions = embeddings
        )


class SimpleDistillationLoss(Loss):
    """
    Distillation loss based on a simple MSE loss
    between the teacher and student embeddings
    """
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss = nn.MSELoss()

    def forward(self, student_embeddings, features):
        teacher_embeddings = features.generate_labels(self.teacher_model)
        loss = self.loss(student_embeddings, teacher_embeddings)
        return ClassifierOutput(
            loss=loss, 
            predictions=torch.stack([student_embeddings, teacher_embeddings], dim=0)
        )