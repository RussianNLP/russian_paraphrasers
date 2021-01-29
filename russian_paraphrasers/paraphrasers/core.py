import logging
from abc import abstractmethod
from sentence_transformers import SentenceTransformer, util
from nlgeval import NLGEval
from typing import Any
import torch
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Paraphraser:
    def __init__(
        self,
        model_name: str = "gpt2",
        range_cand: bool = False,
        make_eval: bool = False,
        tokenizer_path: str = "default",
        pretrained_path: str = "default"
    ) -> None:
        """
        Possible models: mt5-large, mt5-base, mt5-small, gpt2, gpt3
        :param model_name:
        :param make_filter:
        :param cache_file_path:
        """
        self.logger = logging.getLogger(__name__)
        self.tokenizer_path = tokenizer_path
        self.pretrained_path = pretrained_path
        self.make_eval = make_eval
        self.range_cand = range_cand
        self.device = torch.device("cpu")
        if self.range_cand:
            self.smodel = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")
        if self.make_eval:
            self.ngeval = NLGEval(
                metrics_to_omit=[
                    "EmbeddingAverageCosineSimilairty",
                    "CIDEr",
                    "METEOR",
                    "SkipThoughtCS",
                    "VectorExtremaCosineSimilarity",
                    "GreedyMatchingScore",
                ]
            )
        self.model_name = model_name
        self._check_model(model_name)

    def _check_model(self, model_name: str) -> bool:
        __models_dict = ["mt5-large", "mt5-base", "mt5-small", "gpt2", "gpt3"]
        if model_name not in __models_dict:
            self.logger.error(
                "There is no such a model for paraphraser! Use one of these: mt5-large, mt5-base, mt5-small, gpt2, gpt3"
            )
            raise ValueError(
                "It looks like you try to call model we do not have or you write the models name wrong."
            )
        return True

    @abstractmethod
    def load(self):
        raise NotImplemented

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        raise NotImplemented
