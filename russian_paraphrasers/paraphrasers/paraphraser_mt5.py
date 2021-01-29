import torch
from typing import Dict
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from russian_paraphrasers.utils import set_seed
from russian_paraphrasers.paraphrasers import Paraphraser
from russian_paraphrasers.candidates_filter_metrics import (
    get_scores,
    range_candidates,
    check_input,
)
import logging


class Mt5Paraphraser(Paraphraser):
    def __init__(
        self,
        model_name: str = "mt5-large",
        range_cand: bool = False,
        make_eval: bool = False,
        tokenizer_path: str = "default",
        pretrained_path: str = "default"
    ):
        """
        The class for Mt5 hugging_face interface.
        :param model_name: "mt5-small" or "mt5-base" or "mt5-large". Will call models
        :param range_cand: True/False. Range candidates
        :param make_eval: True/False. Make or not average evaluation for n samples.
        :param tokenizer_path: "default" or some model name in hugging_face format
        :param pretrained_path: "default" or some model name in hugging_face format
        """
        super().__init__(model_name, range_cand, make_eval, tokenizer_path, pretrained_path)
        self.logger = logging.getLogger(__name__)
        if tokenizer_path == "default":
            tokenizer_path = "alenusch/{}-ruparaphraser".format(
                model_name.replace("-", "")
            )
        if pretrained_path == "default":
            pretrained_path = "alenusch/{}-ruparaphraser".format(
                model_name.replace("-", "")
            )
        self.tokenizer_path = tokenizer_path
        self.pretrained_path = pretrained_path
        self.load()

    def load(self):
        set_seed(42)
        _model = MT5ForConditionalGeneration.from_pretrained(self.pretrained_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _model.to(self.device)
        self.logger.info(
            "Pretrained file and tokenizer for model {} were loaded.".format(
                self.model_name
            )
        )

    def generate(
        self,
        sentence: str,
        n: int = 10,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.95,
        max_length: int = 150,
        repetition_penalty: float = 1.5,
        threshold: float = 0.8,
        strategy: str = "cs"
    ) -> Dict:
        """
        Generate paraphrase. You can set parameters
        :param sentence: str: obligatory one sentence
        :param n: number of sequences to generate
        :param temperature: temperature
        :param top_k: top_k
        :param top_p: top_p
        :param max_length: max_length
        :param repetition_penalty: repetition_penalty
        :param threshold: param for cosine similarity range
        :param strategy: param for range strategy
        :return: dict with fields
        obligatory: origin, predictions;
        optional: warning, best_candidates, average_metrics
        """
        result = {"origin": sentence, "results": []}
        warning, sentences = check_input(sentence)
        if warning:
            result["warning"] = warning

        for sentence in sentences:
            final_outputs = []
            lsentence = "перефразируй: " + sentence + "</s>"
            encoding = self.tokenizer.encode_plus(
                lsentence, pad_to_max_length=True, return_tensors="pt"
            )
            input_ids, attention_masks = (
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
            )

            beam_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                do_sample=True,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                early_stopping=True,
                num_return_sequences=n,
                repetition_penalty=repetition_penalty,
            )
            for beam_output in beam_outputs:
                sent = self.tokenizer.decode(
                    beam_output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                if sent.lower() != sentence.lower() and sent not in final_outputs:
                    final_outputs.append(sent)

            sentence_res = {"predictions": final_outputs}
            best_candidates = []
            if self.range_cand:
                best_candidates = range_candidates(
                    final_outputs, sentence, self.smodel,
                    threshold=threshold, strategy=strategy
                )
                sentence_res["best_candidates"] = best_candidates
            if self.make_eval:
                if not best_candidates:
                    best_candidates = final_outputs
                metrics = get_scores(self.ngeval, best_candidates, sentence)
                sentence_res["average_metrics"] = metrics

            result["results"].append(sentence_res)
        return result