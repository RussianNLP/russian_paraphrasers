# coding=utf-8
from typing import Dict
from russian_paraphrasers.paraphrasers import Paraphraser
from russian_paraphrasers.candidates_filter_metrics import (
    get_scores,
    range_candidates,
    check_input,
)
from russian_paraphrasers.utils import clean
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import logging
import torch


class GPTParaphraser(Paraphraser):
    def __init__(
        self,
        model_name: str = "gpt2",
        range_cand: bool = False,
        make_eval: bool = False,
        tokenizer_path: str = "default",
        pretrained_path: str = "default"
    ):
        """
        The class for GPT2 hugging_face interface.
        :param model_name: "gpt3" or "gpt2". Will call models
        :param range_cand: True/False. Range candidates
        :param make_eval: True/False. Make or not average evaluation for n samples.
        :param tokenizer_path: "default" or some model name in hugging_face format
        :param pretrained_path: "default" or some model name in hugging_face format
        """
        super().__init__(model_name, range_cand, make_eval, tokenizer_path, pretrained_path)
        self.logger = logging.getLogger(__name__)
        if tokenizer_path == "default":
            tokenizer_path = "alenusch/ru{}-paraphraser".format(model_name)
        if pretrained_path == "default":
            pretrained_path = "alenusch/ru{}-paraphraser".format(model_name)
        self.tokenizer_path = tokenizer_path
        self.pretrained_path = pretrained_path
        self.batch_size = 1
        self.load()

    def load(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_path)
        model = GPT2LMHeadModel.from_pretrained(self.pretrained_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.logger.info(
            "Pretrained file and tokenizer for model {} were loaded. {}, {}".format(
                self.model_name, self.tokenizer_path, self.pretrained_path
            )
        )

    def generate(
        self,
        sentence: str,
        n: int = 10,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.9,
        max_length: int = 100,
        repetition_penalty: float = 1.5,
        threshold: float = 0.7,
        strategy: str = "cs",
        stop_token: str = "</s>"
    ) -> Dict:
        """
        Generate paraphrase. You can set parameters
        :param sentence: str: obligatory one sentence
        :param n: number of sequences to generate
        :param temperature: temperature
        :param top_k: top_k
        :param top_p: top_p
        :param max_length: max_length (default is -1)
        :param repetition_penalty: repetition_penalty
        :param threshold: param for cosine similarity range
        :param strategy: param for range strategy
        :param stop_token </s> for gpt2s
        :return: dict with fields
        obligatory: origin, predictions;
        optional: warning, best_candidates, average_metrics
        """
        result = {"origin": sentence, "results": []}
        warning, sentences = check_input(sentence)
        if warning:
            result["warning"] = warning

        for sentence in sentences:
            my_sentence = "<s>{} === ".format(sentence)

            encoded_prompt = self.tokenizer.encode(
                my_sentence, add_special_tokens=False, return_tensors="pt"
            )
            encoded_prompt = encoded_prompt.to(self.device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=n,
            )

            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            generated_sequences = []
            for generated_sequence_idx, generated_sequence in enumerate(
                output_sequences
            ):
                generated_sequence = generated_sequence.tolist()
                text = self.tokenizer.decode(
                    generated_sequence, clean_up_tokenization_spaces=True
                )

                text = text[: text.find(stop_token) if stop_token else None]

                text = text.split("</s>")[0].split("\n")[0]
                total_sequence = (
                    text[
                        len(
                            self.tokenizer.decode(
                                encoded_prompt[0], clean_up_tokenization_spaces=True
                            )
                        ):
                    ]
                )
                total_sequence = clean(total_sequence)
                generated_sequences.append(total_sequence)

            sentence_res = {"predictions": generated_sequences}
            best_candidates = []
            if self.range_cand:
                best_candidates = range_candidates(
                    generated_sequences, sentence, self.smodel,
                    threshold=threshold, strategy=strategy)
                sentence_res["best_candidates"] = best_candidates
            if self.make_eval:
                if not best_candidates:
                    best_candidates = generated_sequences
                metrics = get_scores(self.ngeval, best_candidates, sentence)
                sentence_res["average_metrics"] = metrics

            result["results"].append(sentence_res)
        return result
