from difflib import SequenceMatcher
from sentence_transformers import util
from _collections import defaultdict
from rusenttokenize import ru_sent_tokenize
import logging

logger = logging.getLogger(__name__)


def range_candidates(sentences, sent, smodel, top_k=3):
    """
    Range all possible candidates
    :param sentences: candidates
    :param sent: origin sentence reference
    :param smodel: sentence transformer model
    :param top_k: 3
    :return: list: top k best candidates
    """
    sentences = list(set(sentences))
    candidates = []
    for generated_sent in sentences:
        if sent not in generated_sent:
                if SequenceMatcher(None, sent, generated_sent).ratio() < 0.95:
                    candidates.append(generated_sent)

    good_hyp = set()
    if candidates:
        try:
            paraphrases = util.paraphrase_mining(smodel, candidates)
            for paraphrase in paraphrases:
                score, i, j = paraphrase
                if 0.75 < score < 1.00:
                    good_hyp.add(sentences[j])
                elif sent in sentences[j]:
                    good_hyp.add(sentences[i])
        except Exception as e:
            logger.warning("Error: " + str(e))
    if good_hyp:
        hypothesis = list(good_hyp)[:top_k]
    else:
        hypothesis = candidates[:top_k]
    return hypothesis


def get_scores(ngeval, best_candidates, sentence):
    """

    :param ngeval:
    :param best_candidates:
    :param sentence:
    :return:
    """
    average_metrics = defaultdict(list)
    if best_candidates:
        for hyp in best_candidates:
            metrics_dict = ngeval.compute_individual_metrics([sentence], hyp)
            for key, value in metrics_dict.items():
                average_metrics[key].append(value)
    metrics = {}
    for key, value in average_metrics.items():
        metrics[key] = sum(value)/len(value)
    return metrics


def check_input(sentence):
    warning = None
    if len(sentence) <= 7:
        warning = "Your sentence is too short. The results can be strange."
    sentences = ru_sent_tokenize(sentence)
    if len(sentences) > 1:
        warning = "There are more than one sentence! We split it and paraphrase separately."
    return warning, sentences
