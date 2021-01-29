from difflib import SequenceMatcher
from sentence_transformers import util
from collections import defaultdict
from scipy.spatial import distance
import logging
import nltk

logger = logging.getLogger(__name__)


def range_by_allcs(sentences, sent, smodel, threshold=0.7):
    sentences = sentences + [sent]
    hypothesis = []
    paraphrases = util.paraphrase_mining(smodel, sentences)
    good_hyp = set()
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        if threshold < score < 1.00:
            if sentences[i] == sent:
                good_hyp.add(sentences[j])
            elif sentences[j] == sent:
                good_hyp.add(sentences[i])
    if len(list(good_hyp)) > 1:
        hypothesis.extend(list(good_hyp))
    else:
        if paraphrases[0][1] != sent:
            hypothesis.append(sentences[paraphrases[0][1]])
        else:
            hypothesis.append(sentences[paraphrases[0][2]])
    return hypothesis


def range_by_cs(sentences, sent, smodel, threshold=0.9):
    try:
        sentence_embeddings = smodel.encode(sentences)
        origin_emb = smodel.encode([sent])
        best_cands = []
        for sentence, embedding in zip(sentences, sentence_embeddings):
            if sent not in sentence:
                if SequenceMatcher(None, sent, sentence).ratio() < 0.95:
                    score = 1 - distance.cosine(embedding, origin_emb)
                    if score >= threshold:
                        if score != 1.0:
                            if [score, sentence] not in best_cands:
                                best_cands.append([score, sentence])
        hypothesis = sorted(best_cands)
        hypothesis = list([val for [_, val] in hypothesis])
    except Exception as e:
        logger.warning("Can't measure embeddings scores. Error: " + str(e))
        cands = []
        for sentence in sentences:
            if sent not in sentence:
                if SequenceMatcher(None, sent, sentence).ratio() < 0.95:
                    cands.append(sentence)
        hypothesis = list(set(cands))
    return hypothesis


def range_candidates(sentences, sent, smodel, threshold=0.9, strategy="cs"):
    """
    Range all possible candidates by one of the strategies
    :param sentences: candidates
    :param sent: origin sentence reference
    :param smodel: sentence transformer model
    :param threshold: threshold for cosine similarity score
    :param strategy: best by cosine similarity between sentence origin and generated  - flag "cs",
    best by cosine similary between all generated pairs - flag "all_cs"
    :return: list: best candidates
    """
    sentences = list(set(sentences))
    if strategy == "cs":
        hypothesis = range_by_cs(sentences, sent, smodel, threshold=threshold)
    else:
        hypothesis = range_by_allcs(sentences, sent, smodel, threshold=threshold)
    return hypothesis


def get_scores(ngeval, best_candidates, sentence):
    """
    Average metrics of candidates
    :param ngeval:
    :param best_candidates:
    :param sentence:
    :return: metrics
    """
    average_metrics = defaultdict(list)
    if best_candidates:
        for hyp in best_candidates:
            metrics_dict = ngeval.compute_individual_metrics([sentence.lower()], hyp.lower())
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
    sentences = nltk.sent_tokenize(sentence)
    if len(sentences) > 1:
        warning = "There are more than one sentence! We split it and paraphrase separately."
    return warning, sentences
