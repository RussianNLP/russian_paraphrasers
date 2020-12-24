from russian_paraphrasers import GPTParaphraser
from russian_paraphrasers import Mt5Paraphraser
import time
import pprint


sentence_examples = [
        "Коронавирус уничтожил экономику РФ.",
        "Как можно совместить отдых и домашние дела? А то у меня что-то не получается отдохнуть.",
        "В чем смысл жизни?",
        "Когда же этот год закончится?"
]

def run_gpt2_example():
    """EXAMPLE FOR GPT2-large paraphraser"""
    paraphraser = GPTParaphraser(model_name="gpt2", range_cand=True, make_eval=False)
    for sentence_example in sentence_examples:
        start = time.time()
        results = paraphraser.generate(sentence_example)
        print("time:", time.time() - start)
        pprint.pprint(results)


def run_mt5_example():
    """EXAMPLE FOR Mt5 paraphrasers"""
    paraphraser = Mt5Paraphraser(model_name="mt5-large", range_cand=True, make_eval=True)
    for sentence_example in sentence_examples:
        start = time.time()
        results = paraphraser.generate(sentence_example, top_k=20, max_length=100)
        print("time:", time.time() - start)
        pprint.pprint(results)


run_mt5_example()
# run_gpt2_example()
