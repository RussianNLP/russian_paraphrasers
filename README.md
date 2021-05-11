# Russian Paraphrasers

The library for Russian paraphrase generation.
Paraphrase generation is an increasingly popular task in NLP that can be used in many areas:

- style transfer: 
    - translation from rude to polite
    - translation from professional to simple language
- data augmentation: increasing the number of examples for training ML-models
- increasing the stability of ML-models: training models on a wide variety of examples, in different styles, with different sentiment, but the same meaning / intent of the user

## Install

```
pip install --upgrade pip
pip install -r requirements.txt
pip install russian_paraphrasers
```
in this case you need to add `git+https://github.com/Maluuba/nlg-eval.git@master`

Warning important in requirements.txt (versions!):
```
sentence-transformers==0.4.0
transformers>=4.0.1
git+https://github.com/Maluuba/nlg-eval.git@master
```


Or you can install last version from git:
```
pip install git+https://github.com/RussianNLP/russian_paraphrasers@master
```


## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IjBeV--kiBoPQM6bqg9h2cX4Vhf1ofNK?usp=sharing)

1) First, import one of the models and set general parameters for your paraphraser:

```
from russian_paraphrasers import GPTParaphraser

paraphraser = GPTParaphraser(model_name="gpt2", range_cand=False, make_eval=False)
```

```
from russian_paraphrasers import Mt5Paraphraser

paraphraser = Mt5Paraphraser(model_name="mt5-base", range_cand=False, make_eval=False)
```

You can choose 1) to filter candidates or not 2) to add some evaluation of best candidates or all `n` samples.

Arguments:
- model_name: `mt5-small`, `mt5-base`, `mt5-large`, `gpt2`
- range_cand: `True/False`
- make_eval: `True/False`

2) Pass sentence (obligatory) and parameters for generating to generate function and see the results.

```
sentence = "Мама мыла раму."
results = paraphraser.generate(
    sentence, n=10, temperature=1, 
    top_k=10, top_p=0.9, 
    max_length=100, repetition_penalty=1.5,
    threshold=0.7
)
```
You can set the `threshold` parameter to range candidates, 
it is calculated as similarity score between original vector and the candidate vector.


Results for one sentence look like this:

```
{'average_metrics': {'Bleu_1': 0.06666666665333353,
                     'Bleu_2': 2.3570227263379004e-09,
                     'Bleu_3': 8.514692649183842e-12,
                     'Bleu_4': 5.665278056606597e-13,
                     'ROUGE_L': 0.07558859975216851},
 'best_candidats': ['В чём цель существования человека?',
                    'Для чего нужна жизнь?',
                    'Что такое жизнь в смысле смысла ее существования, и зачем '
                    'она нужна человеку.'],
 'predictions': ['В чём счастье людей, проживающих в мире сегодня',
                 'В чём счастье человека?)',
                 'Для чего нужна жизнь и какова цель ее существования?',
                 'Что означает фраза в том чтобы жить жизнью?',
                 'В чём ценность человеческой Жизни?',
                 'В чём счастье людей в мире? и т. д.',
                 'Зачем нужна жизнь и что в ней главное докуменция дл',
                 'В чём цель существования человека?',
                 'Что такое жизнь в смысле смысла ее существования, и зачем '
                 'она нужна человеку.',
                 'Для чего нужна жизнь?']
}
```

## Models

All models were fine-tuned on the same dataset (see below) and uploaded to hugging_face.
Available models:
- `gpt2` = [rugpt2-large](https://huggingface.co/sberbank-ai/rugpt2large) trained by Sberbank team https://github.com/sberbank-ai/ru-gpts
- `gpt3` = [rugpt3large_based_on_gpt2](https://huggingface.co/sberbank-ai/rugpt3large_based_on_gpt2) trained by Sberbank team https://github.com/sberbank-ai/ru-gpts
- [mt5-small](https://huggingface.co/google/mt5-small)
- [mt5-base](https://huggingface.co/google/mt5-base)
- [mt5-large](https://huggingface.co/google/mt5-large)

To be continued... =)

## Dataset

All models were finetuned on the dataset based on two parts:

1) part of the [ParaPhraser data](http://paraphraser.ru/download/), about 200k filtered examples
2) filtered questions to chatbots and filtered subtitles from [here](https://github.com/rysshe/paraphrase/tree/master/data)

The dataset is in the folder `dataset`.

The article is published and was presenteed in BSNLP. Read it [here](http://bsnlp.cs.helsinki.fi/papers-2021/2021.bsnlp-1.2.pdf). 
