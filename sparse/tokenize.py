import sys
import csv
import math
from multiprocessing import Pool
from fog.tokenizers import WordTokenizer
from sparse.stop_words import STOP_WORDS_FR
from tqdm import tqdm
from collections import Counter

tokenizer = WordTokenizer(
    keep=['word', 'mention'],
    lower=True,
    unidecode=True,
    split_hashtags=True,
    stoplist=STOP_WORDS_FR + [t + "'" for t in STOP_WORDS_FR] + [t + "’" for t in STOP_WORDS_FR],
    reduce_words=True,
    decode_html_entities=True
)

DOCUMENTS = []
DOCUMENT_FREQUENCIES = Counter()

def tokenize(row):
    return set(value for _, value in tokenizer(row[3]))

with open(sys.argv[1]) as f:
    reader = csv.reader(f)
    next(reader)

    loading_bar = tqdm(unit='tweet', total=7000000)

    with Pool(8) as pool:
        for tokens in pool.imap(tokenize, reader):
            loading_bar.update()

            for token in tokens:
                DOCUMENT_FREQUENCIES[token] += 1

            DOCUMENTS.append(tokens)

loading_bar.close()
print('Size of vocabulary:', len(DOCUMENT_FREQUENCIES))

N = len(DOCUMENTS)
ID = 0
TOKEN_IDS = {}
INVERSE_DOCUMENT_FREQUENCIES = Counter()

for token, df in DOCUMENT_FREQUENCIES.items():
    if df > 10:
        TOKEN_IDS[token] = ID
        ID += 1
        INVERSE_DOCUMENT_FREQUENCIES[token] = 1 + math.log((N + 1) / (df + 1))

print('Size of vocabulary after df trimming:', len(INVERSE_DOCUMENT_FREQUENCIES))

with open('data/vectors.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['dimensions', 'weights'])

    for doc in tqdm(DOCUMENTS):
        vector = [
            (TOKEN_IDS[token], INVERSE_DOCUMENT_FREQUENCIES[token])
            for token in doc
            if token in TOKEN_IDS
        ]

        vector = sorted(vector, key=lambda t: (t[1], t[0]), reverse=True)

        writer.writerow([
            '|'.join(str(dim) for dim, _ in vector),
            '|'.join(str(weight) for _, weight in vector)
        ])
