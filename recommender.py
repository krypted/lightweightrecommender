import gensim
from nltk.tokenize import word_tokenize
from csv import reader
import argparse

parser = argparse.ArgumentParser(prog='Tagger')
parser.add_argument('--file', metavar='N', type=str, nargs='?', help='File')
parser.add_argument('--column', metavar='N', type=str, nargs='?', help='Column name')
parser.add_argument('--text', metavar='N', type=str, nargs='?', help='Query/text to compare')
parser.add_argument('--recs', metavar='N', type=int, nargs='?', help='number of limits to return')

args = parser.parse_args()
file = args.file
column = args.column
text = args.text
limit = args.recs
if not file or not column or not text or not limit:
    raise Exception("--file, --column, --text and --recs options must be set")


def check_similarity(doc, raw_documents, limit):
    gen_docs = [[w.lower() for w in word_tokenize(text)]
                for text in raw_documents]
    dictionary = gensim.corpora.Dictionary(gen_docs)

    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

    tf_idf = gensim.models.TfidfModel(corpus)

    sims = gensim.similarities.Similarity('', tf_idf[corpus],
                                          num_features=len(dictionary))

    query_doc = [w.lower() for w in word_tokenize(doc)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]

    match = {}
    for i, s in enumerate(sims[query_doc_tf_idf]):
        match.update({
            raw_documents[i]: s
        })
    return {k: v for k, v in sorted(match.items(), key=lambda item: item[1], reverse=True)[:limit]}


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# parse csv and return column required
def parse_csv(filename, column_name):
    data = load_csv(filename)
    header = data[0]
    column = False
    for i, h in enumerate(header):
        if h == column_name:
            column = i

    if not column:
        return False

    data_set = []
    for i in range(1, len(data) - 1):
        data_set.append(data[i][column])
    return data_set


data = parse_csv(file, column)
if not data:
    raise Exception("Column with that name not found")

sim = check_similarity(text, data, limit)

for s in sim.keys():
    print(s)
