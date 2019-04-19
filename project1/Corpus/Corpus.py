import logging
import sys

from gensim.corpora import WikiCorpus


wiki_corpus = WikiCorpus(sys.argv[1], dictionary={})
texts_num = 0

with open("corpus.txt",'w',encoding='utf-8') as output:
    for text in wiki_corpus.get_texts():
        output.write(' '.join(text) + '\n')
        texts_num += 1
        if texts_num % 10000 == 0:
            logging.info("已處理 %d 篇文章" % texts_num)
