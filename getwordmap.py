import os
import sys

VOCABULARY_FILEPATH = './nn_word_map'
VOCAB_SPLIT_FLAG = ','
RESULT_DIR = '../word_map'
splitVocab = lambda l: [word.strip().split(VOCAB_SPLIT_FLAG) for word in l if word.strip()]


def getVacabularyTable(vocabulary_filepath):
    assert os.path.exists(vocabulary_filepath) == True, "Can't open %s" % (vocabulary_filepath)
    with open(vocabulary_filepath, 'r') as f:
        vocabulary_list = splitVocab(f.readlines())
    word_map_list = list()
    word_map_list.append('<UNK>,0')
    word2id = [str(item[0]) + ',' + str(i + 1) for i, item in enumerate(vocabulary_list)]
    word_map_list.extend(word2id)
    # add unknown index
    print('Vocabulary list size : %d' % len(word_map_list))
    word_map_str = '\n'.join(word_map_list)
    return word_map_str


def getWordMap(result_dir, vocabulary_filepath):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    word_map_str = getVacabularyTable(vocabulary_filepath)
    result_filepath = os.path.join(result_dir, 'nn_word_map.txt')
    with open(result_filepath, 'w+') as f:
        f.write(word_map_str)


if __name__ == '__main__':
    getWordMap(RESULT_DIR, VOCABULARY_FILEPATH)
