# encoding=utf-8
import numpy as np
import os
from tqdm import tqdm
import random
import sys

RECORD_FLAG = '[dat]'
RESULT_DIR = '../eval_splitword/'
INPUT_FILEPATH = './tmp_timeliness_3k_query_segment'
VOCABULARY_FILEPATH = './nn_word_map'
WORD2LABEL_FILEPATH = './query_timeliness_3k_review.CSV'
ITEM_SPLIT_FLAG = '\t\t'
WORD2LABEL_SPLIT_FLAG = ','
QUERY_SPLIT_FLAG = ' '
VOCAB_SPLIT_FLAG = ','
RECORD_SPLIT_FLAG = ','
MAX_RECORDS_PER_FILE = 200000
SAVE_FILEPREFIX = 'part-'
splitRecord = lambda l: [query.strip().split(ITEM_SPLIT_FLAG) for query in l if query.strip()]
flatList = lambda l: [item.strip() for item in l if item.strip()]
splitVocab = lambda l: [word.strip().split(VOCAB_SPLIT_FLAG) for word in l if word.strip()]
splitWord2Label = lambda l: [word.strip().split(WORD2LABEL_SPLIT_FLAG) for word in l if word.strip()]
"""
----------------------------------
query | split_qurey | label | score|
xxxxx |  xxxxxxxx   |  xxxx | xxxxx |  
-----------------------------------
standerKeyValue

[dat]                             // record_flag，标识一条record的起始，标识符后必须换行！
label=0:0                         // "="前是数据列名（key），":"前是value长度， ":"后是value
user=0:jack                       // 单行的value，长度配置0即可。value以换行符为结束（不包含换行符）
item=0:14520275084855801810
top_category=0:01001,01002,01003  // 多值分隔符","在数据输入的schema中声明
clk=4:0.06                        // 指定了value长度，因此后4位 0.06 是value；允许下个key是换行开始的(不换行也可)
query=0:3,20,
weight=0:0.5,-1.0
"""


def getVacabularyTable(vocabulary_filepath):
    assert os.path.exists(vocabulary_filepath) == True, "Can't open %s" % (vocabulary_filepath)
    with open(vocabulary_filepath, 'r') as f:
        vocabulary_list = splitVocab(f.readlines())
    word2id = {item[0]: item[1] for item in vocabulary_list}
    # add unknown index
    word2id['<UNK>'] = '0'
    print('Vocabulary list size : %d' % len(word2id))
    return word2id


def getWord2Label(word2label_filepath):
    assert os.path.exists(word2label_filepath) == True, "Can't open %s" % (word2label_filepath)
    with open(word2label_filepath, 'r') as f:
        word2label_list = splitWord2Label(f.readlines())
    word2label = {item[0]: item[1] for item in word2label_list}
    word2timeliness = {item[0]: item[2] for item in word2label_list}
    print('Word2Label size : %d' % len(word2label))
    return word2label, word2timeliness


def generateStanderKeyValue(result_dir, input_filepath, vocabulary_filepath, word2label_filepath):
    """
    Parameters
    ----------
    result_dir : str
        Directory to save the result data.
    input_filepath : str
        Input filepath.
    """
    assert os.path.exists(input_filepath) == True, "Can't open file %s" % (input_filepath)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    word2id = getVacabularyTable(vocabulary_filepath)
    word2label, word2timeliness = getWord2Label(word2label_filepath)
    convert2id = lambda query: str(word2id[query]) if word2id.get(query) is not None else str(word2id['<UNK>'])
    assert isinstance(word2id, dict), 'Vocabulary table must be a dict'
    with open(input_filepath, 'r', encoding='utf-8') as f:
        print('reading %s content' % input_filepath)
        total_records = f.readlines()
        print('total len ', len(total_records))
        total_records = splitRecord(total_records)
    num_records = len(total_records)
    print('Total %d records' % (num_records))
    # compute number files to save data
    num_files = num_records // MAX_RECORDS_PER_FILE + 1
    print('Create %s [%04d - %04d] to save result' % ('part', 0, num_files - 1))

    # process every record and write to corresponding file
    for i, record in tqdm(enumerate(total_records)):
        """ query | split query | label | score"""
        # parse each item in a record
        query = record[0]
        split_query = record[1]
        label = word2label.get(query)
        timeliness = word2timeliness.get(query)
        if label is None or label not in ('0', '1') or timeliness is None or timeliness not in (
                '0', '1', '2'):
            continue
        # score = record[3]
        # convert split query to id
        split_query = flatList(split_query.split(QUERY_SPLIT_FLAG))
        split_query_id = [convert2id(word) for word in split_query]
        # convert to str
        split_query_id = RECORD_SPLIT_FLAG.join(split_query_id)
        record_list = list()
        keys = ['query', 'input_word', 'label', 'timeliness']
        values = [query, split_query_id, label, timeliness]
        record_list.append(RECORD_FLAG)
        for key, value in zip(keys, values):
            item = "%s=%s:%s" % (key, 0, value)
            record_list.append(item)
        per_result = '\n'.join(record_list) + '\n'

        # write to file
        save_file_suffix = i // MAX_RECORDS_PER_FILE
        save_filename = '%s%04d' % (SAVE_FILEPREFIX, save_file_suffix)
        save_filepath = os.path.join(result_dir, save_filename)
        with open(save_filepath, 'a+', encoding='utf-8') as f:
            f.write(per_result)


if __name__ == '__main__':
    generateStanderKeyValue(RESULT_DIR, INPUT_FILEPATH, VOCABULARY_FILEPATH, WORD2LABEL_FILEPATH)
