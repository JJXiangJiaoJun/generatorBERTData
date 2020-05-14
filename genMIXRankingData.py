# encoding=utf-8
import numpy as np
import os
from tqdm import tqdm
import random
import sys
import shutil

RECORD_FLAG = '[dat]'
RESULT_DIR = '/apsarapangu/disk1/develop/zqh/MIX_medRanking/'
INPUT_FILEPATH = './med_bert_score.txt'
INPUT_FILEPATH_LIST = ['/apsarapangu/disk1/develop/med_bert_mix_tk_pos','/apsarapangu/disk1/develop/med_bert_mix_tk_neg']
VOCABULARY_FILEPATH = './nn_word_map'
ITEM_SPLIT_FLAG = '^------^'
QUERY_SPLIT_FLAG = ' '
VOCAB_SPLIT_FLAG = ','
RECORD_SPLIT_FLAG = ','
MAX_RECORDS_PER_FILE = 200000
SAVE_FILEPREFIX = 'part-'
TRANING_DATA_RATIO = 0.8
ITEM_NUMS = 8
splitRecord = lambda l: [query.strip().split(ITEM_SPLIT_FLAG) for query in l if query.strip()]
flatList = lambda l: [item.strip() for item in l if item.strip()]
splitVocab = lambda l: [word.strip().split(VOCAB_SPLIT_FLAG) for word in l if word.strip()]
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


def get_train_and_test_dataset(input_filepath_list):
    total_records = list()
    for input_filepath in input_filepath_list:
        assert os.path.exists(input_filepath) == True, "Can't open file %s" % (input_filepath)
        with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            print('reading %s content' % input_filepath)
            total_records.extend(f.readlines())
    print('total len ', len(total_records))
    print('start Delete duplicate record')
    total_records_set = set(total_records)
    total_records = list(total_records_set)
    print('Complete Delete duplicate record')
    total_records = flatList(total_records)
    print('start shuffle ...............')
    random.shuffle(total_records)
    print('Complete shuffle ...............')
    num_train_examples = int(len(total_records) * TRANING_DATA_RATIO)
    train_dataset = total_records[:num_train_examples]
    test_dataset = total_records[num_train_examples:]
    print('Train dataset size : %d' % len(train_dataset))
    print('Test dataset size : %d' % len(test_dataset))
    return train_dataset, test_dataset


def getBertCharVocabTable(vocabulary_filepath):
    assert os.path.exists(vocabulary_filepath), "Can't open %s " % vocabulary_filepath
    with open(vocabulary_filepath, 'r') as f:
        charVocab = f.readlines()
        # charVocab = flatList(charVocab)
    char2id = {v.rstrip(): k for k, v in enumerate(charVocab)}
    print('Vocabulary list size : %d' % len(char2id))
    return char2id


def getVacabularyTable(vocabulary_filepath):
    assert os.path.exists(vocabulary_filepath) == True, "Can't open %s" % (vocabulary_filepath)
    with open(vocabulary_filepath, 'r') as f:
        vocabulary_list = splitVocab(f.readlines())
    word2id = {item[0]: item[1] for item in vocabulary_list}
    # add unknown index
    word2id['<UNK>'] = '0'
    print('Vocabulary list size : %d' % len(word2id))
    return word2id


def generateStanderKeyValue(result_dir, input_data, vocabulary_filepath, is_training=True):
    """
    Parameters
    ----------
    result_dir : str
        Directory to save the result data.
    input_data : list
        Input data for record.
    vocabulary_filepath : str
        Vocabulary file path.
    """
    if is_training:
        result_dir = os.path.join(result_dir, 'train')
    else:
        result_dir = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        shutil.rmtree(result_dir)
        os.makedirs(result_dir)
    word2id = getVacabularyTable(vocabulary_filepath)
    assert isinstance(word2id, dict), 'Vocabulary table must be a dict'
    convert2id = lambda word: str(word2id[word]) if word2id.get(word) is not None else str(word2id['<UNK>'])

    print('Start splitRecord')
    total_records = splitRecord(input_data)
    print('Complete splitRecord .............')
    num_records = len(total_records)
    print('Total %d records' % num_records)
    # compute number files to save data
    num_files = num_records // MAX_RECORDS_PER_FILE + 1
    print('Create %s [%04d - %04d] to save result' % ('part', 0, num_files - 1))

    # process every record and write to corresponding file
    for i, record in tqdm(enumerate(total_records)):
        """ query | query_split | title1 | title1_split | title2 | title2_split | title1_score | title2_score """
        # parse each item in a record
        # bert input need 1. input_ids(char) 2.segment_ids(0) 3.input_masks 4.label
        if len(record) < ITEM_NUMS:
            print('Find a broken data,this will be ignored ....')
            print(record)
            continue
        query = record[0]
        query_split = record[1]
        title1 = record[2]
        title1_split = record[3]
        title2 = record[4]
        title2_split = record[5]
        title1_score = record[6]
        title2_score = record[7]
        label = '1'

        try:
            score_1 = float(title1_score)
            score_2 = float(title2_score)
            if score_1 > score_2:
                label = '1'
            else:
                label = '0'
        except:
            print('score must be float type but got  title1_score %s  title2_score %s' % (title1_score, title2_score))
            continue

        query_split = query_split.split(QUERY_SPLIT_FLAG)
        query_ids = [convert2id(word.strip()) for word in query_split if word.strip()]
        query_ids_join = RECORD_SPLIT_FLAG.join(query_ids)
        title1_split = title1_split.split(QUERY_SPLIT_FLAG)
        title1_ids = [convert2id(word.strip()) for word in title1_split if word.strip()]
        title1_ids_join = RECORD_SPLIT_FLAG.join(title1_ids)

        title2_split = title2_split.split(QUERY_SPLIT_FLAG)
        title2_ids = [convert2id(word.strip()) for word in title2_split if word.strip()]
        title2_ids_join = RECORD_SPLIT_FLAG.join(title2_ids)

        try:
            float(label)
        except:
            print('broken label %s' % label)
            continue
        keys = ['input_query', 'input_query_ids', 'input_title1', 'input_title1_ids',
                'input_title2', 'input_title2_ids','title1_score','title2_score','label']
        values = [query, query_ids_join, title1, title1_ids_join,
                  title2,title2_ids_join,title1_score,title2_score,label]
        record_list = list()
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
    print("Complete all Data ......")


if __name__ == '__main__':
    train_dataset, test_dataset = get_train_and_test_dataset(INPUT_FILEPATH_LIST)
    print('Start parse training dataset ..............')
    generateStanderKeyValue(RESULT_DIR, train_dataset, VOCABULARY_FILEPATH)
    print('Complete parse training dataset.............')
    print('Start parse testing dataset ................')
    generateStanderKeyValue(RESULT_DIR, test_dataset, VOCABULARY_FILEPATH, is_training=False)
    print('Complete parse testing dataset.............')
