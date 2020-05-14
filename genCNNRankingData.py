# encoding=utf-8
import numpy as np
import os
from tqdm import tqdm
import random
import sys

RECORD_FLAG = '[dat]'
TRAIN_DATA_RESULT_DIR = '../train_med_CNN_splitword_ranking/'
TEST_DATA_RESULT_DIR = '../test_med_CNN_splitword_ranking/'
INPUT_FILEPATH = './med_split_cnn.txt'
VOCABULARY_FILEPATH = './nn_word_map'
ITEM_SPLIT_FLAG = '\t\t'
QUERY_SPLIT_FLAG = ' '
VOCAB_SPLIT_FLAG = ','
RECORD_SPLIT_FLAG = ','
MAX_RECORDS_PER_FILE = 2000
SAVE_FILEPREFIX = 'part-'
TRAIN_DATA_RATIO = 0.9
splitRecord = lambda l: [query.strip().split(ITEM_SPLIT_FLAG) for query in l if query.strip()]
flatList = lambda l: [item.strip() for item in l if item.strip()]
splitVocab = lambda l: [word.strip().split(VOCAB_SPLIT_FLAG) for word in l if word.strip()]
"""
----------------------------------
query | split_qurey | label |
xxxxx |  xxxxxxxx   |  xxxx |  
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


def get_train_and_test_dataset(input_filepath):
    assert os.path.exists(input_filepath) == True, "Can't open file %s" % (input_filepath)
    with open(input_filepath, 'r', encoding='utf-8') as f:
        print('reading %s content' % input_filepath)
        total_records = f.readlines()
        print('total len ', len(total_records))
        print('start Delete duplicate record')
        total_records_set = set(total_records)
        total_records = list(total_records_set)
        print('Complete Delete duplicate record')
        total_records = flatList(total_records)
        print('start shuffle ...............')
        random.shuffle(total_records)
        random.shuffle(total_records)
        random.shuffle(total_records)
        print('Complete shuffle ...............')
    num_train_examples = int(len(total_records) * TRAIN_DATA_RATIO)
    train_dataset = total_records[:num_train_examples]
    test_dataset = total_records[num_train_examples:]
    print('Train dataset size : %d' % len(train_dataset))
    print('Test dataset size : %d' % len(test_dataset))
    return train_dataset, test_dataset


def getVacabularyTable(vocabulary_filepath):
    assert os.path.exists(vocabulary_filepath) == True, "Can't open %s" % (vocabulary_filepath)
    with open(vocabulary_filepath, 'r') as f:
        vocabulary_list = splitVocab(f.readlines())
    word2id = {item[0]: item[1] for item in vocabulary_list}
    # add unknown index
    word2id['<UNK>'] = '0'
    print('Vocabulary list size : %d' % len(word2id))
    return word2id


def generateStanderKeyValue(result_dir, input_dataset, vocabulary_filepath):
    """
    Parameters
    ----------
    result_dir : str
        Directory to save the result data.
    input_dataset : list
        Input data.
    vocabulary_filepath : str
        Vocabulary filepath.
    """

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    word2id = getVacabularyTable(vocabulary_filepath)
    convert2id = lambda query: str(word2id[query]) if word2id.get(query) is not None else str(word2id['<UNK>'])
    assert isinstance(word2id, dict), 'Vocabulary table must be a dict'

    print("Start splitRecord .................")
    total_records = splitRecord(input_dataset)
    print('Complete splitRecord ..............')
    num_records = len(total_records)
    print('Total %d records' % (num_records))
    # compute number files to save data
    num_files = num_records // MAX_RECORDS_PER_FILE + 1
    print('Create %s [%04d - %04d] to save result' % ('part', 0, num_files - 1))
    per_record_min_word_len = 50
    per_record_max_word_len = 0
    total_word_count = 0

    # process every record and write to corresponding file
    for i, record in tqdm(enumerate(total_records)):
        """ split query | label """
        # parse each item in a record
        if len(record) < 3:
            print('Find a broken data,this will be ignored ....')
            print(record)
            continue
        query = record[0]
        split_query = record[1]
        label = record[2]
        if label not in ('1', '4', '7'):
            print('Find a broken data,label not in (1,4,7)')
            print(record)
            continue
        # convert split query to id
        split_query = flatList(split_query.split(QUERY_SPLIT_FLAG))
        ####################################

        per_record_min_word_len = min(len(split_query), per_record_min_word_len)
        per_record_max_word_len = max(len(split_query), per_record_max_word_len)
        total_word_count += len(split_query)

        #####################################
        split_query_id = [convert2id(word) for word in split_query]
        # convert to str
        split_query_id = RECORD_SPLIT_FLAG.join(split_query_id)
        record_list = list()
        keys = ['query', 'input_word', 'label']
        values = [query, split_query_id, label]
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
    print('Complete all Data ............')
    print('Word max len : %d' % per_record_max_word_len)
    print('Word min len : %d' % per_record_min_word_len)
    print('Word avg len : %d' % (total_word_count / num_records))


if __name__ == '__main__':
    train_dataset, test_dataset = get_train_and_test_dataset(INPUT_FILEPATH)
    print("Processing training data ...................")
    generateStanderKeyValue(TRAIN_DATA_RESULT_DIR, train_dataset, VOCABULARY_FILEPATH)
    print("Processing test data ...................")
    generateStanderKeyValue(TEST_DATA_RESULT_DIR, test_dataset, VOCABULARY_FILEPATH)
