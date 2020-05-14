# encoding=utf-8
import numpy as np
import os
from tqdm import tqdm
import random
import sys
import shutil

BERT_CLS = '101'
BERT_SEP = '102'
BERT_INPUT_WORD_LEN = 64
BERT_INPUT_WORD_LEN_EXCLUDE_CLS_SEP = BERT_INPUT_WORD_LEN - 2
RECORD_FLAG = '[dat]'
RESULT_DIR = '../finalTest5WData/'
INPUT_FILEPATH = './test_final_5w_tab.tsv'
VOCABULARY_FILEPATH = './vocab.txt'
ITEM_SPLIT_FLAG = '\t'
QUERY_SPLIT_FLAG = ' '
VOCAB_SPLIT_FLAG = ','
RECORD_SPLIT_FLAG = ','
MAX_RECORDS_PER_FILE = 200000
SAVE_FILEPREFIX = 'part-'
TRANING_DATA_RATIO = 1
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

category2id = \
    {
        'diabetes': '0', 'hypertension': '1', 'hepatitis': '2', 'aids': '3', 'breast_cancer': '4'
    }

categorylist = ['diabetes', 'hypertension', 'hepatitis', 'aids', 'breast_cancer']


def get_train_and_test_dataset(input_filepath):
    assert os.path.exists(input_filepath) == True, "Can't open file %s" % (input_filepath)
    with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        print('reading %s content' % input_filepath)
        total_records = f.readlines()
        total_records = total_records[1:]
        print('total len ', len(total_records))
        total_records = flatList(total_records)
    num_train_examples = int(len(total_records) * TRANING_DATA_RATIO)
    train_dataset = total_records[:num_train_examples]
    test_dataset = total_records[num_train_examples:]
    print('Train dataset size : %d' % len(train_dataset))
    print('Test dataset size : %d' % len(test_dataset))
    return train_dataset, test_dataset


def parse_input(question1_ids, question2_ids):
    """this function will do following steps
    1. truncate input_ids to max_length
    2. add [CLS] & [SEP] flag to input_ids
    3. generate segment_ids and input_masks
    4.
    """
    input_ids = list()
    input_ids.append(BERT_CLS)
    input_ids.extend(question1_ids)
    input_ids.append(BERT_SEP)
    input_ids.extend(question2_ids)
    input_ids_truncated = input_ids[:BERT_INPUT_WORD_LEN]
    # print(input_ids_truncated)
    assert len(input_ids_truncated) <= BERT_INPUT_WORD_LEN, 'input_ids len can not exceed %d' % BERT_INPUT_WORD_LEN
    # print('input_ids_truncated_len ', len(input_ids_truncated))
    segment_ids = list()
    segment_question1_ids = ['0'] * (len(question1_ids) + 2)
    segment_qustion2_ids = ['1'] * (len(input_ids_truncated) - len(question1_ids) - 2)
    segment_ids.extend(segment_question1_ids)
    segment_ids.extend(segment_qustion2_ids)
    input_masks = ['1'] * len(input_ids_truncated)
    input_ids_parsed = RECORD_SPLIT_FLAG.join(input_ids_truncated)
    segment_ids_str = RECORD_SPLIT_FLAG.join(segment_ids)
    input_masks_str = RECORD_SPLIT_FLAG.join(input_masks)
    # print('segmend_ids ', segment_ids_str)
    # print('input_masks ', input_masks_str)
    return input_ids_parsed, segment_ids_str, input_masks_str


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


def parse_answer(answer):
    answer_split = flatList(answer.strip().split(' '))
    answer = ''.join(answer_split)
    return answer


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
    char2id = getBertCharVocabTable(vocabulary_filepath)
    convert2id = lambda query: str(char2id[query]) if char2id.get(query) is not None else str(char2id["[UNK]"])
    # assert isinstance(word2id, dict), 'Vocabulary table must be a dict'
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
        """ id |question1 | question2  | category """
        # parse each item in a record
        # bert input need 1. input_ids(char) 2.segment_ids(0) 3.input_masks 4.label
        if len(record) != 4:
            print('Find a broken data,this will be ignored ....')
            print(record)
            continue
        id = record[1]
        question1 = record[2]
        question2 = record[3]
        category = record[3]

        if record[0] not in categorylist:
            print('Find a broken data , category not in (0 ,1),')
            print(record)
            raise ValueError
        category = category2id[record[0]]

        question1_ids = [convert2id(ch) for ch in list(question1.strip())]
        question2_ids = [convert2id(ch) for ch in list(question2.strip())]

        # parse input_ids
        input_ids_parse, segment_ids, input_masks = parse_input(question1_ids, question2_ids)
        # convert to str
        record_list = list()
        keys = ['id', 'question1', 'question2', 'input_ids', 'segment_ids',
                'input_masks', 'category', 'category_ids']
        values = [id, question1, question2, input_ids_parse, segment_ids,
                  input_masks, record[3], category]
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
    train_dataset, test_dataset = get_train_and_test_dataset(INPUT_FILEPATH)
    print('Start parse training dataset ..............')
    generateStanderKeyValue(RESULT_DIR, train_dataset, VOCABULARY_FILEPATH)
    print('Complete parse training dataset.............')
