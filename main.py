import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import time
from loader import make_gazette_to_dic, prepare_sentence

from utils import evaluate_lexicon_tagger
from loader import load_sentences
from model import Model
from konlpy.tag import Kkma, Mecab
from konlpy.utils import pprint
import re
import optparse

from operator import itemgetter

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", default="", help="Input file location")
optparser.add_option("-o", "--output", default="", help="Output file location")
optparser.add_option("-m", "--model", default="./models/temp", help="Model file location")
optparser.add_option("-d", "--dictionary", default="./data/gazette", help="Dictionary file location")
optparser.add_option("-p", "--preprocess", default="0", help="Check input file format"
                                                            "PoS tagged text(=1) or raw text(=0)")

opts = optparser.parse_args()[0]
print 'input file path : ', opts.input
print 'output file path : ', opts.output

input_filename = opts.input
output_filename = opts.output
model_path = opts.model
dict_path = opts.dictionary
is_preprocess = opts.preprocess

assert os.path.isfile(opts.input)
assert len(opts.preprocess) > 0


def split_sentence(input_filename):
    """
    Split raw document into sentences
    :return: list of sentences
    """
    sentences = []
    with open(input_filename, 'rb') as f:
        for line in f.readlines():
            line = line.strip()
            if line != '':
                line = re.split(r' *[\.\?!][\'"\)\]]* *', line)
                for l in line:
                    if len(l) > 0:
                        sentences.append(l)
    return sentences


def transform_pos(value, tagger='kkma'):
    """
    Transformation rule from KKMA to Sejong pos tag
    :param value: predicted pos tag (KKMA)
    :return: transformed pos tag (Sejong)
    """
    if tagger == 'kkma':
        if value == 'NNM':
            return 'NNB'
        elif value == 'VXV' or value == 'VXA':
            return 'VX'
        elif value == 'MDT' or value == 'MDN':
            return 'MM'
        elif value == 'MAC':
            return 'MAJ'
        elif value == 'JKM':
            return 'JKB'
        elif value == 'JKI':
            return 'JKV'
        elif value == 'EPH' or value == 'EPT' or value == 'EPP':
            return 'EP'
        elif value == 'EFN' or value == 'EFQ' or value == 'EFO' or value == 'EFA' or value == 'EFI' or value == 'EFR':
            return 'EF'
        elif value == 'ECE' or value == 'ECD' or value == 'ECS':
            return 'EC'
        elif value == 'ETD':
            return 'ETM'
        elif value == 'UN':
            return 'NF'
        elif value == 'UV':
            return 'NV'
        elif value == 'UE':
            return 'NA'
        elif value == 'OL':
            return 'SL'
        elif value == 'OH':
            return 'SH'
        elif value == 'ON':
            return 'SN'
        elif value == 'XPV':
            return 'XPN'
        else:
            return value
    elif tagger == 'mecab':
        if value == 'NNBC':
            return 'NNB'
        elif value == 'SSO' or value == 'SSC':
            return 'SS'
        elif value == 'SC':
            return 'SP'
        # elif value == 'SY':
        else:
            return value
          

def tag_pos(sentences, tagger='kkma'):
    """
    Predict Part-of-Speech tag of input sentences
    PoS tagger: KKMA
    :param sentences: list of input sentences
    :return: tagged sentences
    """
    if tagger == 'kkma':
        kkma = Kkma()
    elif tagger == 'mecab':
        mecab = Mecab()
                
    morph_lists = []
    for sent in sentences:
        morph_list = []
        if tagger == 'kkma':
            pos_tagged_sentences = kkma.pos(sent)
        elif tagger == 'mecab':
            pos_tagged_sentences = mecab.pos(sent)
            
        for (key, value) in pos_tagged_sentences:
            value = transform_pos(value, tagger)
            morph_list.append([key, value])
        morph_lists.append(morph_list)

    return morph_lists


if __name__=="__main__":
    ############################
    # Load input text
    ############################
    if is_preprocess == 1:
        test_sentences = load_sentences(input_filename, zeros=1)
    else:
        sentences = split_sentence(input_filename)
        test_sentences = tag_pos(sentences, tagger='mecab')
        pprint(test_sentences)


    print "Loading...NER model"
    model = Model(model_path=model_path)
    parameters = model.parameters

    # Load the mappings
    word_to_id, slb_to_id, char_to_id, tag_to_id, pos_to_id = [
        {v: k for k, v in x.items()}
        for x in [model.id_to_word, model.id_to_slb, model.id_to_char, model.id_to_tag, model.id_to_pos]
    ]
    id_to_tag = model.id_to_tag

    # Load the model
    _, f_eval = model.build(training=False, **parameters)
    model.reload()

    start = time.time()

    print 'Running...NER'
    test_data = prepare_sentence(test_sentences, word_to_id, slb_to_id, char_to_id, pos_to_id)
    gazette_dict = make_gazette_to_dic(dict_path)

    gazette_dict_for, gazette_dict_len = dict(), dict()
    with open(dict_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            words, tag = line[0], line[1]

            if len(words) > 3:
                gazette_dict_len[words] = len(words)
                gazette_dict_for[words] = tag

    gazette_dict_len = sorted(gazette_dict_len.iteritems(), key=itemgetter(1), reverse=True)
    sentence_lists = evaluate_lexicon_tagger(parameters, f_eval, test_sentences, test_data,
                                            id_to_tag, gazette_dict, max_label_len=parameters['lexicon_dim'],
                                            output_path=output_filename)

