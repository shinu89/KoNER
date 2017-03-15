import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import re
import codecs
from utils import create_dico, create_mapping, isNumber
from utils import iob2, transform_iob
from unicodedata import name

def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    with open(path, 'r') as f:
    # for line in codecs.open(path, 'r', 'utf8'):
        for line in f.readlines():
            line = isNumber(line.rstrip()) if zeros else line.rstrip()
            if not line:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
            else:
                word = line.split('\t')
                assert len(word) >= 2
                sentence.append([word[0], word[1], word[2]])
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def make_gazette_to_dic(input_path):
    """
    Represent words in dictionary with BIO boundary
    For example, "LA Lakers    OG" is represented as "LG   B-OG, Lakers   E-OG".
    Final output becomes LG = [0, 0, 1, 1, ..., 0] (1 means possible label for word "LG")
    :param input_path: file path for dictionary
    :return: list of zero or one representation
    """
    gazette_dict = dict()
    label_dict = dict()
    with open(input_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            words, tag = line[0], line[1]
            if len(words.split()) > 1:
                words_list = words.split()
                for i in range(len(words_list)):
                    tag_list = []
                    if i==0:
                        new_tag = 'B-'+tag
                    elif i==len(words_list)-1:
                        new_tag = 'E-'+tag
                    else:
                        new_tag = 'I-'+tag
                    if words_list[i] in gazette_dict:
                        tag_list = gazette_dict[words_list[i]]
                    if new_tag not in tag_list:
                        tag_list.append(new_tag)
                        gazette_dict[words_list[i]] = tag_list
                    if new_tag not in label_dict:
                        label_dict[new_tag] = len(label_dict)

            else:
                dict_list = []
                if words in gazette_dict:
                    dict_list = gazette_dict[words]
                new_tag = 'S-'+tag
                if new_tag not in dict_list:
                    dict_list.append(new_tag)
                    gazette_dict[words] = dict_list
                if new_tag not in label_dict:
                    label_dict[new_tag] = len(label_dict)

    new_gazette_dict = dict()
    for word in gazette_dict:
        zero_list = [0] * len(label_dict)
        for l in gazette_dict[word]:
            zero_list[label_dict[l]] = 1
        new_gazette_dict[word] = zero_list

    return new_gazette_dict


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = transform_iob(tags, type='iobes')
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobs':
            new_tags = transform_iob(tags, type='iobs')
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0]+'/'+x[1] for x in s] for s in sentences]

    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print "Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    )
    return dico, word_to_id, id_to_word


def separate(ch):
    uindex = ord(ch) - 0xac00
    jongseong = uindex % 28
    joongseong = ((uindex - jongseong) // 28) % 21
    choseong = ((uindex - jongseong) // 28) // 21
    if jongseong == 0:
        return [choseong, joongseong+100]
    else:
        return [choseong, joongseong+100, jongseong+1000]


def char_mapping(sentences, isCharacter=True):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    sentence_lists = []
    for s in sentences:
        str_words = [unicode(w[0]) for w in s]
        char_lists = []
        for w in str_words:
            for c in w:
                if name(c).startswith('HANGUL') and isCharacter:
                    for ch in separate(c):
                        char_lists.append(ch)
                else:
                    char_lists.append(c)
        sentence_lists.append(char_lists)

    chars = sentence_lists
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique characters" % len(dico)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag


def pos_mapping(sentences, position=1):
    """
    Create a dictionary and a mapping of poss, sorted by frequency.
    """
    # tags = [[word[position][0] for word in s] for s in sentences]
    tags = [[word[position] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag


def prepare_sentence(sentences, word_to_id, slb_to_id, char_to_id, pos_to_id):
    data = []
    new_id = len(pos_to_id)+1
    for s in sentences:
        str_words = [unicode(w[0]) for w in s]
        words = []
        for w in s:
            if w[0]+'/'+w[1] in word_to_id:
                words.append(word_to_id[w[0]+'/'+w[1]])
            else:
                words.append(word_to_id['<UNK>'])
        slbs = [[slb_to_id[c] for c in w if c in slb_to_id]
                 for w in str_words]
        chars = []
        for w in str_words:
            char_list = []
            for sl in w:
                if name(sl).startswith('HANGUL'):
                    for c in separate(sl):
                        if c in char_to_id:
                            char_list.append(char_to_id[c])
                else:
                    if sl in char_to_id:
                        char_list.append(char_to_id[sl])
            chars.append(char_list)

        poss = [pos_to_id[w[1]] if w[1] in pos_to_id else new_id
                for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'slbs' : slbs,
            'chars': chars,
            'poss': poss,
        })
    return data


def prepare_dataset(sentences, word_to_id, slb_to_id, char_to_id, pos_to_id, tag_to_id, is_train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    new_id = len(pos_to_id)+1
    for s in sentences:
        str_words = [unicode(w[0]) for w in s]
        words = [word_to_id[w+'/'+p if w+'/'+p in word_to_id else '<UNK>'] for (w, p, l) in s]

        # Skip characters that are not in the training set
        words_slb_list, words_char_list = [], []
        for w in str_words:
            slbs_list, chars_list = [], []
            for sl in w:
                if sl in slb_to_id:
                    slbs_list.append(slb_to_id[sl])
                if name(sl).startswith('HANGUL'):
                    for c in separate(sl):
                        if c in char_to_id:
                            chars_list.append(char_to_id[c])
                else:
                    if sl in char_to_id:
                        chars_list.append(char_to_id[sl])
            words_char_list.append(chars_list)
            words_slb_list.append(slbs_list)
        slbs = words_slb_list
        chars = words_char_list

        poss = [pos_to_id[p] if p in pos_to_id else new_id for (w, p, l) in s]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'slbs': slbs,
            'chars': chars,
            'poss': poss,
            'tags': tags,
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split('\t')[0].strip()
        for line in codecs.open(ext_emb_path, 'r', encoding='utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0
    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word
