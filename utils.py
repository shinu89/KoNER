import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import re
import codecs
import numpy as np

eval_path = "./evaluation"
eval_script = os.path.join(eval_path, "conlleval")


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def isNumber(s):
    try:
        float(s.split('\t')[0])
        new = 'NUM'+'\t'+s.split('\t')[1]+'\t'+s.split('\t')[2]
        return new
    except ValueError:
        return s


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def transform_iob(tags, type='iobes'):
    """
    IOB -> IOBES or IOB -> IOBS
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            elif type=='iobes':
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES or IOBS -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input(data, parameters, add_label, singletons=None, gazette_dict=dict(), max_label_len=20):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    slbs = data['slbs']
    chars = data['chars']
    str_words = data['str_words']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['pos_dim']:
        poss = data['poss']
    if parameters['lexicon_dim']:
        lexs = []
        for w in str_words:
            w = str(w)
            # print 'ww', w
            if w in gazette_dict:
                # print 'w', w
                lexs.append(gazette_dict[w])
            else:
                lexs.append([0]*max_label_len)
    slb_for, slb_rev, slb_pos = pad_word_chars(slbs)
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    if parameters['word_dim']:
        input.append(words)
    if parameters['slb_dim']:
        input.append(slb_for)
        if parameters['slb_lstm_dim']:
            if parameters['slb_bidirect']:
                input.append(slb_rev)
            input.append(slb_pos)
    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_lstm_dim']:
            if parameters['char_bidirect']:
                input.append(char_rev)
            input.append(char_pos)
    if parameters['pos_dim']:
        input.append(poss)
    if parameters['lexicon_dim']:
        input.append(lexs)
    if add_label:
        input.append(data['tags'])
    return input


def evaluate_lexicon(parameters, f_eval, raw_sentences, parsed_sentences,
                     id_to_tag, dictionary_tags, eval_temp, gazette_dict, max_label_len):

    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        # input = create_input(data, parameters, False)
        input = create_input(data, parameters, False, singletons=None, gazette_dict=gazette_dict, max_label_len=max_label_len)
        if parameters['crf']:
            y_preds = np.array(f_eval(*input))[1:-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)
        y_reals = np.array(data['tags']).astype(np.int32)
        assert len(y_preds) == len(y_reals)

        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]

        if parameters['tag_scheme'] == 'iobes' or parameters['tag_scheme'] == 'iobs':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = str(raw_sentence[i][0]) + ' ' + str(raw_sentence[i][1]) + \
                       ' ' + str(r_tags[i]) + ' ' + str(p_tags[i])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")

    # Write predictions to disk and run CoNLL script externally
    eval_id = np.random.randint(1000000, 2000000)
    output_path = os.path.join(eval_temp, "eval.%i.output" % eval_id)
    scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)

    with open(output_path, 'w') as f:
        for p in predictions:
            temp = p.split()
            for i in temp:
                f.write(str(i).decode('utf-8').encode('cp949')+' ')
            f.write('\n')

    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    for line in eval_lines:
        print line

    # Confusion matrix with accuracy for each tag
    print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
            "ID", "NE", "Total",
            *([id_to_tag[i] for i in xrange(n_tags)] + ["Percent"])
    )
    for i in xrange(n_tags):
        print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
                str(i), id_to_tag[i], str(count[i].sum()),
                *([count[i][j] for j in xrange(n_tags)] +
                  ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        )

    # Global accuracy
    print "%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    )

    return float(eval_lines[1].strip().split()[-1])


def evaluate_lexicon_tagger(parameters, f_eval, raw_sentences, parsed_sentences,
                            id_to_tag, gazette_dict, max_label_len, output_path):

    """
    Evaluate current model using CoNLL script.
    """
    predictions = []
    count = 0
    sentence_lists = []
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        sentence_list = []
        count += 1
        if count % 50 == 0:
            print count, "/", len(raw_sentences),"..."
        input = create_input(data, parameters, False, singletons=None, gazette_dict=gazette_dict,
                             max_label_len=max_label_len)
        if parameters['crf']:
            y_preds = np.array(f_eval(*input))[1:-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)

        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]

        if parameters['tag_scheme'] == 'iobes' or parameters['tag_scheme'] == 'iobs':
            p_tags = iobes_iob(p_tags)

        for i in range(len(y_preds)):
            new_line = str(raw_sentence[i][0]) + '\t' + str(raw_sentence[i][1]) + '\t' + str(p_tags[i])
            predictions.append(new_line)
            sentence_list.append(new_line)
        sentence_lists.append(sentence_list)
        predictions.append("")

    output_path = os.path.join(output_path)

    with open(output_path, 'w') as f:
        for p in predictions:
            temp = p.split()
            for i in temp:
                f.write(str(i).decode('utf-8').encode('cp949')+'\t')
            f.write('\n')

    return sentence_lists

