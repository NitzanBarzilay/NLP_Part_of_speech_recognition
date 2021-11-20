import nltk
import pandas as pd

nltk.download('brown')

# ------------------------------------------- QUESTION A ------------------------------------------------------ #
from nltk.corpus import brown

def create_sentences_data():
    full_data_sents_raw = brown.tagged_sents(categories=["news"])
    full_data_sents = []
    for sentence in full_data_sents_raw:
        new_sent = [('START', 'START')]
        for (word, tag) in sentence:
            cur_tuple = (word, tag.split('-')[0].split('+')[0])
            cur_tuple = cur_tuple if cur_tuple[1] != "" else (word, tag)
            new_sent.append(cur_tuple)
        full_data_sents.append(new_sent)
    train_data_sents = full_data_sents[:int(len(full_data_sents) * 0.9)]
    test_data_sents = full_data_sents[int(len(full_data_sents) * 0.9):]
    return train_data_sents, test_data_sents

def create_words_data():
    full_data_words_raw = brown.tagged_words(categories=["news"])
    full_data_words = []
    for (word, tag) in full_data_words_raw:
        cur_tuple = (word, tag.split('-')[0].split('+')[0])
        cur_tuple = cur_tuple if cur_tuple[1] != "" else (word, tag)
        full_data_words.append(cur_tuple)
    train_data_words = full_data_words[:int(len(full_data_words) * 0.9)]
    test_data_words = full_data_words[int(len(full_data_words) * 0.9):]
    return train_data_words, test_data_words

# ------------------------------------------- QUESTION B ------------------------------------------------------ #

def create_all_MLE_labels(words):
    probabilities_df = pd.DataFrame(words, columns=['word', 'tag'])
    label_count_df = probabilities_df.groupby(['word', 'tag']).size().reset_index().rename(columns={0: 'count'})
    MLE_labels_df = label_count_df.sort_values('count', ascending=False).drop_duplicates(['word'])
    MLE_labels_df.drop(['count'], axis=1, inplace=True)
    return MLE_labels_df


def get_MLE_label(word, MLE_labels):
    label = MLE_labels[MLE_labels['word'] == word]['tag'].values
    label = label[0] if len(label) > 0 else "NN"
    return label


def get_accuracy(test_words, MLE_labels):
    predictions_df = pd.DataFrame(test_words, columns=['word', 'true_label'])
    predictions_df = pd.merge(predictions_df, MLE_labels, how="left", on=['word'])
    predictions_df.reset_index(inplace=True, drop=True)
    predictions_df['is_correct'] = predictions_df['tag'] == predictions_df['true_label']
    accuracy = predictions_df['is_correct'].sum() / len(predictions_df.index)
    return (accuracy)

# ------------------------------------------- QUESTION C ------------------------------------------------------ #

# PART A - TRAINING PHASE:

def add_start_tags_to_words_df(sentences, words):
    start_rows = []
    for i in range(len(sentences)):
        start_rows.append(["START", "START"])
    start_rows_df = pd.DataFrame(start_rows, columns=['word', 'tag'])
    words_df = pd.DataFrame(words, columns=['word', 'tag'])
    words_updated = pd.concat([words_df, start_rows_df]).reset_index(drop=True)
    return words_updated


def create_transition_emission_dicts(sentences, words):
    words = add_start_tags_to_words_df(sentences, words)
    transitions_dict, tags = create_transitions_dict(sentences, words)
    emission_dict = create_emission_dict(sentences, words)
    return transitions_dict, emission_dict, tags


def create_transitions_dict(sentences, words):
    # Count occurrences of each tag (on its own) in the training data:
    tags = words['tag']
    all_tags = pd.DataFrame(tags)
    tags = pd.DataFrame(tags).drop_duplicates()
    tags.reset_index(inplace=True, drop=True)
    tags_count = all_tags.value_counts().reset_index()

    # Count occurrences of consecutive pairs of tags in the training data:
    tag_pairs = list(product(tags['tag'], tags['tag']))
    transitions_count = dict((tag_pair, 0) for tag_pair in tag_pairs)
    for sentence in sentences:
        for i in range(1, len(sentence)):
            prev_tag = sentence[i - 1][1]
            cur_tag = sentence[i][1]
            pair = (prev_tag, cur_tag)
            transitions_count[pair] += 1

    # Compute the transition probability for each pair of tags:
    transitions_df = pd.DataFrame.from_dict(transitions_count, orient="index").reset_index()
    transitions_df[['prev_tag', 'cur_tag']] = pd.DataFrame(transitions_df['index'].to_list(),
                                                           index=transitions_df.index)
    transitions_df = transitions_df[['prev_tag', 'cur_tag', 0]]
    transitions_df = transitions_df.rename({0: "pair_count"}, axis=1)
    transitions_df = pd.merge(transitions_df, tags_count, how="left", left_on=["prev_tag"], right_on=["tag"])
    transitions_df = transitions_df[['prev_tag', 'cur_tag', "pair_count", 0]]
    transitions_df = transitions_df.rename({0: "tag_count"}, axis=1)
    transitions_df['transition'] = transitions_df['pair_count'] * (1 / transitions_df['tag_count'])
    transitions_df['pair'] = list(zip(transitions_df.prev_tag, transitions_df.cur_tag))
    transition_dict = dict(zip(transitions_df.pair, transitions_df.transition))
    return transition_dict, tags


def create_emission_dict(words):
    # Count occurrences of each word (on its own) in the training data:
    only_words = words['word']
    all_words = pd.DataFrame(only_words)
    words_count = all_words.value_counts().reset_index()

    # Compute the emission probability for each pair of a word and it's tag:
    words_and_tags = pd.DataFrame(words, columns=['word', 'tag'])
    emissions_df = words_and_tags.drop_duplicates()
    words_and_tags = words_and_tags.groupby(['word', 'tag']).size().reset_index().rename(columns={0: 'pair_count'})
    emissions_df = pd.merge(emissions_df, words_and_tags, how="left", on=['word', 'tag'])
    emissions_df = pd.merge(emissions_df, words_count, how='left', on='word').rename(columns={0: 'word_count'})
    emissions_df['emission'] = emissions_df['pair_count'] * (1 / emissions_df['word_count'])
    emissions_df['pair'] = list(zip(emissions_df.word, emissions_df.tag))
    emissions_dict = dict(zip(emissions_df.pair, emissions_df.emission))
    return emissions_dict


# PART B - VITERBI ALGORITHM:

def create_viterbi_tables(sentence, transitions_dict, emissions_dict, possible_tags):
    pi_table = defaultdict(float)
    back_pointer_table = dict()
    pi_table[(0, "START")] = 1

    for i in range(len(sentence)):
        word_possible_tags = possible_tags if i>0 else ['START']
        word = sentence[i]
        for j in possible_tags:
            max_tag = None
            max_score = -1
            for tag in word_possible_tags:
                tag_score = pi_table.get((i - 1), 0) * transitions_dict[(tag, j)] * emissions_dict[(j, word)]
                if tag_score > max_score:
                    max_tag, max_score = tag, tag_score
            pi_table[(i + 1, j)] = max_score
            back_pointer_table[(i + 1, j)] = max_tag
    return pi_table, back_pointer_table


def predict_sentence_tags(sentence, transitions_dict, emissions_dict, possible_tags):
    pi_table, back_pointer_table = create_viterbi_tables(sentence, transitions_dict, emissions_dict, possible_tags)
    max_tag = None
    max_score = -1
    n = len(sentence)
    for tag in possible_tags:
        tag_score = pi_table.get((n, tag), 0) * transitions_dict[(tag, ".")]
        if tag_score > max_score:
            max_tag, max_score = tag, tag_score

    # Fill tags:
    p_tags = n * [0] # initialize empty tag array
    p_tags[-1] = max_tag # fill base case.
    for i in (n - 2, -1, -1): # bp the tags.
        p_tags[i] = back_pointer_table[(i + 2, p_tags[i + 1])]
    return p_tags

breakpoint()
