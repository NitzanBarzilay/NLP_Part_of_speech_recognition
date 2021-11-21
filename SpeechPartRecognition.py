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
    return dict(zip(MLE_labels_df.word,MLE_labels_df.tag))


def get_MLE_label(word, MLE_labels):
    is_unknown = True
    tag = "NN"
    if word in MLE_labels:
        is_unknown = False
        tag = MLE_labels[word]
    return tag, is_unknown


def MLE_error_rates(test_words, MLE_labels):
    unknown_right = 0
    unknown_wrong = 0
    known_right = 0
    known_wrong = 0
    for (word,tag) in test_words:
        predicted_tag,is_unknown = get_MLE_label(word, MLE_labels)
        if is_unknown:
            if predicted_tag == tag:
                unknown_right += 1
            else:
                unknown_wrong += 1
        else:
            if predicted_tag == tag:
                known_right += 1
            else:
                known_wrong += 1

    known_error_rate = known_wrong / (known_wrong + known_right)
    unknown_error_rate = unknown_wrong / (unknown_wrong + unknown_right)
    total_error_rate = (known_wrong + unknown_wrong) / len(test_words)
    return known_error_rate, unknown_error_rate, total_error_rate

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
    transitions_dict, tags = create_transitions_dict(sentences, words)
    emission_dict = create_emission_dict(words)
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
    transitions_df = transitions_df.rename({0: "prev_tag_count"}, axis=1)
    transitions_df['transition'] = transitions_df['pair_count'] * (1 / transitions_df['prev_tag_count'])
    transitions_df['pair'] = list(zip(transitions_df.prev_tag, transitions_df.cur_tag))
    transition_dict = dict(zip(transitions_df.pair, transitions_df.transition))


    return transition_dict, tags


def create_emission_dict(words):
    # Count occurrences of each word (on its own) in the training data:
    tags = words['tag']
    all_tags = pd.DataFrame(tags)
    tags_count = all_tags.value_counts().reset_index()

    # Compute the emission probability for each pair of a word and it's tag:
    words_and_tags = pd.DataFrame(words, columns=['word', 'tag'])
    emissions_df = words_and_tags.drop_duplicates()
    words_and_tags = words_and_tags.groupby(['word', 'tag']).size().reset_index().rename(columns={0: 'pair_count'})
    emissions_df = pd.merge(emissions_df, words_and_tags, how="left", on=['word', 'tag'])
    emissions_df = pd.merge(emissions_df, tags_count, how='left', on='tag').rename(columns={0: 'tag_count'})
    emissions_df['emission'] = emissions_df['pair_count'] * (1 / emissions_df['tag_count'])
    emissions_df['pair'] = list(zip(emissions_df.word, emissions_df.tag))
    emissions_dict = dict(zip(emissions_df.pair, emissions_df.emission))

    emi_dict = defaultdict(lambda: 1e-24)
    for key in emissions_dict:
        emi_dict[key] = emissions_dict[key]

    # return emissions_dict

    return emi_dict


# PART B - VITERBI ALGORITHM:

def create_viterbi_tables(sentence, transitions_dict, emissions_dict, possible_tags):
    pi_table = defaultdict(float)
    back_pointer_table = dict()
    pi_table[(0, "START")] = 1
    possible_tags = possible_tags[possible_tags['tag'] != "START"]
    possible_tags = possible_tags['tag'].tolist()
    sentence=sentence[1:]

    for i in range(1,len(sentence)+1):
        cur_possible_tags = ['START'] if i<1 else possible_tags

        word = sentence[i-1][0]
        for cur_tag in cur_possible_tags:
            max_tag = None
            max_score = float("-Inf")

            prev_possible_tags = ['START'] if i-1<1 else possible_tags
            for prev_tag in prev_possible_tags:
                pi_val = pi_table.get((i - 1, prev_tag), 0.0)
                transition = transitions_dict.get((prev_tag, cur_tag),0.0)
                emission = emissions_dict.get((word, cur_tag), 1e-24)
                tag_score = pi_val * transition * emission

                if tag_score > max_score:
                    max_tag, max_score = prev_tag, tag_score

            pi_table[(i, cur_tag)] = max_score

            back_pointer_table[(i, cur_tag)] = max_tag
    return pi_table, back_pointer_table


def viterbi_predict_sentence_tags(sentence, transitions_dict, emissions_dict, possible_tags):
    pi_table, back_pointer_table = create_viterbi_tables(sentence, transitions_dict, emissions_dict, possible_tags)
    possible_tags = possible_tags[possible_tags['tag'] != "START"]
    possible_tags = possible_tags['tag'].tolist()
    max_tag = None
    max_score = -1
    sentence = sentence[1:]
    n = len(sentence)
    for tag in possible_tags:
        pi_val = pi_table.get((n , tag), 0)
        transition = 0
        if (tag, '.') in transitions_dict:
            transition = transitions_dict[(tag, ".")]
        tag_score = pi_val * transition
        if tag_score > max_score:
            max_tag, max_score = tag, tag_score

    # Fill tags:
    p_tags = [""]*n  # initialize empty tag array
    p_tags[-1] = max_tag # fill base case.
    for i in range(n - 2, -1, -1): # bp the tags.
        p_tags[i] = back_pointer_table[(i + 2, p_tags[i + 1])]
    return p_tags

def viterbi_error_rates(test_sentences, train_sentences, transitions_dict, emissions_dict, possible_tags):
    for sentence in test_sentences:
        sentence_tags = viterbi_predict_sentence_tags(sentence, transitions_dict, emissions_dict, possible_tags)
    known_error_rate, unknown_error_rate, total_error_rate = 0, 0, 0
    return known_error_rate, unknown_error_rate, total_error_rate

if __name__ == '__main__':
    # Qustion A:
    train_sentences, test_sentences = create_sentences_data()
    train_words, test_words = create_words_data()

    # Qusetion B:
    # Part a:
    mle_labels = create_all_MLE_labels(train_words)
    known_error_rate, unknown_error_rate, total_error_rate = MLE_error_rates(test_words, mle_labels)
    print("------------------------ QUESTION B PART a ---------------------------")
    print("MLE tagger results:")
    print("Error rate for known words:", known_error_rate)
    print("Error rate for unknown words:", unknown_error_rate)
    print("Total error rate:", total_error_rate)

    # Part b:
    print("------------------------ QUESTION B PART b ---------------------------")
    train_words_start = add_start_tags_to_words_df(train_sentences, train_words)
    test_words_start = add_start_tags_to_words_df(test_sentences, test_words)
    transitions_dict, emissions_dict, possible_tags = create_transition_emission_dicts(train_sentences, train_words_start)
    predicted_tags = 0
    viterbi_error_rates(test_sentences, train_sentences, transitions_dict, emissions_dict, possible_tags)