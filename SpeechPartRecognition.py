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


def create_transition_emission_dicts(sentences, words, smoothing=False):
    transitions_dict, possible_tags = create_transitions_dict(sentences, words)
    emission_dict = create_emission_dict(words, smoothing)
    possible_tags = possible_tags[possible_tags['tag'] != "START"]['tag'].tolist()
    possible_tags.insert(0, "START")
    return transitions_dict, emission_dict, possible_tags


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


def create_emission_dict(words, smoothing):
    # Count occurrences of each word (on its own) in the training data:
    tags = words['tag']
    all_tags = pd.DataFrame(tags)
    tags_count = all_tags.value_counts().reset_index()


    # Compute the emission probability for each pair of a word and it's tag:
    words_and_tags = pd.DataFrame(words, columns=['word', 'tag'])
    emissions_df = words_and_tags.drop_duplicates()
    words_and_tags = words_and_tags.groupby(['word', 'tag']).size().reset_index().rename(columns={0: 'pair_count'})
    emissions_df = pd.merge(emissions_df, words_and_tags, how="left", on=['word', 'tag'])
    if smoothing:
        emissions_df['pair_count'] += 1
    emissions_df = pd.merge(emissions_df, tags_count, how='left', on='tag').rename(columns={0: 'tag_count'})
    delta = len(tags_count.index) if smoothing else 0
    emissions_df['emission'] = emissions_df['pair_count'] * (1 / (emissions_df['tag_count'] + delta))
    emissions_df['pair'] = list(zip(emissions_df.word, emissions_df.tag))
    emissions_dict = dict(zip(emissions_df.pair, emissions_df.emission))
    return emissions_dict


# PART B - VITERBI ALGORITHM:

def create_viterbi_tables(sentence, transitions_dict, emissions_dict, possible_tags):
    n_words = len(sentence)
    n_tags = len(possible_tags)
    back_pointer_table = np.zeros((n_tags, n_words + 1)).astype(int)
    pi_table = np.zeros((n_tags, n_words + 1))

    for t in range(n_words):  # for each word
        word = sentence[t][0]

        for j in range(n_tags):  # for each possible tag
            end_max_tag_index = None
            end_max_score = -1.0
            cur_tag = possible_tags[j]

            for i in range(n_tags):  # for each possible previous tag
                prev_tag = possible_tags[i]
                pi_value = 1 if t == 0 else pi_table[i][t - 1]  # else condition regards the word START
                transition = transitions_dict.get((prev_tag, cur_tag),
                                                  0.0) if t != 0 else 1  # else condition regards the word START
                emission = emissions_dict.get((word, cur_tag), 0.0)
                score = pi_value * transition * emission
                if score > end_max_score:  # update max score up until current tag vs prev tag
                    end_max_tag_index, end_max_score = i, score
            pi_table[j][t] = end_max_score
            back_pointer_table[j][t] = end_max_tag_index

        t_col = pi_table[:, t]
        t_col_sum = np.sum(t_col)
        if t_col_sum == 0:  # if the entire column for word t is empty (meaning no possible tag was found)
            pi_table[:, t] = (1 / n_tags)  # enter the word "NN" with the same probability for each prev word
            back_pointer_table[:, t] = 3

    # calculate column of END word:
    for j in range(n_tags):  # for each possible tag
        cur_tag = possible_tags[j]
        pi_val = pi_table[j][n_words - 1]
        transition = transitions_dict.get((cur_tag, '.'), 0.0)
        end_score = transition * pi_val
        pi_table[j][n_words] = end_score
        back_pointer_table[j][n_words] = j

    return pi_table, back_pointer_table


def viterbi_predict_sentence_tags(sentence, transitions_dict, emissions_dict, possible_tags):
    n_words = len(sentence)
    n_tags = len(possible_tags)
    pi_table, back_pointer_table = create_viterbi_tables(sentence, transitions_dict, emissions_dict, possible_tags)

    tag_prediction = [''] * n_words
    # pick last word:
    max_score = -1
    max_tag_index = None
    for j in range(n_tags):
        score = pi_table[j][n_words]
        tag_index = back_pointer_table[j][n_words]
        if score > max_score:
            max_score, max_tag_index = score, tag_index
    tag_prediction[n_words - 1] = max_tag_index

    for word_index in range(n_words - 2, -1, -1):
        tag_index = back_pointer_table[(tag_prediction[word_index + 1], word_index)]
        if tag_index == 0 and word_index != 0:
            cur_tags_list = []
            next_tag = possible_tags[tag_prediction[word_index + 1]]
            for i in range(n_tags):
                other_tag = possible_tags[i]
                if transitions_dict[(other_tag, next_tag)] > 0:
                    cur_tags_list.append(i)
            tag_index = 3 if len(cur_tags_list) == 0 else rn.choice(cur_tags_list)
        tag_prediction[word_index] = tag_index

    for word_index, tag_index in enumerate(tag_prediction):
        tag_prediction[word_index] = possible_tags[tag_index]

    return tag_prediction


def viterbi_error_rates(test_sentences, train_words, transitions_dict, emissions_dict, possible_tags):
    unknown_true = 0
    unknown_false = 0
    known_true = 0
    known_false = 0
    train_words_list = train_words['word'].drop_duplicates().tolist()
    counter = 0
    for sentence in test_sentences:
        predicted_tags = viterbi_predict_sentence_tags(sentence, transitions_dict, emissions_dict, possible_tags)
        # print(counter + 0.5)
        for i, (word, true_tag) in enumerate(sentence):
            predicted_tag = predicted_tags[i]
            known = word in train_words_list
            correct = predicted_tag == true_tag
            unknown_true += (not known and correct)
            unknown_false += (not known and not correct)
            known_true += (known and correct)
            known_false += (known and not correct)
        # print(counter)
        counter += 1

    total = unknown_false + unknown_true + known_false + known_true
    known_error_rate = known_false / (known_false + known_true)
    unknown_error_rate = unknown_false / (unknown_false + unknown_true)
    total_error_rate = (known_false + unknown_false) / total

    return known_error_rate, unknown_error_rate, total_error_rate

if __name__ == '__main__':
    # ------------------Question a------------------#
    train_sentences, test_sentences = create_sentences_data()
    train_words, test_words = create_words_data()

    # ------------------Question b------------------#
    # Part a:
    mle_labels = create_all_MLE_labels(train_words)

    # Part b:
    mle_known_error_rate, mle_unknown_error_rate, mle_total_error_rate = MLE_error_rates(test_words, mle_labels)
    print("MLE tagger results:")
    print("Error rate for known words:", mle_known_error_rate)
    print("Error rate for unknown words:", mle_unknown_error_rate)
    print("Total error rate:", mle_total_error_rate,"\n")

    # ------------------Question c------------------#
    # Part a:
    train_words_start = add_start_tags_to_words_df(train_sentences, train_words)
    test_words_start = add_start_tags_to_words_df(test_sentences, test_words)
    transitions_dict, emissions_dict, possible_tags = create_transition_emission_dicts(train_sentences, train_words_start)
    predicted_tags = 0
    viterbi_error_rates(test_sentences, train_sentences, transitions_dict, emissions_dict, possible_tags)