import nltk
import pandas as pd

nltk.download('brown')

# This program loads the news section from the "brown" dataset of NLTK,
# Generates different part-of-speech tagigng algorithms, trains them,
# And plots their error rates as long as their confusion matrices.

# ------------------------------------------- QUESTION A ------------------------------------------------------ #
from nltk.corpus import brown

def create_sentences_data():
    full_data_sents_raw = brown.tagged_sents(categories=["news"])
    full_data_sents = []
    for sentence in full_data_sents_raw:
        new_sent = [('START', 'START')] if add_start else []
        for (word, tag) in sentence:
            cur_tuple = (word, tag.split('-')[0].split('+')[0])
            cur_tuple = cur_tuple if cur_tuple[1] != "" else (word, tag)
            new_sent.append(cur_tuple)
        full_data_sents.append(new_sent)
    train_data_sents = full_data_sents[:int(len(full_data_sents) * 0.9)]
    test_data_sents = full_data_sents[int(len(full_data_sents) * 0.9):]
    words_count = get_word_count_dict(train_data_sents)
    if pseudo_words:
        for i in range(len(train_data_sents)):
            for j in range(len(train_data_sents[i])):
                word, tag = train_data_sents[i][j]
                if word != "START" and words_count[word] <= 5:  # if it is a rare word
                    pseudo_word = get_pseudo_word(word)
                    train_data_sents[i][j] = (pseudo_word, tag)
    train_data_words, test_data_words = create_train_test_words_data(train_data_sents, test_data_sents)
    return train_data_sents, test_data_sents, train_data_words, test_data_words, words_count

def create_words_data():
    full_data_words_raw = brown.tagged_words(categories=["news"])
    full_data_words = []
    for (word, tag) in full_data_words_raw:
        cur_tuple = (word, tag.split('-')[0].split('+')[0])
        cur_tuple = cur_tuple if cur_tuple[1] != "" else (word, tag)
        full_data_words.append(cur_tuple)
    train_data_words = full_data_words[:int(len(full_data_words) * 0.9)]
    test_data_words = full_data_words[int(len(full_data_words) * 0.9):]
    train_words_count = ()
    if pseudo_words:
        train_words_count = get_word_count_dict(train_data_words)
        for i in range(len(train_data_words)):
            word, tag = train_data_words[i]
            if word != "START" and train_words_count[word] <= 5: # if it is a rare word
                pseudo_word = get_pseudo_word(word)
                train_data_words[i] = (pseudo_word,tag)
    return train_data_words, test_data_words, train_words_count


# ------------------------------------------- QUESTION B ------------------------------------------------------ #

def create_all_MLE_labels(words):
    """
    Creates MLE labels for all words in given dataset.
    :param words: Dataset of words and their tags.
    :return: A dictionary that maps each word in given dataset to it's predicted tag.
    """
    probabilities_df = pd.DataFrame(words, columns=['word', 'tag'])
    label_count_df = probabilities_df.groupby(['word', 'tag']).size().reset_index().rename(columns={0: 'count'})
    MLE_labels_df = label_count_df.sort_values('count', ascending=False).drop_duplicates(['word'])
    MLE_labels_df.drop(['count'], axis=1, inplace=True)
    return dict(zip(MLE_labels_df.word,MLE_labels_df.tag))


def get_MLE_label(word, MLE_labels):
    """
    Predicts MLE label for a single word.
    :param word: word to predict.
    :param MLE_labels: A dictionary that maps each word in given dataset to it's predicted tag.
    :return: predicted tag and a boolean that represents if the word is unseen in the training data.
    """
    is_unknown = True
    tag = "NN"
    if word in MLE_labels:
        is_unknown = False
        tag = MLE_labels[word]
    return tag, is_unknown


def MLE_error_rates(test_words, MLE_labels):
    """
    Calculates MLE error rates.
    :param test_words: test dataset of words and their true tags.
    :param MLE_labels: A dictionary that maps each word in given dataset to it's predicted tag.
    :return: known_error_rate, unknown_error_rate, total_error_rate
    """
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


def create_transition_emission_dicts(sentences, words, smoothing=False):
    """
    Creates transition and emmision tables for given training dataset of sentences.
    :param sentences:  training dataset of sentences.
    :param words:  training dataset of words.
    :param smoothing: True for add-1 smoothing, default is False
    :return: transsmition table (as dict), emmision table (as dict), a list of all possible tags.
    """
    transitions_dict, possible_tags = create_transitions_dict(sentences, words)
    emission_dict = create_emission_dict(words, smoothing)
    possible_tags = possible_tags[possible_tags['tag'] != "START"]['tag'].tolist()
    possible_tags.insert(0, "START")
    return transitions_dict, emission_dict, possible_tags


def create_transitions_dict(sentences, words):
    """
    Creates a transmition table for given dataset.
    :param sentences:  training dataset of sentences.
    :param words:  training dataset of words.
    :return: transsmition table (as dict)
    """
    # Count occurrences of each tag (on its own) in the training data:
    words = pd.DataFrame(words, columns=['word', 'tag'])
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
    """
    Creates an emission table for given dataset.
    :param sentences:  training dataset of sentences.
    :param words:  training dataset of words.
    :return: emissiontable (ad dict)
    """
    # Count occurrences of each word (on its own) in the training data:
    words = pd.DataFrame(words, columns=['word', 'tag'])
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

def create_viterbi_tables(sentence, transitions_dict, emissions_dict, possible_tags, train_words_dict, pseudo_words):
    """
    Creates Viterbi tables (pi table and back pointer table) for given dataset.
    :param sentence:  dataset of tagged sentences.
    :param transitions_dict:  transsmition table (as dict)
    :param emissions_dict: emissionn table (ad dict)
    :param possible_tags: a list of all possible tags in training data
    :param train_words_dict: a dictionary that maps each word in the training data to it's count
    :param pseudo_words: True for using pseudo words on rare words, default is False
    :return: pi_table, back_pointer_table
    """
    n_words = len(sentence)
    n_tags = len(possible_tags)
    back_pointer_table = np.zeros((n_tags, n_words + 1)).astype(int)
    pi_table = np.zeros((n_tags, n_words + 1))

    for t in range(n_words):  # for each word
        word = sentence[t][0]
        if pseudo_words and word not in train_words_dict:
            word = get_pseudo_word(word)

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


def viterbi_predict_sentence_tags(sentence, transitions_dict, emissions_dict, possible_tags, train_words_dict,
                                  pseudo_words):
    """
    Predicts tags for a given sentence.
    :param sentence:  dataset of tagged sentences.
    :param transitions_dict:  transsmition table (as dict)
    :param emissions_dict: emissionn table (ad dict)
    :param possible_tags: a list of all possible tags in training data
    :param train_words_dict: a dictionary that maps each word in the training data to it's count
    :param pseudo_words: True for using pseudo words on rare words, default is False
    :return: A list of predicted tags for given sentence.
    """
    n_words = len(sentence)
    n_tags = len(possible_tags)
    pi_table, back_pointer_table = create_viterbi_tables(sentence, transitions_dict, emissions_dict, possible_tags,
                                                         train_words_dict, pseudo_words)

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

    # Run backwords from end of sentence and presict each word's tag:
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

    # replace indexes with actual tags:
    for word_index, tag_index in enumerate(tag_prediction):
        tag_prediction[word_index] = possible_tags[tag_index]

    return tag_prediction


def viterbi_error_rates(test_sentences, train_words, transitions_dict, emissions_dict, possible_tags,
                        train_words_dict=(), pseudo_words=False, return_tags=False):
    """

    :param sentence:  dataset of tagged sentences.
    :param train_words:  dataset of tagged words.
    :param transitions_dict:  transsmition table (as dict)
    :param emissions_dict: emissionn table (ad dict)
    :param possible_tags: a list of all possible tags in training data
    :param train_words_dict: a dictionary that maps each word in the training data to it's count
    :param pseudo_words: True for using pseudo words on rare words, default is False
    :param return_tags:  True for returning arrays of true and predicted tags
    :return: known_error_rate, unknown_error_rate, total_error_rate, true_tags, predicted_tags_list
    """
    unknown_true = 0
    unknown_false = 0
    known_true = 0
    known_false = 0
    train_words = pd.DataFrame(train_words, columns=['word', 'tag'])
    train_words_list = train_words['word'].drop_duplicates().tolist()
    counter = 0
    true_tags = []
    predicted_tags_list = []
    for sentence in test_sentences:
        predicted_tags = viterbi_predict_sentence_tags(sentence, transitions_dict, emissions_dict, possible_tags,
                                                       train_words_dict, pseudo_words)
        for i, (word, true_tag) in enumerate(sentence):
            predicted_tag = predicted_tags[i]
            true_tags.append(true_tag)
            predicted_tags_list.append(predicted_tag)
            known = word in train_words_list
            correct = predicted_tag == true_tag
            unknown_true += (not known and correct)
            unknown_false += (not known and not correct)
            known_true += (known and correct)
            known_false += (known and not correct)

    total = unknown_false + unknown_true + known_false + known_true
    known_error_rate = known_false / (known_false + known_true)
    unknown_error_rate = unknown_false / (unknown_false + unknown_true)
    total_error_rate = (known_false + unknown_false) / total
    if not return_tags:
        return known_error_rate, unknown_error_rate, total_error_rate
    else:
        return known_error_rate, unknown_error_rate, total_error_rate, true_tags, predicted_tags_list

# ------------------------------------------- QUESTION E ------------------------------------------------------ #

def get_pseudo_word(word):
    """
    Gets the pseudo word for given word.
    :param word: word
    :return: pseudo word for given word.
    """
    pseudo_words_reg = [
        (r'^\w+ing$', 'ending_with_ing'),  # suffixes
        (r'^\w+ed$', 'ending_with_ed'),
        (r'^\w+tion$', 'ending_with_tion'),
        (r'^\w+er$', 'ending_with_er'),
        (r'^\w+s$', 'plural_word'),

        (r'^\d{2}$', 'number_2digit'),  # numbers
        (r'^\d{4}$', 'number_4digit'),
        (r'^\d+$', 'number_general'),
        (r'^\'\d{2}', 'year'),
        (r'^\d{1,2}:\d{2}(:\d{2})?$', 'timestamp'),
        (r'\d+%$', 'percentage'),
        (r'^\d+\.\d+$', 'number_point'),
        (r'^\$(\d|,)+(\.\d+)?$', 'dollar_price'),

        (r'^\w+\'[a-z]*$', 'contain_apostrophe'),  # special characters
        (r'^(\w|-)+-(\w|-)+$', 'contain_hyphen'),
        (r'^[A-Z]+$', 'acronym'),
        (r'^[A-Z][a-z]+$', 'capitalized'),
        (r'[a-z]+$', 'not_capitalized')]

    for regex, pseudo_word in pseudo_words_reg:
        if re.search(regex, word):
            return pseudo_word
    return "general_pseudo"  # if no other tag matched


def plot_confusion_matrix(tags_true, tags_predict, unique_tags):
    """
    Creates a confusion matrix for given lists of tags.
    :param tags_true: A list of true tags.
    :param tags_predict: A list of predicted tags.
    :param unique_tags: A list of all unique tags in
    :return: None
    """
    my_mat = confusion_matrix(tags_true, tags_predict, labels=unique_tags)
    # my_mat_norm = (1 / my_mat.sum(axis=1)[:, np.newaxis]).T * my_mat
    plt.imshow(my_mat)

    plt.xlabel('predicted tags')
    plt.xticks(ticks=np.arange(len(unique_tags)), labels=unique_tags)

    plt.ylabel('true tags')
    plt.yticks(ticks=np.arange(len(unique_tags)), labels=unique_tags)

    # plt.colorbar()
    plt.show()

# ------------------------------------------- RUN PROGRAM ------------------------------------------------------ #


if __name__ == '__main__':
    # ------------------Question a------------------#
    train_sentences, test_sentences, train_words, test_words, words_count = create_data()

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