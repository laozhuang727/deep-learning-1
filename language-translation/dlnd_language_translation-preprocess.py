# coding: utf-8

# # Language Translation
# In this project, you’re going to take a peek into the realm of neural network machine translation.  You’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
# ## Get the Data
# Since translating the whole language of English to French will take lots of time to train, we have provided you with a small portion of the English corpus.

# In[1]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)

# ## Explore the Data
# Play around with view_sentence_range to view different parts of the data.

# In[2]:

view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# ## Implement Preprocessing Function
# ### Text to Word Ids
# As you did with other RNNs, you must turn the text into a number
# so the computer can understand it.
# In the function `text_to_ids()`,
# you'll turn `source_text` and `target_text` from words to ids.
# However, you need to add the `<EOS>` word id at the end of each sentence from `target_text`.
# This will help the neural network predict when the sentence should end.
# 
# You can get the `<EOS>` word id by doing:
# ```python
# target_vocab_to_int['<EOS>']
# ```
# You can get other word ids using `source_vocab_to_int`
# and `target_vocab_to_int`.

# In[ ]:

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_sentences = source_text.split("\n")
    target_sentences = target_text.split("\n")
    source_id_text_list = []
    target_id_text_list = []
    for sentence in source_sentences:
        source_id_text = [source_vocab_to_int[word] for word in sentence.split()]
        source_id_text_list.append(source_id_text)

    for sentence in target_sentences:
        target_id_text = [target_vocab_to_int[word] for word in sentence.split()]
        target_id_text.append(target_vocab_to_int['<EOS>'])
        target_id_text_list.append(target_id_text)

    return source_id_text_list, target_id_text_list


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)

# ### Preprocess all the data and save it
# Running the code cell below will preprocess all the data and save it to file.

# In[ ]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
