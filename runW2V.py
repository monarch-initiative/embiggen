from urllib.request import urlopen
import os.path

# Tale of Two Cities, Dickens, from Gutenberg project
from hn2v.text_encoder import TextEncoder
from hn2v.word2vec import SkipGramWord2Vec

local_file = 'twocities.txt'


if not os.path.exists(local_file):
    url = 'https://www.gutenberg.org/files/98/98-0.txt'
    data = urlopen(url)
    fh = open(local_file, 'w')
    fh.write(data)
else:
    print("{} was previously downloaded".format(local_file))

encoder = TextEncoder(local_file)
data, count, dictionary, reverse_dictionary = encoder.build_dataset()
print("Extracted a dataset with %d words" % len(data))
model = SkipGramWord2Vec(data, worddictionary=dictionary, reverse_worddictionary=reverse_dictionary)
model.add_display_words(count)
model.train(display_step=1000)