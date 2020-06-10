import embiggen
from embiggen import TextEncoder
from embiggen import Embiggen
from embiggen import Shelldump

# read a corpise of texts
path = '/home/peter/data/embiggen/Emails.csv'
encoder = TextEncoder(path, data_type="words")
tensor_data, count_list, dictionary, reverse_dictionary = encoder.build_dataset()
print("Done reading data, got n={} words".format(len(count_list)))

# do embedding
embiggen = Embiggen()

embiggen.fit(
        data=tensor_data,
        worddict=dictionary,
        reverse_worddict=reverse_dictionary,
        embedding_model = "cbow",
        epochs = 1,
        embedding_size = 20,
        callbacks=[Shelldump()]
    )