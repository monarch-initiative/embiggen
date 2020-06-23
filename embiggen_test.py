import embiggen
from embiggen import TextTransformer
from embiggen import Embiggen
from embiggen import Shelldump, History
from embiggen import CooccurrenceEncoder

# read a corpise of texts
path = '/home/peter/data/embiggen/hardy.txt'
path = '/home/peter/data/embiggen/Emails.csv'
#path='/Users/robinp/Documents/data/return-native.txt'
encoder = TextTransformer(path, data_type="words")
tensor_data, count_list, dictionary, reverse_dictionary = encoder.build_dataset()
print("Done reading data, got n={} words".format(len(count_list)))

print("Starting embiggen fit")
# do embedding
embiggen = Embiggen()

hist = History()

embiggen.fit(
        data=tensor_data,
        worddict=dictionary,
        reverse_worddict=reverse_dictionary,
        embedding_method = "skipgram",
        epochs = 10,
        embedding_size = 20,
        callbacks=[Shelldump(), hist]
    )

history = hist.get_loss_history()

with open("loss.txt", "wt") as f:
    for l in history:
        f.write("%f\n" % l)
