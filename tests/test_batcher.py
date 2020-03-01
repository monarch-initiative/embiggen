from unittest import TestCase

from xn2v import CBOWListBatcher
from xn2v import SkipGramBatcher


class TestCBOWListBatcher(TestCase):
    def setUp(self):
        list1 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        list2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        list3 = [112, 114, 116, 118, 1110, 1112, 1114, 1116, 1118, 1120]
        data = [list1, list2, list3]
        self.batcher = CBOWListBatcher(data, window_size=1, sentences_per_batch=1)

    def test_ctor(self):
        self.assertIsNotNone(self.batcher)
        self.assertEqual(0, self.batcher.sentence_index)
        self.assertEqual(0, self.batcher.word_index)
        self.assertEqual(1, self.batcher.window_size)

        # span is 2*window_size +1
        self.assertEqual(3, self.batcher.span)
        self.assertEqual(3, self.batcher.sentence_count)
        self.assertEqual(10, self.batcher.sentence_len)

        # batch size is calculated as (sentence_len - span + 1)=10-3+1
        self.assertEqual(8, self.batcher.batch_size)

    def test_generate_batch(self):
        # The first batch is from the window [1, 3, 5] and [3, 5, 7]
        batch, labels = self.batcher.generate_batch()
        batch_shape = (8, 2)
        self.assertEqual(batch.shape, batch_shape)

        label_shape = (8, 1)
        self.assertEqual(labels.shape, label_shape)

        # batch represents the context words [[1,5],[3,7]]
        # label represents the center words [[3],[5]]
        self.assertEqual(1, batch[0][0])
        self.assertEqual(5, batch[0][1])
        self.assertEqual(3, labels[0])

        # now the second example
        self.assertEqual(3, batch[1][0])
        self.assertEqual(7, batch[1][1])
        self.assertEqual(5, labels[1])

        # get another batch. We expect the second list
        batch, labels = self.batcher.generate_batch()
        self.assertEqual(batch.shape, batch_shape)
        self.assertEqual(2, batch[0][0])
        self.assertEqual(6, batch[0][1])
        self.assertEqual(4, labels[0])

        # get another batch. We expect the third list
        batch, labels = self.batcher.generate_batch()
        self.assertEqual(batch.shape, batch_shape)
        self.assertEqual(112, batch[0][0])
        self.assertEqual(116, batch[0][1])
        self.assertEqual(114, labels[0])

        # get another batch. We expect to go back to the first list
        batch, labels = self.batcher.generate_batch()
        self.assertEqual(batch.shape, batch_shape)
        self.assertEqual(1, batch[0][0])
        self.assertEqual(5, batch[0][1])
        self.assertEqual(3, labels[0])



class TestSkipGramBatcher(TestCase):
    """
    Test the batcher function for SkipGrams with input data being one long list of integers
    usually representing a text.
    """
    def setUp(self):
        list1 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        list2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        list3 = [112, 114, 116, 118, 1110, 1112, 1114, 1116, 1118, 1120]
        data = [list1, list2, list3]
        data = [item for sublist in data for item in sublist]
        # ctor has batch_size, num_skips, skip_window
        batch_size = 8
        num_skips = 1
        skip_window = 1
        self.batcher = SkipGramBatcher(data, batch_size=batch_size, num_skips=num_skips, skip_window=skip_window)

    def test_ctor(self):
        self.assertIsNotNone(self.batcher)
        self.assertEqual(0, self.batcher.data_index)
        self.assertEqual(1, self.batcher.skip_window)
        # span is 2*window_size +1
        self.assertEqual(3, self.batcher.span)
        # batch size is calculated as (sentence_len - span + 1)=10-3+1
        self.assertEqual(8, self.batcher.batch_size)
