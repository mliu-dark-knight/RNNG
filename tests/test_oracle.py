import pytest
from nltk.tree import Tree

from rnng.actions import NT, REDUCE, SHIFT
from rnng.oracle import Oracle


class TestOracle:
    def test_init(self):
        actions = [NT('S'), SHIFT]
        pos_tags = ['NNP']
        words = ['John']

        oracle = Oracle(actions, pos_tags, words)

        assert oracle.actions == actions
        assert oracle.pos_tags == pos_tags
        assert oracle.words == words

    def test_init_with_unequal_shift_count_and_number_of_words(self):
        actions = [NT('S')]
        pos_tags = ['NNP']
        words = ['John']
        with pytest.raises(ValueError) as excinfo:
            Oracle(actions, pos_tags, words)
        assert 'number of words should match number of SHIFT actions' in str(excinfo.value)

    def test_init_with_unequal_number_of_words_and_pos_tags(self):
        actions = [NT('S'), SHIFT]
        pos_tags = ['NNP', 'VBZ']
        words = ['John']
        with pytest.raises(ValueError) as excinfo:
            Oracle(actions, pos_tags, words)
        assert 'number of POS tags should match number of words' in str(excinfo.value)

    def test_from_tree(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = [
            NT('S'),
            NT('NP'),
            SHIFT,
            REDUCE,
            NT('VP'),
            SHIFT,
            NT('NP'),
            SHIFT,
            REDUCE,
            REDUCE,
            REDUCE,
        ]
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']
        expected_words = ['John', 'loves', 'Mary']

        oracle = Oracle.from_tree(Tree.fromstring(s))

        assert isinstance(oracle, Oracle)
        assert oracle.actions == expected_actions
        assert oracle.pos_tags == expected_pos_tags
        assert oracle.words == expected_words

    def test_to_tree(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        actions = [
            NT('S'),
            NT('NP'),
            SHIFT,
            REDUCE,
            NT('VP'),
            SHIFT,
            NT('NP'),
            SHIFT,
            REDUCE,
            REDUCE,
            REDUCE,
        ]
        pos_tags = ['NNP', 'VBZ', 'NNP']
        words = ['John', 'loves', 'Mary']

        oracle = Oracle(actions, pos_tags, words)

        assert str(oracle.to_tree()) == s

