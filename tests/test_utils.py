from nltk.tree import Tree

from rnng.utils import add_dummy_pos, id2parsetree

id2nonterm = 'S NP VP'.split()
id2word = 'John loves Mary'.split()


def test_id2parsetree():
    tree = Tree(0, [
        Tree(1, [0]),
        Tree(2, [1, Tree(1, [2])])
    ])
    expected = '(S (NP John) (VP loves (NP Mary)))'

    assert str(id2parsetree(tree, id2nonterm, id2word)) == expected


def test_add_dummy_pos():
    s = '(S (NP John) (VP loves (NP Mary)))'
    expected = '(S (NP (XX John)) (VP (XX loves) (NP (XX Mary))))'
    tree = Tree.fromstring(s)

    assert str(add_dummy_pos(tree)) == expected

