from PYEVALB import scorer, summary
from nltk.tree import Tree


def add_dummy_pos(tree):
	if not isinstance(tree, Tree):
		return Tree('XX', [tree])
	return Tree(tree.label(), [add_dummy_pos(t) for t in tree])


def id2parsetree(tree, id2nonterm, id2word):
	if not isinstance(tree, Tree):
		return id2word[tree]
	children = [id2parsetree(t, id2nonterm, id2word) for t in tree]
	return Tree(id2nonterm[tree.label()], children)


def compute_f1(f_gold, f_test):
	try:
		f1 = summary.summary(scorer.Scorer().score_corpus(f_gold, f_test)).bracker_fmeasure
		return f1
	except ZeroDivisionError:
		return 0.0
