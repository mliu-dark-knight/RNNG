from typing import List, Sequence

from nltk.tree import Tree

from rnng.actions import NT, REDUCE, SHIFT, get_nonterm, is_nt
from rnng.typing_ import Action, POSTag, Word


class Oracle(object):
    def __init__(self,
                 actions: Sequence[Action],
                 pos_tags: Sequence[POSTag],
                 words: Sequence[Word]) -> None:
        shift_cnt = sum(1 if a == SHIFT else 0 for a in actions)
        if len(words) != shift_cnt:
            raise ValueError('number of words should match number of SHIFT actions')
        if len(pos_tags) != len(words):
            raise ValueError('number of POS tags should match number of words')

        self._actions = actions
        self._pos_tags = pos_tags
        self._words = words

    @property
    def actions(self) -> List[Action]:
        return list(self._actions)

    @property
    def pos_tags(self) -> List[POSTag]:
        return list(self._pos_tags)

    @property
    def words(self) -> List[Word]:
        return list(self._words)

    def to_tree(self) -> Tree:
        stack = []
        pos_tags = list(reversed(self.pos_tags))
        words = list(reversed(self.words))
        for a in self.actions:
            if is_nt(a):
                stack.append(get_nonterm(a))
            elif a == REDUCE:
                children = []
                while stack and isinstance(stack[-1], Tree):
                    children.append(stack.pop())
                if not children or not stack:
                    raise ValueError(
                        f'invalid {REDUCE} action, please check if the actions are correct')
                parent = stack.pop()
                tree = Tree(parent, list(reversed(children)))
                stack.append(tree)
            else:
                tree = Tree(pos_tags.pop(), [words.pop()])
                stack.append(tree)
        if len(stack) != 1:
            raise ValueError('actions do not produce a single parse tree')
        return stack[0]


    @classmethod
    def from_tree(cls, tree: Tree) -> 'Oracle':
        actions = cls.get_actions(tree)
        words, pos_tags = zip(*tree.pos())
        return cls(actions, list(pos_tags), list(words))

    @classmethod
    def get_action_at_pos_node(cls, pos_node: Tree) -> Action:
        if len(pos_node) != 1 or isinstance(pos_node[0], Tree):
            raise ValueError('input is not a valid POS node')
        return SHIFT

    @classmethod
    def get_actions(cls, tree: Tree) -> List[Action]:
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return [cls.get_action_at_pos_node(tree)]

        actions: List[Action] = [NT(tree.label())]
        for child in tree:
            actions.extend(cls.get_actions(child))
        actions.append(REDUCE)
        return actions

