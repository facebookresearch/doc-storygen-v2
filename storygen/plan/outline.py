# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Sequence
from functools import partial
import string
import uuid

from storygen.common.llm.llm import *
from storygen.common.util import *

class OutlineNode(Sequence):
    @staticmethod
    def from_dict(d, parent=None):
        node = OutlineNode(d['text'], parent, d['scene'], d['entities'], d['id'])
        node.children = [OutlineNode.from_dict(child, node) for child in d['children']]
        return node

    @staticmethod
    def num_converter(depth):
        if depth == 0:
            return lambda num: '' # assume the root node should be empty
        if depth % 3 == 1:
            return str
        elif depth % 3 == 2:
            return num_to_char
        elif depth % 3 == 0:
            return num_to_roman
    
    @staticmethod
    def indent(depth):
        if depth == 0:
            return ''
        return '\t' * (depth-1)

    def __init__(self, text, parent, scene='', entities=None, id=None):
        self.text = text.strip()
        self.entities = entities if entities is not None else []
        self.scene = scene
        self.children = []
        self.parent = parent
        self.id = str(uuid.uuid4()) if id is None else id
        super().__init__()
    
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id
    
    def to_dict(self):
        return {
            'text': self.text,
            'scene': self.scene,
            'entities': self.entities,
            'children': [child.to_dict() for child in self.children],
            'id': self.id
        }
    
    def format_self(self):
        s = self.number() + self.text
        if len(self.scene) > 0:
            s += ' Scene: ' + self.scene 
        if len(self.entities) > 0:
            s += ' Characters: ' + ', '.join(self.entities)
        return s
    
    def __str__(self, include_self=True):
        ordered_nodes = [node for node in self.depth_first_traverse(include_self=include_self)]
        return '\n\n'.join([node.format_self() for node in ordered_nodes]).strip()
    
    def __len__(self):
        return len(self.children)
    
    def __getitem__(self, index):
        return self.children[index]

    def get_node_by_id(self, id):
        for node in self.root().depth_first_traverse():
            if node.id == id:
                return node
        return None
        
    def number(self, depth_shift=0, lookforward=0, convert=True):
        if self.parent is None:
            num = 1
        else:
            num = self.parent.children.index(self) + 1
        num += lookforward
        if convert:
            depth = self.depth() + depth_shift
            if depth == 0:
                return ''
            else:
                return '\t' * (depth-1) + OutlineNode.num_converter(depth)(num) + '. '
        else:
            return num

    def depth(self):
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.depth()

    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()
    
    def predecessor(self, max_depth=1e8):
        nodes = list(self.root().depth_first_traverse(max_depth=max_depth))
        return nodes[nodes.index(self)-1] if nodes.index(self) > 0 else None
    
    def successor(self, max_depth=1e8):
        nodes = list(self.root().depth_first_traverse(max_depth=max_depth))
        return nodes[nodes.index(self)+1] if nodes.index(self) < len(nodes)-1 else None

    def ancestors(self, include_self=False):
        if self.parent is None:
            return [self] if include_self else []
        else:
            return self.parent.ancestors(include_self=True) + ([self] if include_self else [])
    
    def siblings(self, include_self=False):
        if self.parent is None:
            return []
        else:
            return [child for child in self.parent.children if (include_self or child != self)]
    
    def leaves(self):
        if len(self.children) == 0:
            return [self]
        else:
            return sum([child.leaves() for child in self.children], [])
    
    def depth_first_traverse(self, include_self=True, max_depth=1e8):
        if self.depth() <= max_depth and include_self:
            yield self
        for child in self.children:
            yield from child.depth_first_traverse(max_depth=max_depth)
    
    def breadth_first_traverse(self, include_self=True, max_depth=1e8):
        if self.depth() <= max_depth and include_self:
            yield self
        if self.depth() < max_depth:
            queue = [c for c in self.children]
            while len(queue) > 0:
                yield queue[0]
                if queue[0].depth() < max_depth:
                    queue += [c for c in queue[0].children]
                queue = queue[1:]
    
    def context(self, context_type):
        if context_type == 'full':
            selected_nodes = set(list(self.root().depth_first_traverse(include_self=False)))
        elif context_type == 'ancestors':
            selected_nodes = set(list(self.ancestors(include_self=False)))
        elif context_type == 'ancestors-with-siblings':
            ancestors = list(self.ancestors(include_self=True))
            selected_nodes = set(sum([ancestor.siblings(include_self=True) for ancestor in ancestors], []))
        elif context_type == 'ancestors-with-siblings-children':
            ancestors = list(self.ancestors(include_self=True))
            ancestors_with_siblings = sum([ancestor.siblings(include_self=True) for ancestor in ancestors], [])
            selected_nodes = set(ancestors_with_siblings + sum([node.children for node in ancestors_with_siblings], []))
        else:
            raise NotImplementedError(f"Outline expansion context type {context_type} not implemented.")
        prefix_nodes = []
        suffix_nodes = []
        in_prefix = True
        for node in self.root().depth_first_traverse(include_self=False):
            if node == self:
                in_prefix = False
            elif node in selected_nodes:
                if in_prefix:
                    prefix_nodes.append(node)
                else:
                    suffix_nodes.append(node)
        return '\n\n'.join([node.format_self() for node in prefix_nodes]), '\n\n'.join([node.format_self() for node in suffix_nodes])