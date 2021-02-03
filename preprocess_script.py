import os
import numpy as np
import sympy as sp
import torch

from src.utils import AttrDict
from src.envs import build_env
from src.model import build_modules

from src.utils import to_cuda
from src.envs.sympy_utils import simplify

OPERATORS = {
        # Elementary functions
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        'rac': 2,
        'inv': 1,
        'pow2': 1,
        'pow3': 1,
        'pow4': 1,
        'pow5': 1,
        'sqrt': 1,
        'exp': 1,
        'ln': 1,
        'abs': 1,
        'sign': 1,
        # Trigonometric Functions
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'cot': 1,
        'sec': 1,
        'csc': 1,
        # Trigonometric Inverses
        'asin': 1,
        'acos': 1,
        'atan': 1,
        'acot': 1,
        'asec': 1,
        'acsc': 1,
        # Hyperbolic Functions
        'sinh': 1,
        'cosh': 1,
        'tanh': 1,
        'coth': 1,
        'sech': 1,
        'csch': 1,
        # Hyperbolic Inverses
        'asinh': 1,
        'acosh': 1,
        'atanh': 1,
        'acoth': 1,
        'asech': 1,
        'acsch': 1,
        # Derivative
        'derivative': 2,
        # custom functions
        'f': 1,
        'g': 2,
        'h': 3,
    }

symbols = ['I', 'INT+', 'INT-', 'INT', 'FLOAT', '-', '.', '10^', 'Y', "Y'", "Y''"]


constants = ['pi', 'E']
variables = ['x', 'y', 'z', 't']
functions = ['f', 'g', 'h']
elements = [str(i) for i in range(-10, 10)]
coefficients = [f'a{i}' for i in range(10)]


no_child_symbols = constants + variables + functions + elements + coefficients


from tqdm import tqdm
import queue

def get_ancestors(exp_list, exp_len):
    q = queue.LifoQueue()
    q.put(-1)                            # so last element gets this parent but doesn't save it

    ancestors = {0: []}
    node2parent = {}
    levels = {0: -1}

    parent = 0
    for i in range(exp_len):
        op_now = exp_list[i]

        node2parent[i] = parent
        try:
            levels[i] = levels[parent] + 1
        except:
            print('you are in except')
            return {}, {}

        if op_now in OPERATORS or op_now in symbols:   # <=> node has children
            if op_now in OPERATORS and OPERATORS[op_now] == 2:    # <=> node has 2 children
                q.put(i)
            parent = i
        elif op_now in no_child_symbols:
            if op_now.isdigit() and i + 1 < exp_len and exp_list[i + 1].isdigit():   # e.x. 18
                parent = i
            else:
                parent = q.get()
        else:
            print(op_now)
            #raise(NotFound)
            return {}, {}
        ancestors[i] = [i] + ancestors[node2parent[i]]

    return ancestors, levels


def get_path(i, j):
    if i == j:
        return "<self>"
    anc_i = set(ancestors[i])
      
    for node in ancestors[j][-(levels[i] + 1) :]:
        if node in anc_i:
            up_n = levels[i] - levels[node]
            down_n = levels[j] - levels[node]
            return str(round(up_n + 0.001 * down_n, 5))


def get_ud_masks(ancestors, levels, exp_len):
    path_rels = []
    for i in range(exp_len):
        path_rels.append(" ".join([get_path(i, j) for j in range(exp_len)]))
    
    return path_rels




import json
import jsonlines

for set_name in ['test', 'valid', 'train']:
    with open('data/prim_fwd.' + set_name, 'r') as expressions:
        with jsonlines.open('data/rel_matrix_'+set_name+'.jsonl', 'w') as rel_matrix_json:
            for i, line in tqdm(enumerate(expressions)):
                #print(line)
                qa = line.split('|')[1].split('\t')
                if len(qa) == 2:
                    q, a = qa
                else:
                    print(i,'is broken')
                    continue
                #print(q, ';', a)
                
                q = q.split()
                #a = a.split()
                
                ancestors, levels = get_ancestors(q, len(q))
                if len(ancestors) == 0:
                    print(i, 'is broken; see previous line')
                    continue
                rel_matrix_q = get_ud_masks(ancestors, levels, len(q))

                #ancestors, levels = get_ancestors(a, len(a))
                #rel_matrix_a = get_ud_masks(ancestors, levels, len(a))
                
                rel_matrix_json.write(json.dumps(rel_matrix_q, indent=0))
