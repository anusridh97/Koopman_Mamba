"""
multihop_task.py -- variable-binding multi-hop NIAH task (verified: 2000/2000
independent-resolver agreement). Symbolic chain  v0=lit; v1=v0; ...; vk=v(k-1)
plus distractor assignments, all shuffled; query resolves vk to the root literal.
"""
import numpy as np

PAD, SEP, ASSIGN, QUERY = 0, 1, 2, 3
NVAR, NVAL = 96, 48
VAR0 = 4
VAL0 = VAR0 + NVAR
VOCAB = VAL0 + NVAL

def var_tok(i): return VAR0 + i
def val_tok(v): return VAL0 + v

def make_example(rng, hop, n_distract, gap=0):
    perm = rng.permutation(NVAR)
    chain_vars = perm[:hop + 1]; distract_vars = perm[hop + 1:]
    root_val = int(rng.integers(NVAL))
    lines = [(chain_vars[0], val_tok(root_val))]
    for i in range(1, hop + 1):
        lines.append((chain_vars[i], var_tok(chain_vars[i - 1])))
    for j in range(n_distract):
        lhs = distract_vars[j % len(distract_vars)]
        if rng.random() < 0.5 or j == 0:
            rhs = val_tok(int(rng.integers(NVAL)))
        else:
            rhs = var_tok(distract_vars[int(rng.integers(min(j, len(distract_vars))))])
        lines.append((lhs, rhs))
    rng.shuffle(lines)
    toks = []
    for (lhs, rhs) in lines:
        toks += [var_tok(lhs), ASSIGN, rhs, SEP] + [SEP] * gap
    toks += [QUERY, var_tok(chain_vars[hop]), ASSIGN]
    answer_pos = len(toks) - 1
    toks += [val_tok(root_val)]
    return [int(t) for t in toks], answer_pos, int(val_tok(root_val)), hop

def resolve(tokens):
    binding = {}; i = 0; qv = None
    while i < len(tokens):
        t = tokens[i]
        if t == QUERY: qv = tokens[i + 1] - VAR0; break
        if VAR0 <= t < VAL0 and i + 2 < len(tokens) and tokens[i + 1] == ASSIGN:
            binding[t - VAR0] = tokens[i + 2]; i += 3
            while i < len(tokens) and tokens[i] == SEP: i += 1
        else: i += 1
    cur = qv
    for _ in range(NVAR + 1):
        rhs = binding.get(cur)
        if rhs is None: return None
        if rhs >= VAL0: return rhs
        cur = rhs - VAR0
    return None

def selftest():
    rng = np.random.default_rng(0); ok = 0; n = 2000
    for _ in range(n):
        toks, ap, ans, _ = make_example(rng, int(rng.integers(1, 8)), int(rng.integers(0, 40)))
        assert toks[ap + 1] == ans and 0 <= min(toks) and max(toks) < VOCAB
        ok += (resolve(toks) == ans)
    print(f"resolver agrees {ok}/{n}; VOCAB={VOCAB}"); print("PASS" if ok == n else "FAIL")

if __name__ == '__main__':
    selftest()
