import json
from typing import List, Dict, Iterable

MULTI_CHAR_ATOMS = [
    "Cl", "Br", "Si", "Na", "Li", "Ca", "Mg", "Al", "Sn", "Sb", "Se",
]

SINGLE_CHAR_TOKENS = set(
    list("BCNOFPSIH")
    + list("cnosp")
    + list("0123456789")
    + ["*", "=", "-", "#", "/", "\\", "(", ")", "."]
)


class SmilesTokenizer:
    def __init__(self, vocab: Dict[str, int], special_tokens: Dict[str, str], max_length: int = 128):
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.special_tokens = special_tokens
        self.max_length = max_length
        self.pad_id = vocab[special_tokens["pad"]]
        self.mask_id = vocab[special_tokens["mask"]]
        self.bos_id = vocab[special_tokens["bos"]]
        self.eos_id = vocab[special_tokens["eos"]]
        self.unk_id = vocab[special_tokens["unk"]]

    @staticmethod
    def tokenize(smiles: str) -> List[str]:
        tokens = []
        i = 0
        n = len(smiles)
        while i < n:
            ch = smiles[i]
            if ch == "[":
                j = smiles.find("]", i + 1)
                if j == -1:
                    tokens.append(smiles[i:])
                    break
                tokens.append(smiles[i : j + 1])
                i = j + 1
                continue
            if ch == "%" and i + 2 < n and smiles[i + 1 : i + 3].isdigit():
                tokens.append(smiles[i : i + 3])
                i += 3
                continue
            matched = False
            for atom in MULTI_CHAR_ATOMS:
                if smiles.startswith(atom, i):
                    tokens.append(atom)
                    i += len(atom)
                    matched = True
                    break
            if matched:
                continue
            tokens.append(ch)
            i += 1
        return tokens

    def detokenize(self, tokens: Iterable[str]) -> str:
        out = []
        for tok in tokens:
            if tok in self.special_tokens.values():
                continue
            out.append(tok)
        return "".join(out)

    def encode(self, smiles: str, add_special: bool = True, pad: bool = True) -> List[int]:
        toks = self.tokenize(smiles)
        ids = [self.vocab.get(t, self.unk_id) for t in toks]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if pad:
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]
            else:
                ids = ids + [self.pad_id] * (self.max_length - len(ids))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = [self.id_to_token.get(i, self.special_tokens["unk"]) for i in ids]
        return self.detokenize(toks)

    def to_json(self, path: str) -> None:
        payload = {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "max_length": self.max_length,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "SmilesTokenizer":
        with open(path, "r") as f:
            payload = json.load(f)
        return cls(payload["vocab"], payload["special_tokens"], payload["max_length"])


def build_vocab(smiles_iter: Iterable[str], special_tokens: Dict[str, str]) -> Dict[str, int]:
    vocab = {}
    for tok in special_tokens.values():
        vocab[tok] = len(vocab)
    for smiles in smiles_iter:
        for tok in SmilesTokenizer.tokenize(smiles):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def count_oov(smiles_iter: Iterable[str], vocab: Dict[str, int]) -> (int, int):
    total = 0
    oov = 0
    for smiles in smiles_iter:
        toks = SmilesTokenizer.tokenize(smiles)
        total += len(toks)
        for t in toks:
            if t not in vocab:
                oov += 1
    return oov, total
