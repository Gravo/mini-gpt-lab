from collections import Counter


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_state(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        tokenizer = cls("")
        tokenizer.chars = list(state["chars"])
        tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
        tokenizer.itos = {i: ch for ch, i in tokenizer.stoi.items()}
        return tokenizer


class BPETokenizer:
    def __init__(
        self,
        text: str,
        vocab_size: int = 8000,
        min_pair_freq: int = 2,
        merges: list[tuple[int, int]] | None = None,
        pieces: list[str] | None = None,
    ):
        self.min_pair_freq = min_pair_freq
        if pieces is not None and merges is not None:
            self.pieces = list(pieces)
            self.stoi = {piece: i for i, piece in enumerate(self.pieces)}
            self.merges = [tuple(pair) for pair in merges]
            self.merge_to_id = {pair: i + len(self.pieces) - len(self.merges) for i, pair in enumerate(self.merges)}
            self._cached_text = None
            self._cached_ids = None
            return

        chars = sorted(set(text))
        self.pieces = chars[:]
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.merges: list[tuple[int, int]] = []
        self.merge_to_id: dict[tuple[int, int], int] = {}
        ids = [self.stoi[ch] for ch in text]

        while len(self.pieces) < vocab_size:
            pair_counts = Counter(zip(ids, ids[1:]))
            if not pair_counts:
                break
            pair, count = pair_counts.most_common(1)[0]
            if count < min_pair_freq:
                break
            new_id = len(self.pieces)
            self.merges.append(pair)
            self.merge_to_id[pair] = new_id
            self.pieces.append(self.pieces[pair[0]] + self.pieces[pair[1]])
            ids = self._merge_ids(ids, pair, new_id)
        self._cached_text = text
        self._cached_ids = ids

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    @staticmethod
    def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def encode(self, text: str) -> list[int]:
        if text == self._cached_text and self._cached_ids is not None:
            return list(self._cached_ids)
        ids = [self.stoi[ch] for ch in text]
        for pair in self.merges:
            new_id = self.merge_to_id[pair]
            ids = self._merge_ids(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[i] for i in ids)

    def to_state(self) -> dict:
        return {
            "type": "bpe",
            "pieces": self.pieces,
            "merges": self.merges,
            "min_pair_freq": self.min_pair_freq,
        }

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        return cls(
            "",
            min_pair_freq=state.get("min_pair_freq", 2),
            pieces=state["pieces"],
            merges=[tuple(pair) for pair in state["merges"]],
        )


def build_tokenizer(text: str, cfg: dict | None = None) -> CharTokenizer | BPETokenizer:
    cfg = cfg or {"type": "char"}
    kind = cfg.get("type", "char")
    if kind == "char":
        return CharTokenizer(text)
    if kind == "bpe":
        return BPETokenizer(
            text,
            vocab_size=cfg.get("vocab_size", 8000),
            min_pair_freq=cfg.get("min_pair_freq", 2),
        )
    raise ValueError(f"Unknown tokenizer: {kind}")


def tokenizer_from_state(state: dict) -> CharTokenizer | BPETokenizer:
    kind = state.get("type", "char")
    if kind == "char":
        return CharTokenizer.from_state(state)
    if kind == "bpe":
        return BPETokenizer.from_state(state)
    raise ValueError(f"Unknown tokenizer state: {kind}")
