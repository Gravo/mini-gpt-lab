# 025 Text-Split Tokenizer Comparison

## Goal

Recheck the tokenizer/context comparison with a fairer validation split.

Earlier experiments split after tokenization:

```text
text -> tokens -> 90/10 token split
```

This makes char and BPE validation boundaries slightly different in raw text. This experiment adds:

```yaml
split: text
```

which does:

```text
text -> 90/10 raw text split -> tokenize train and val separately
```

## Tokenizer Policy

The tokenizer is still trained from the full corpus. This avoids unknown characters in validation.
The last 10% contains 145 characters not present in the first 90%, so train-only tokenizer fitting
would require an `<unk>` path that this lab does not yet have.

This means 025 primarily fixes validation boundary comparability, not tokenizer-training leakage.

## Runs

Run A:

```text
char tokenizer, block 128
config = configs/hongloumeng_80_char_width256_rope_textsplit.yaml
```

Run B:

```text
char tokenizer, block 192
config = configs/hongloumeng_80_char_width256_rope_context192_textsplit.yaml
```

Run C:

```text
BPE6000 tokenizer, block 192
config = configs/hongloumeng_80_bpe6000_width256_rope_context192_textsplit.yaml
```

All runs use `max_steps=2500` because prior best checkpoints landed before or around step 2250.

## Results

| Run | Tokenizer | Block | Best Step | Best Val Loss | Best Val Nats/Char | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | char | 128 | 2250 | 4.0805 | 4.0805 | same as 016 |
| B | char | 192 | 1750 | 4.0909 | 4.0909 | same as 020 |
| C | BPE6000 | 192 | 1250 | 6.1351 | 4.1810 | worse than 024 token-split |

## BPE Text-Split Token Stats

```text
train chars = 531,388
train BPE tokens = 355,852
train tokens/char = 0.6697

val chars = 59,044
val BPE tokens = 40,238
val tokens/char = 0.6815
```

The validation text is slightly less compressible than the training text under BPE6000. This makes
the raw-text split stricter for BPE than the earlier token split.

Encoded cache files:

```text
data/tokenizers/hongloumeng_80_bpe6000_train_textsplit_ids.pt = 2,848,331 bytes
data/tokenizers/hongloumeng_80_bpe6000_val_textsplit_ids.pt   =   323,329 bytes
```

## Interpretation Questions

```text
Does text-split preserve 016 as best?
Does context 192 become better or worse under raw text split?
Does BPE6000+context192 remain behind char tokenizer?
```

Answers:

```text
Yes, 016 remains best.
Char context 192 is unchanged because char tokens equal raw characters.
BPE6000+context192 falls further behind under raw text split.
```

## Comparison Against Earlier Token-Split Runs

Char tokenizer runs are unchanged:

```text
016 token-split char block128: 4.0805
025A text-split char block128: 4.0805

020 token-split char block192: 4.0909
025B text-split char block192: 4.0909
```

BPE6000 context192 gets worse:

```text
024 token-split BPE6000 block192: 4.1163
025C text-split BPE6000 block192: 4.1810
delta:                            +0.0648
```

This means the previous BPE comparison was mildly optimistic. Once validation is split by raw text,
the current BPE6000 tokenizer no longer looks competitive with the char tokenizer.

## Conclusion

The original ranking is strengthened:

```text
025A char block128:       4.0805
025B char block192:       4.0909
025C BPE6000 block192:    4.1810
```

For this corpus size and 6M-parameter model, character tokenization remains the strongest loss
baseline. BPE is still useful as a learning experiment because it changes generation texture, but it
does not yet improve validation loss.

The next tokenizer experiment should not increase BPE vocab further. Better options:

```text
BPE4500 + text split
or add an <unk> path and train tokenizer only on train text
or use a stronger pretrained Chinese tokenizer for comparison
```

## Commands

```powershell
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_textsplit.yaml
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_char_width256_rope_context192_textsplit.yaml
D:\projects\yolov9-main\yolov9\.venv\Scripts\python.exe -m gptlab.train --config configs\hongloumeng_80_bpe6000_width256_rope_context192_textsplit.yaml
```
