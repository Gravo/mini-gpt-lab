param(
  [string]$Checkpoint = "runs/tiny_shakespeare_baseline/ckpt.pt",
  [string]$Prompt = "To be, or not to be",
  [int]$MaxNewTokens = 200
)

python -m gptlab.generate --checkpoint $Checkpoint --prompt $Prompt --max-new-tokens $MaxNewTokens
