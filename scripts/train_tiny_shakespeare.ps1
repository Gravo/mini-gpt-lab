param(
  [string]$Config = "configs/tiny_shakespeare.yaml"
)

python -m gptlab.train --config $Config
