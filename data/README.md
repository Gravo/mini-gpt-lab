# Data

Put local datasets here. For the first experiment, download Tiny Shakespeare as:

```powershell
Invoke-WebRequest https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile data/tiny_shakespeare.txt
```

For the Hongloumeng character baseline, download public-domain Wikisource text as:

```powershell
python scripts/download_hongloumeng.py --chapters 10 --out data/hongloumeng_10.txt
```

For the full 120-chapter corpus used in experiment 031:

```powershell
python scripts/download_hongloumeng.py --chapters 120 --out data/hongloumeng_120.txt --sleep 0.4 --retries 5
```

Dataset files are intentionally not committed.
