import argparse
import html
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SOURCE = "https://zh.wikisource.org/wiki/紅樓夢"
RAW_ENDPOINT = "https://zh.wikisource.org/w/index.php?title={title}&action=raw"
NAVIGATION_LINE = re.compile(r"^[\s　]*(上一回|下一回|回目录|回目錄|回)[\s　]*(上一回|下一回|回目录|回目錄|回)?[\s　]*$")


def fetch_raw_page(title: str, timeout: int = 30) -> str:
    url = RAW_ENDPOINT.format(title=quote(title))
    request = Request(
        url,
        headers={
            "User-Agent": "mini-gpt-lab/0.1 (educational char-level language model experiment)"
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def strip_templates(text: str) -> str:
    out = []
    depth = 0
    i = 0
    while i < len(text):
        pair = text[i : i + 2]
        if pair == "{{":
            depth += 1
            i += 2
            continue
        if pair == "}}" and depth:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def clean_wikitext(raw: str) -> str:
    raw = re.sub(r"<ref[^>/]*>.*?</ref>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<ref[^>]*/>", "", raw)
    raw = re.sub(r"^.*\[\[\.\./.*$", "", raw, flags=re.MULTILINE)
    text = strip_templates(raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"-\{([^{}]+)\}-", r"\1", text)
    text = re.sub(r"\[\[[^|\]]+\|([^]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^]]+)\]\]", r"\1", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"^\s*[-]{4,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\[\[Category:.*$", "", text, flags=re.MULTILINE)
    text = "\n".join(line for line in text.splitlines() if not NAVIGATION_LINE.match(line))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return html.unescape(text).strip()


def download_chapter(chapter: int) -> str:
    title = f"紅樓夢/第{chapter:03d}回"
    raw = fetch_raw_page(title)
    cleaned = clean_wikitext(raw)
    if not cleaned:
        raise RuntimeError(f"Downloaded empty chapter: {title}")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("data/hongloumeng_10.txt"))
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if args.chapters < 1 or args.chapters > 120:
        raise SystemExit("--chapters must be between 1 and 120")

    parts = []
    for chapter in range(1, args.chapters + 1):
        print(f"Downloading chapter {chapter}/{args.chapters}")
        for attempt in range(1, args.retries + 1):
            try:
                parts.append(download_chapter(chapter))
                break
            except (HTTPError, URLError) as exc:
                if attempt == args.retries:
                    raise SystemExit(f"Failed to download chapter {chapter}: {exc}") from exc
                wait = args.sleep * attempt * 3
                print(f"Retrying chapter {chapter} after error: {exc}. Waiting {wait:.1f}s")
                time.sleep(wait)
        time.sleep(args.sleep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")

    source_note = args.out.with_suffix(".source.md")
    source_note.write_text(
        "\n".join(
            [
                "# Hongloumeng Dataset Source",
                "",
                f"- Source: {SOURCE}",
                "- License/status: public-domain source text as hosted by Wikisource.",
                f"- Chapters: 1-{args.chapters}",
                f"- Output text: `{args.out.name}`",
                "- Note: use the original public-domain text, not a modern annotated or translated edition.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {source_note}")


if __name__ == "__main__":
    main()
