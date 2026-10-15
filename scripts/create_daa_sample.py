"""Create a rating-balanced annotation sample without modifying the source CSV."""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "DAA_M07_CS2_data.csv"
OUTPUT = ROOT / "data" / "daa_reviews_sample.csv"
SEED = 42
PER_RATING = 20


def main():
    groups = defaultdict(list)
    seen = set()
    counts = Counter()

    with SOURCE.open(encoding="cp1252", newline="") as handle:
        for row in csv.DictReader(handle):
            counts["input"] += 1
            title = " ".join(row["title"].split())
            body = " ".join(row["body"].split())
            if not body:
                counts["empty_body"] += 1
                continue
            # Deduplicate by body, including repeated bodies with different titles.
            key = body.casefold()
            if key in seen:
                counts["duplicate_body"] += 1
                continue
            if row["rating"] not in {"1", "2", "3", "4", "5"}:
                counts["invalid_rating"] += 1
                continue
            seen.add(key)
            groups[row["rating"]].append({
                "review_id": row["url"].rstrip("/").rsplit("/", 1)[-1],
                "text": f"{title}\n\n{body}",
                "source": "DAA_M07_CS2_data.csv / ProductReview.com.au",
                "gold_sentiment": "",
                "is_test": "False",
                "rating": row["rating"],
                "pubDate": row["pubDate"],
            })

    rng = random.Random(SEED)
    sample = []
    for rating in ("1", "2", "3", "4", "5"):
        if len(groups[rating]) < PER_RATING:
            raise ValueError(f"Not enough usable reviews with rating {rating}")
        sample.extend(rng.sample(groups[rating], PER_RATING))
    rng.shuffle(sample)
    assert len({row["review_id"] for row in sample}) == len(sample)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)

    print("Audit:", dict(counts))
    print("Usable rows:", sum(map(len, groups.values())))
    print("Sample rating counts:", dict(Counter(row["rating"] for row in sample)))
    print(f"Saved {len(sample)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
