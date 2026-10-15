"""One-time AI draft annotation bootstrap. Never overwrites human annotations.

Drafts below were assigned by reading review text before running VADER/Jev.
The first 50 seed-42 shuffled sample rows were selected without model scores.
"""
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from insurance_sentiment_analysis import clean_text

# Review ID prefix guards against accidental source reordering. Notes explain
# ambiguous decisions; every draft requires independent human confirmation.
DRAFTS = [
    ("84567114", "negative", "Mild criticism of unclear excess terms; review neutral alternative."),
    ("ad436966", "positive", "Praises communication and recommends insurer."),
    ("d2ad9250", "negative", "Nightmare claims process and unhelpful staff."),
    ("bfcb9803", "negative", "Brief purchase praise outweighed by detailed inflexibility complaint."),
    ("e497399d", "positive", "Satisfied with coverage and setup; no claim experience."),
    ("92791905", "negative", "Uncorrected name and unfulfilled promise; product untested."),
    ("60137a74", "negative", "Delayed rejected claim and severe business impact."),
    ("f12259e8", "negative", "Two rejected claims and warning against insurer."),
    ("43d70535", "mixed_demo", "Positive title and possible reuse alongside meaningful email delay criticism."),
    ("fa36cbc0", "negative", "Concern and criticism of claim treatment despite own safe trip; ambiguous target."),
    ("030b9266", "negative", "Praise targets other businesses; insurer receives sustained criticism."),
    ("0183e0a7", "positive", "Fine experience and straightforward purchase; claims untested."),
    ("dd4f6bf1", "negative", "Poor claim management and denial."),
    ("04040462", "negative", "Discrimination and excessive premium dominate generic value praise."),
    ("daba33bd", "positive", "Good value and clear simple setup."),
    ("ae24320b", "positive", "Impressed with coverage and reminder."),
    ("e9c96f53", "negative", "Useless insurance and denied claims."),
    ("90c7142c", "negative", "Unanswered contact attempts and delayed claim; ignore rating."),
    ("c7cdae86", "positive", "Minor card delay resolved; overall all fine."),
    ("4c655bbc", "mixed_demo", "Substantial price/reward praise and meaningful coverage exclusion concern."),
    ("d1fc47e5", "negative", "Incorrect coverage change and negative claim outcome."),
    ("dc1aee1a", "negative", "Coverage date mistake and refusal of discretion."),
    ("884698d5", "negative", "Early assistance praise overwhelmed by six-month claim complaint."),
    ("b80a99ed", "neutral", "Main point is inability to assess unused product; review mild criticism alternative."),
    ("098f54ff", "positive", "Good value and easy signup; claims untested."),
    ("9cc3db8d", "positive", "Efficient claim support and reimbursement."),
    ("8abc9552", "positive", "Great prices and recommendation."),
    ("727b30ad", "negative", "Rejected claim and inaccessible support."),
    ("2bc35861", "positive", "Happy with coverage and price; would reuse."),
    ("d37981b4", "negative", "Criticism of coronavirus exclusions."),
    ("a00cbac1", "negative", "Prolonged documentation dispute and misleading limits."),
    ("fb074abf", "negative", "Excessive claims paperwork."),
    ("02c71c4d", "negative", "Appalling treatment and escalating evidence demands."),
    ("433f4bfb", "negative", "Sustained exclusion and premium-increase criticism."),
    ("bd839983", "negative", "Disappointed by claim refusal; great trip is not insurer praise."),
    ("8bc22a18", "positive", "Successful no-fuss claim despite family medical problems."),
    ("711c52a0", "mixed_demo", "Meaningful value/coverage praise and unfair-premium complaint."),
    ("21915d9e", "negative", "Initial staff praise outweighed by arrogant service and inconvenient changes."),
    ("d034d2e3", "positive", "Happy with premiums coverage and setup."),
    ("741044cc", "negative", "Claim denied on technicality and cautionary title."),
    ("54913408", "negative", "Missing confirmation and information; review neutral request alternative."),
    ("94e4279e", "positive", "Repeated use and first choice; title unclear, human review needed."),
    ("db143142", "positive", "Helpful staff and cheaper premium; claim service not yet known."),
    ("bd915ef4", "negative", "Delayed claim and explicit completed policy cancellations."),
    ("1d87c39e", "positive", "Good value easy setup and proactive communication."),
    ("48b54c24", "positive", "Trouble-free claim and excellent value."),
    ("9ef17c2b", "positive", "Good value easy setup; caveat about untested support."),
    ("caa0df82", "negative", "Rejected claim and buyer warning."),
    ("beafbe1c", "positive", "Affordable cover provided peace of mind."),
    ("e85e16ec", "mixed_demo", "Strong purchase/value praise alongside meaningful delayed-claim criticism."),
]


def main():
    output = ROOT / "data/daa_annotations.csv"
    if output.exists():
        raise FileExistsError("Preserving existing annotations; bootstrap is one-time only")
    with (ROOT / "data/daa_reviews_sample.csv").open(newline="", encoding="utf-8") as handle:
        reviews = list(csv.DictReader(handle))[:50]
    rows = []
    for review, (prefix, label, note) in zip(reviews, DRAFTS, strict=True):
        if not review["review_id"].startswith(prefix):
            raise ValueError("Sample order changed; frozen selection must be reviewed")
        rows.append({"review_id": review["review_id"], "draft_sentiment": label,
                     "human_sentiment": "", "human_approved": "False", "reviewer": "",
                     "annotation_notes": note,
                     "text_sha256": hashlib.sha256(clean_text(review["text"]).encode()).hexdigest()})
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} unapproved AI drafts to {output}")


if __name__ == "__main__":
    main()
