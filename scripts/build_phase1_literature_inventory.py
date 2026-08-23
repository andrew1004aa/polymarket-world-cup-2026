import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp" / "literature_stage1" / "workbook"
OUT = ROOT / "docs" / "literature" / "phase1_existing_literature_inventory.csv"


THEME_MAP = {
    "預測市場與資訊聚合": "A",
    "市場微結構與價格衝擊": "B",
    "知情交易、效率與交易者技能": "C",
    "市場操縱與市場誠信": "G",
    "加密貨幣與區塊鏈市場品質": "G/H",
    "Polymarket / 去中心化預測市場": "H",
    "Crypto / Blockchain 市場品質": "G/H",
    "實證方法與資料品質": "F/G",
}

FLAGS = {
    "Can Asset Markets Be Manipulated? A Field Experiment with Racetrack Betting":
        "metadata_error: journal is Journal of Political Economy 106(3), not JEP",
    "Direct Estimation of Equity Market Impact":
        "citation_mismatch: Almgren et al. (2005), not Almgren and Chriss; rejects square-root temporary-impact model in favour of 3/5 power",
    "Retail Trading in Options and the Rise of the Big Three Wholesalers":
        "outdated_status: final Journal of Finance 78(6), 3465-3514, DOI 10.1111/jofi.13285",
    "Stock Market Manipulations":
        "verification_needed: bibliographic record and exact empirical claim",
    "Polymarket Volume Is Being Double-Counted":
        "non_academic_background_only",
    "Distilling the Wisdom of Crowds: Prediction Markets vs. Prediction Polls":
        "cross_sheet_duplicate",
}


def clean(value):
    return (value or "").replace("\n", " ").strip()


def flag_for(title):
    exact = FLAGS.get(title)
    if exact:
        return exact
    lower = title.lower()
    if "can asset markets be manipulated" in lower:
        return "metadata_error: journal is Journal of Political Economy 106(3), not JEP"
    if "distilling the wisdom of crowds" in lower:
        return "cross_sheet_duplicate"
    return None


rows = []
with (SRC / "main_literature.csv").open(encoding="utf-8-sig", newline="") as fh:
    for r in csv.DictReader(fh):
        title = clean(r["篇名"])
        rows.append({
            "source_sheet": "Main Literature",
            "author_year": clean(r["作者(年)"]),
            "title": title,
            "year": "",
            "venue": clean(r["期刊"]),
            "doi_or_url": "",
            "current_category": clean(r["分類"]),
            "provisional_theme": THEME_MAP.get(clean(r["分類"]), "unmapped"),
            "verification_level": "workbook_only_not_two_source_verified",
            "phase1_flag": flag_for(title) or "requires_record_and_claim_verification",
        })

with (SRC / "additional_literature.csv").open(encoding="utf-8-sig", newline="") as fh:
    for r in csv.DictReader(fh):
        title = clean(r["Title"])
        theme = {
            "1. Prediction Markets and Information Aggregation": "A",
            "2. Market Microstructure": "B",
            "3. Whale Trading": "B/C",
            "3. Decentralised Exchanges": "G/H",
            "4. Market Manipulation and Market Integrity": "G",
            "5. Polymarket and Blockchain Prediction Markets": "H",
        }.get(clean(r["Literature Section"]), "unmapped")
        rows.append({
            "source_sheet": "Additional Literature",
            "author_year": clean(r["Authors"]),
            "title": title,
            "year": clean(r["Year"]),
            "venue": clean(r["Journal"]),
            "doi_or_url": clean(r["DOI / Stable URL"]),
            "current_category": clean(r["Literature Section"]),
            "provisional_theme": theme,
            "verification_level": "citation_present_not_two_source_verified",
            "phase1_flag": flag_for(title) or "requires_claim_and_method_verification",
        })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8-sig", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {len(rows)} rows to {OUT}")
