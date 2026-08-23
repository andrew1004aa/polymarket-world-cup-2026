#!/usr/bin/env python3
"""Build mutually exclusive prematch timing bands for H4."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "robustness_inputs" / "v1"

SOURCE_FIELDS = [
    "source_analysis_record_id", "market_id", "event_id", "calendar_hour_utc",
    "minute_start_timestamp", "minute_start_utc", "minutes_to_actual_kickoff",
    "delta_yes_price", "signed_log_whale_net_flow",
    "signed_log_nonwhale_net_flow", "yes_price_t",
    "lagged_30m_price_change", "log_lagged_30m_volume",
    "log_minutes_to_kickoff",
]
BANDS = ["gt_24h", "12_to_24h", "6_to_12h", "1_to_6h", "last_60m"]
DUMMIES = ["band_12_to_24h", "band_6_to_12h", "band_1_to_6h", "band_last_60m"]
INTERACTIONS = [
    *(f"whale_x_{name.removeprefix('band_')}" for name in DUMMIES),
    *(f"nonwhale_x_{name.removeprefix('band_')}" for name in DUMMIES),
]
FIELDS = SOURCE_FIELDS + ["time_band"] + DUMMIES + INTERACTIONS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def band(minutes: float) -> str:
    if minutes >= 1440: return "gt_24h"
    if minutes >= 720: return "12_to_24h"
    if minutes >= 360: return "6_to_12h"
    if minutes >= 60: return "1_to_6h"
    if minutes >= 0: return "last_60m"
    raise ValueError(f"Negative prematch minutes: {minutes}")


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--force",action="store_true")
    parser.add_argument("--version",default="v1");parser.add_argument("--output-dir",default=None);args=parser.parse_args()
    outdir=ROOT/args.output_dir if args.output_dir else DEFAULT_OUT
    config_path=outdir/"h4_timing_config.json";config=json.loads(config_path.read_text(encoding="utf-8"));source=ROOT/config["source"];output=ROOT/config["output"]
    source_fields=config.get("source_fields",SOURCE_FIELDS)
    split_fields=config.get("split_flow_fields",["signed_log_whale_net_flow","signed_log_nonwhale_net_flow"])
    prefixes=config.get("interaction_prefixes",["whale","nonwhale"])
    interactions=[*(f"{prefixes[0]}_x_{name.removeprefix('band_')}" for name in DUMMIES),
                  *(f"{prefixes[1]}_x_{name.removeprefix('band_')}" for name in DUMMIES)]
    fields=source_fields+["time_band"]+DUMMIES+interactions
    qc_path=outdir/"h4_timing_input_qc.json"
    if output.exists() and qc_path.exists() and not args.force:
        print("H4 timing input already built; use --force to rebuild.");return 0
    if sha256(source)!=config["source_sha256"]: raise SystemExit("H4 source checksum mismatch")
    temporary=output.with_suffix(output.suffix+".tmp");counts=Counter();zero=Counter()
    groups={name:{"markets":set(),"events":set(),"hours":set()} for name in BANDS}
    with gzip.open(source,"rt",encoding="utf-8",newline="") as src, gzip.open(temporary,"wt",encoding="utf-8",newline="",compresslevel=6) as dst:
        reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        missing=sorted(set(source_fields)-set(reader.fieldnames or []))
        if missing: raise RuntimeError(f"Missing H4 fields: {missing}")
        for row in reader:
            counts["all"]+=1;label=band(float(row["minutes_to_actual_kickoff"]));counts[label]+=1
            if float(row["delta_yes_price"])==0: zero[label]+=1
            whale=float(row[split_fields[0]]);nonwhale=float(row[split_fields[1]])
            out={field:row.get(field,"") for field in source_fields};out["time_band"]=label
            for dummy in DUMMIES:
                name=dummy.removeprefix("band_");indicator=int(label==name);out[dummy]=indicator
                out[f"{prefixes[0]}_x_{name}"]=whale*indicator;out[f"{prefixes[1]}_x_{name}"]=nonwhale*indicator
            writer.writerow(out);groups[label]["markets"].add(row["market_id"]);groups[label]["events"].add(row["event_id"]);groups[label]["hours"].add(row["calendar_hour_utc"])
            if counts["all"]%250000==0: print(f"Constructed {counts['all']:,} H4 rows",flush=True)
    temporary.replace(output);errors=[]
    if counts["all"]!=int(config["expected_rows"]):errors.append("H4 row count mismatch")
    if sum(counts[name] for name in BANDS)!=counts["all"]:errors.append("Timing bands do not partition source")
    if any(counts[name]==0 for name in BANDS):errors.append("Empty timing band")
    qc={"generated_at":now(),"version":args.version,"source_rows":counts["all"],"bands":{},
        "output":config["output"],"output_sha256":sha256(output),"output_bytes":output.stat().st_size,"errors":errors}
    for name in BANDS:
        qc["bands"][name]={"rows":counts[name],"zero_price_changes":zero[name],"zero_share":zero[name]/counts[name],
                            "markets":len(groups[name]["markets"]),"events":len(groups[name]["events"]),"calendar_hours":len(groups[name]["hours"])}
    atomic_json(qc_path,qc);config["status"]="built_qc_passed" if not errors else "built_qc_failed";config["built_at"]=qc["generated_at"]
    atomic_json(config_path,config);print(json.dumps(qc,indent=2),flush=True);return 0 if not errors else 1


if __name__=="__main__":raise SystemExit(main())
