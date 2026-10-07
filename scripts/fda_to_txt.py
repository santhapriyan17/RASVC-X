import json, pathlib

IN_DIR = pathlib.Path("data/medical_raw/c1_real")
OUT_DIR = pathlib.Path("data/medical_raw/c1_real")

FIELDS = [
    "purpose", "indications_and_usage", "dosage_and_administration",
    "warnings", "contraindications", "adverse_reactions",
    "drug_interactions", "how_supplied"
]

total = 0
for batch_file in sorted(IN_DIR.glob("fda_labels_batch*.json")):
    data = json.loads(batch_file.read_text(encoding="utf-8"))
    results = data.get("results", [])
    for i, item in enumerate(results):
        brand_list = item.get("openfda", {}).get("brand_name", ["unknown"])
        brand = brand_list[0] if brand_list else "unknown"
        brand = "".join(c if c.isalnum() or c in "-_" else "_" for c in brand)[:40]
        parts = []
        for field in FIELDS:
            val = item.get(field)
            if val:
                heading = field.upper().replace("_", " ")
                parts.append(f"=== {heading} ===\n" + "\n".join(val))
        if parts:
            fname = OUT_DIR / f"fda_{brand}_{i:03d}.txt"
            fname.write_text("\n\n".join(parts), encoding="utf-8", errors="replace")
            total += 1
    print(f"Processed {batch_file.name}: {len(results)} labels")

print(f"\nTotal text files written: {total}")