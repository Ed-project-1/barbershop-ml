"""Resolve customer aliases using explicit name/phone rules; preserve an audit trail."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata
import pandas as pd


def normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"\w+", text))


def normalize_phone(value):
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 11 and digits.startswith("48"):
        digits = digits[2:]
    return digits if len(digits) == 9 else ""


def similar_names(left, right):
    if left == right:
        return True
    a, b = left.split(), right.split()
    if not a or not b:
        return False
    if len(a) == 1 or len(b) == 1:
        first_a, first_b = a[0], b[0]
        prefix = min(len(first_a), len(first_b)) >= 4 and (first_a.startswith(first_b) or first_b.startswith(first_a))
        return prefix or SequenceMatcher(None, first_a, first_b).ratio() >= 0.88
    return SequenceMatcher(None, left, right).ratio() >= 0.88


def clean_customers(data):
    if "Customer" not in data or not ({"Phone", "CustomerPhone"} & set(data.columns)):
        raise ValueError("Need Customer and either Phone or CustomerPhone columns.")
    df = data.copy()
    names = df["Customer"].fillna("").astype(str).str.strip()
    raw_phone = df["Phone"].astype("string") if "Phone" in df else pd.Series("", index=df.index, dtype="string")
    if "CustomerPhone" in df:
        fallback = df["CustomerPhone"].astype("string").str.rsplit("|", n=1).str[-1]
        raw_phone = raw_phone.where(raw_phone.notna() & raw_phone.str.strip().ne(""), fallback)
    phones = raw_phone.map(normalize_phone)
    valid = names.ne("") & phones.ne("")
    rejected = df[~valid].copy()
    rejected["Rejection_Reason"] = "Missing name or phone not a valid nine-digit Polish number"
    df = df[valid].copy()
    df["Original_Customer"] = df["Customer"]
    df["Original_Phone"] = raw_phone[valid]
    df["_name"] = names[valid].map(normalize_name)
    df["_phone"] = phones[valid]
    nodes = sorted(set(zip(df["_name"], df["_phone"])))
    parent = list(range(len(nodes)))
    edges = []

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def merge(a, b, reason):
        edges.append({"Name_A": nodes[a][0], "Phone_A": nodes[a][1], "Name_B": nodes[b][0], "Phone_B": nodes[b][1], "Rule": reason})
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    phone_groups, full_name_groups = defaultdict(list), defaultdict(list)
    for index, (name, phone) in enumerate(nodes):
        phone_groups[phone].append(index)
        if len(name.split()) >= 2:
            full_name_groups[" ".join(sorted(name.split()))].append(index)
    for group in phone_groups.values():
        for index in group[1:]:
            merge(group[0], index, "1: same phone")
    for group in full_name_groups.values():
        for index in group[1:]:
            merge(group[0], index, "3: same full name")
    for a, (name_a, phone_a) in enumerate(nodes):
        for b in range(a + 1, len(nodes)):
            name_b, phone_b = nodes[b]
            differences = sum(x != y for x, y in zip(phone_a, phone_b))
            if 1 <= differences <= 2 and similar_names(name_a, name_b):
                merge(a, b, "2: similar name and 1-2 changed phone digits")
    lookup = {node: index for index, node in enumerate(nodes)}
    if "CustomerID" not in df:
        df["CustomerID"] = pd.Series(index=df.index, dtype="string")
    df["_group"] = [root(lookup[node]) for node in zip(df["_name"], df["_phone"])]
    audit = []
    for _, group in df.groupby("_group"):
        name_counts = Counter(str(name).strip() for name in group["Original_Customer"])
        canonical = sorted(name_counts, key=lambda n: (-len(normalize_name(n).split()), -len(normalize_name(n)), -name_counts[n], n.casefold()))[0]
        phone_counts = Counter(group["_phone"])
        primary = sorted(phone_counts, key=lambda p: (-phone_counts[p], p))[0]
        existing_ids = sorted(set(group.get("CustomerID", pd.Series(dtype=str)).dropna().astype(str)))
        if len(existing_ids) > 1:
            raise ValueError("A merge joins existing CustomerIDs; review identity mapping before proceeding.")
        customer_id = existing_ids[0] if existing_ids else "customer_" + min(phone_counts)
        df.loc[group.index, "CustomerID"] = customer_id
        df.loc[group.index, "Customer"] = canonical
        df.loc[group.index, "Phone"] = primary
        df.loc[group.index, "CustomerPhone"] = canonical + " | " + primary
        for (original_name, original_phone), count in group.groupby(["Original_Customer", "_phone"]).size().items():
            audit.append({"CustomerID": customer_id, "Canonical_Name": canonical, "Canonical_Phone": primary,
                          "Original_Name": original_name, "Original_Phone": original_phone, "Rows": int(count)})
    # Collapse consecutive appointments less than three calendar days apart.
    # Within each customer cluster, retain the latest appointment date/time.
    before = len(df)
    if {"Date", "Time"}.issubset(df.columns):
        dates = pd.to_datetime(df["Date"], format="mixed", errors="coerce").dt.normalize()
        times = pd.to_datetime(df["Time"].astype(str).str.upper(), format="%I:%M %p", errors="coerce")
        df["_date_key"] = dates.astype(str).where(dates.notna(), df["Date"].astype(str))
        df["_time_key"] = times.astype(str).where(times.notna(), df["Time"].astype(str))
        df = df.drop_duplicates(["CustomerID", "_date_key", "_time_key"], keep="last")
        df["_appointment_date"] = dates.loc[df.index]
        df["_appointment_minutes"] = (times.dt.hour * 60 + times.dt.minute).loc[df.index].fillna(-1)
        valid_dates = df[df["_appointment_date"].notna()].sort_values(
            ["CustomerID", "_appointment_date", "_appointment_minutes"], kind="stable"
        ).copy()
        gaps = valid_dates.groupby("CustomerID")["_appointment_date"].diff().dt.days
        valid_dates["_cluster"] = (gaps.isna() | gaps.ge(3)).cumsum()
        keep = valid_dates.groupby(["CustomerID", "_cluster"], sort=False).tail(1).index
        df = df.loc[df.index.isin(keep) | df["_appointment_date"].isna()].copy()
    print(f"Valid rows: {before}; after appointment deduplication: {len(df)}; customers: {df['CustomerID'].nunique()}; rejected: {len(rejected)}")
    return (df.drop(columns=[c for c in df if c.startswith("_")]),
            pd.DataFrame(audit), pd.DataFrame(edges, columns=["Name_A", "Phone_A", "Name_B", "Phone_B", "Rule"]), rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/appointments.csv")
    parser.add_argument("--output", default="data/appointments_cleaned.csv")
    parser.add_argument("--audit-dir", default="reports/customer_cleaning")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Output must differ from input to preserve the original data.")
    data = pd.read_csv(args.input, dtype="string")
    clean, mapping, merges, rejected = clean_customers(data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    folder = Path(args.audit_dir)
    folder.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    mapping.to_csv(folder / "identity_map.csv", index=False)
    merges.to_csv(folder / "merge_rules.csv", index=False)
    rejected.to_csv(folder / "rejected_rows.csv", index=False)
    print(f"Cleaned appointments: {args.output}")


if __name__ == "__main__":
    main()
