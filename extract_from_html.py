import glob
import re
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path


HTML_FOLDER = "html_exports/*.html"
OUTPUT_FILE = "data/appointments.csv"


# ----------------------------
# Normalize phone (last 9 digits)
# ----------------------------
def normalize_phone(phone_raw):
    if not phone_raw:
        return ""
    digits = re.sub(r"\D", "", phone_raw)
    return digits[-9:] if len(digits) >= 9 else digits


def main():
    appointments = []

    files = glob.glob(HTML_FOLDER)

    print(f"Found {len(files)} files")
    if not files:
        raise SystemExit("No HTML exports found in html_exports/. Add exports before extracting.")

    for file_path in files:

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        messages = soup.find_all("div", class_="text")

        for msg in messages:
            text = msg.get_text(separator="\n").strip()

            if "You have a new appointment request!" in text:

                customer = None
                date = None
                time = None
                phone = None

                lines = [l.strip() for l in text.split("\n") if l.strip()]

                for i, line in enumerate(lines):

                    if line.startswith("Customer:"):
                        customer = line.replace("Customer:", "").strip()

                    elif line.startswith("Date:"):
                        date = line.replace("Date:", "").strip()

                    elif line.startswith("Time:"):
                        time = line.replace("Time:", "").strip()

                    elif "Phone" in line:
                        # handle inline phone
                        if ":" in line:
                            phone = line.split(":")[-1].strip()
                        elif i + 1 < len(lines):
                            phone = lines[i + 1].strip()

                if customer and date and time:
                    appointments.append({
                        "Customer": customer,
                        "Date": date,
                        "Time": time,
                        "Phone": normalize_phone(phone)
                    })


    # ----------------------------
    # Convert to DataFrame
    # ----------------------------

    df = pd.DataFrame(appointments, columns=["Customer", "Date", "Time", "Phone"])

    print("Initial records:", len(df))
    if df.empty:
        raise SystemExit("No matching appointment messages found; check the export format.")


    # ----------------------------
    # Attach phone to same-name records
    # ----------------------------

    # Create mapping of name -> phone (only where phone exists)
    known = df[df["Phone"] != ""]
    unique_names = known.groupby("Customer")["Phone"].nunique()
    phone_map = (known[known["Customer"].isin(unique_names[unique_names == 1].index)]
                 .drop_duplicates("Customer").set_index("Customer")["Phone"].to_dict())

    # Fill missing phones using name
    df["Phone"] = df.apply(
        lambda row: row["Phone"] or phone_map.get(row["Customer"], ""),
        axis=1
    )

    # ----------------------------
    # Remove rows still missing phone
    # ----------------------------
    print(len(df))
    # Clean phone column
    df["Phone"] = df["Phone"].fillna("").str.strip()

    # Remove rows without valid phone
    df = df[df["Phone"] != ""]

    print("After removing missing phones:", len(df))


    # ----------------------------
    # Deduplicate properly
    # ----------------------------

    df = df.drop_duplicates(subset=["Customer", "Phone", "Date", "Time"])

    # Create unique key
    df["CustomerPhone"] = df["Customer"] + " | " + df["Phone"]


    # ----------------------------
    # Save
    # ----------------------------

    Path("data").mkdir(exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} clean appointments to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
