"""Generate fictional appointment fixtures, never real contact details."""
import csv
from datetime import date, timedelta
from pathlib import Path
import random

SEED = 42
OBSERVATION_END = date(2026, 5, 28)


def generate(output="examples/appointments_sample.csv"):
    rng = random.Random(SEED)
    phones = []
    while len(phones) < 50:
        candidate = "0" + "".join(str(rng.randrange(10)) for _ in range(8))
        if all(sum(a != b for a, b in zip(candidate, previous)) >= 5 for previous in phones):
            phones.append(candidate)
    rows = []
    for index, phone in enumerate(phones):
        customer = f"Synthetic Customer {index + 1:03d}"
        day = date(2020, 1, 1) + timedelta(days=rng.randrange(365))
        cycle = (18, 28, 42, 60, 85)[index % 5]
        visit = 0
        while day <= OBSERVATION_END:
            time = ("10:00 AM", "12:30 PM", "03:20 PM")[visit % 3]
            # Occasional name aliases with the same phone.
            name = f"Synthetic {index + 1:03d}" if visit % 13 == 5 else customer
            number = phone
            if visit % 19 == 8:
                number = phone[:-1] + str((int(phone[-1]) + 1) % 10)
            rows.append({"Customer": name, "Date": day.isoformat(), "Time": time,
                         "Phone": number, "CustomerPhone": f"{name} | {number}"})
            if visit % 17 == 3 and day + timedelta(days=1) <= OBSERVATION_END:
                next_day = day + timedelta(days=1)
                rows.append({"Customer": name, "Date": next_day.isoformat(), "Time": "04:00 PM",
                             "Phone": number, "CustomerPhone": f"{name} | {number}"})
            visit += 1
            if index >= 45:  # Include one-time customers, not false negatives.
                break
            day += timedelta(days=max(5, int(rng.gauss(cycle, cycle * .28))))
    rows.sort(key=lambda row: (row["Date"], row["CustomerPhone"], row["Time"]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Customer", "Date", "Time", "Phone", "CustomerPhone"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} fictional rows, 50 customers; observation end {OBSERVATION_END}.")


if __name__ == "__main__":
    generate()
