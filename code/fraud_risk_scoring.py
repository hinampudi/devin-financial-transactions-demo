"""
Fraud Risk Scoring System

Computes a risk score (0-100) for each transaction and assigns a risk category
(LOW, MEDIUM, HIGH) based on the following guidelines:

Risk Factors:
1. Transactions above 10,000 are high risk.
2. CASH_OUT and TRANSFER are higher risk transaction types.
3. Transactions to new or previously unseen destination accounts are risky.
4. Rapid sequence of transactions from the same account increases risk.
5. Fraudulent transactions often involve high amounts followed by cash-out.

Risk Levels:
- LOW: score < 40
- MEDIUM: score between 40 and 70
- HIGH: score > 70
"""

import csv
import os
from collections import defaultdict


INPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "Example1.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "transaction_risk_report.csv")

HIGH_RISK_TYPES = {"CASH_OUT", "TRANSFER"}
HIGH_AMOUNT_THRESHOLD = 10000


def load_transactions(filepath):
    """Load transactions from CSV file."""
    transactions = []
    with open(filepath, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["step"] = int(row["step"])
            row["amount"] = float(row["amount"])
            row["oldbalanceOrg"] = float(row["oldbalanceOrg"])
            row["newbalanceOrig"] = float(row["newbalanceOrig"])
            row["oldbalanceDest"] = float(row["oldbalanceDest"])
            row["newbalanceDest"] = float(row["newbalanceDest"])
            row["isFraud"] = int(row["isFraud"])
            row["isFlaggedFraud"] = int(row["isFlaggedFraud"])
            transactions.append(row)
    return transactions


def build_account_profiles(transactions):
    """Build profiles tracking destination account history and origin account
    transaction frequency per step."""
    dest_first_seen = {}
    origin_step_counts = defaultdict(lambda: defaultdict(int))

    for i, txn in enumerate(transactions):
        dest = txn["nameDest"]
        orig = txn["nameOrig"]
        step = txn["step"]

        if dest not in dest_first_seen:
            dest_first_seen[dest] = i

        origin_step_counts[orig][step] += 1

    return dest_first_seen, origin_step_counts


def compute_risk_score(txn, txn_index, dest_first_seen, origin_step_counts):
    """Compute a risk score (0-100) for a single transaction.

    Scoring breakdown (points are additive, capped at 100):
    - High amount (>10,000):              up to 25 points
    - High-risk transaction type:         20 points
    - New/unseen destination account:     15 points
    - Rapid transactions from same origin: up to 20 points
    - Balance anomaly indicators:         up to 20 points
    """
    score = 0

    # Factor 1: High transaction amount (up to 25 points)
    amount = txn["amount"]
    if amount > HIGH_AMOUNT_THRESHOLD:
        # Scale from 10 to 25 based on how far above threshold
        ratio = min(amount / HIGH_AMOUNT_THRESHOLD, 10)
        score += 10 + (ratio - 1) * (15 / 9)

    # Factor 2: High-risk transaction type (20 points)
    if txn["type"] in HIGH_RISK_TYPES:
        score += 20

    # Factor 3: New or previously unseen destination account (15 points)
    dest = txn["nameDest"]
    if dest_first_seen.get(dest) == txn_index:
        score += 15

    # Factor 4: Rapid sequence of transactions from same account (up to 20 points)
    orig = txn["nameOrig"]
    step = txn["step"]
    txn_count_in_step = origin_step_counts[orig][step]
    if txn_count_in_step > 1:
        # More transactions in the same step = higher risk
        rapid_score = min((txn_count_in_step - 1) * 10, 20)
        score += rapid_score

    # Factor 5: Balance anomaly - account drained or suspicious patterns (up to 20 points)
    old_balance = txn["oldbalanceOrg"]
    new_balance = txn["newbalanceOrig"]

    # Account fully drained
    if old_balance > 0 and new_balance == 0:
        score += 15

    # Sending more than account balance (overdraft)
    if amount > old_balance and old_balance > 0:
        score += 10

    # Destination balance unchanged despite receiving funds (possible mule account)
    old_dest_balance = txn["oldbalanceDest"]
    new_dest_balance = txn["newbalanceDest"]
    if amount > 0 and old_dest_balance > 0 and new_dest_balance == 0:
        score += 5

    # Cap score at 100
    return min(score, 100)


def assign_risk_category(score):
    """Assign risk category based on score thresholds."""
    if score < 40:
        return "LOW"
    elif score <= 70:
        return "MEDIUM"
    else:
        return "HIGH"


def generate_risk_report(transactions, output_filepath):
    """Compute risk scores for all transactions and generate a CSV report."""
    dest_first_seen, origin_step_counts = build_account_profiles(transactions)

    report_rows = []
    summary = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    for i, txn in enumerate(transactions):
        risk_score = compute_risk_score(
            txn, i, dest_first_seen, origin_step_counts
        )
        risk_category = assign_risk_category(risk_score)
        summary[risk_category] += 1

        report_rows.append({
            "step": txn["step"],
            "type": txn["type"],
            "amount": txn["amount"],
            "nameOrig": txn["nameOrig"],
            "nameDest": txn["nameDest"],
            "oldbalanceOrg": txn["oldbalanceOrg"],
            "newbalanceOrig": txn["newbalanceOrig"],
            "oldbalanceDest": txn["oldbalanceDest"],
            "newbalanceDest": txn["newbalanceDest"],
            "isFraud": txn["isFraud"],
            "isFlaggedFraud": txn["isFlaggedFraud"],
            "risk_score": round(risk_score, 2),
            "risk_category": risk_category,
        })

    # Write CSV report
    fieldnames = [
        "step", "type", "amount", "nameOrig", "nameDest",
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud", "risk_score", "risk_category",
    ]
    with open(output_filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    return report_rows, summary


def print_summary(report_rows, summary):
    """Print a human-readable summary of the risk report."""
    total = len(report_rows)
    print("=" * 70)
    print("TRANSACTION FRAUD RISK REPORT SUMMARY")
    print("=" * 70)
    print(f"Total transactions analyzed: {total}")
    print(f"  LOW  risk (score < 40):      {summary['LOW']:>4} ({summary['LOW']/total*100:.1f}%)")
    print(f"  MEDIUM risk (40 <= score <= 70): {summary['MEDIUM']:>4} ({summary['MEDIUM']/total*100:.1f}%)")
    print(f"  HIGH risk (score > 70):      {summary['HIGH']:>4} ({summary['HIGH']/total*100:.1f}%)")
    print()

    high_risk = [r for r in report_rows if r["risk_category"] == "HIGH"]
    if high_risk:
        print(f"HIGH RISK TRANSACTIONS ({len(high_risk)}):")
        print("-" * 70)
        print(f"{'Type':<12} {'Amount':>14} {'Origin':<16} {'Dest':<16} {'Score':>6} {'Fraud':>6}")
        print("-" * 70)
        for r in high_risk:
            print(
                f"{r['type']:<12} {r['amount']:>14,.2f} "
                f"{r['nameOrig']:<16} {r['nameDest']:<16} "
                f"{r['risk_score']:>6.1f} {'YES' if r['isFraud'] else 'NO':>6}"
            )
    print("=" * 70)


def main():
    print(f"Loading transactions from: {INPUT_FILE}")
    transactions = load_transactions(INPUT_FILE)
    print(f"Loaded {len(transactions)} transactions.")

    print(f"Computing risk scores...")
    report_rows, summary = generate_risk_report(transactions, OUTPUT_FILE)

    print(f"Risk report saved to: {OUTPUT_FILE}")
    print()
    print_summary(report_rows, summary)


if __name__ == "__main__":
    main()
