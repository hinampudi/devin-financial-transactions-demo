"""
Anomalous Transaction Sequence Detection

Detects anomalous transaction sequences for each customer by:
1. Loading the dataset
2. Grouping transactions by customer (nameOrig)
3. Sorting transactions by time (step)
4. Analyzing sequences of transaction types and amounts
5. Identifying suspicious patterns:
   - Repeated high-value transactions
   - Transfer followed by cash-out (including cross-account chains)
   - Sudden increase in transaction amounts
6. Assigning anomaly levels (LOW, MEDIUM, HIGH)
7. Generating a sequence anomaly report

Output includes: customer_id, sequence_pattern, anomaly_level, and explanation.
"""

import os

import pandas as pd

INPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "Example1.csv")
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sequence_anomaly_report.csv"
)

HIGH_AMOUNT_THRESHOLD = 10000
HIGH_RISK_TYPES = {"CASH_OUT", "TRANSFER"}

# Anomaly level thresholds (based on cumulative anomaly score)
LOW_THRESHOLD = 40
HIGH_THRESHOLD = 70


def load_dataset(filepath):
    """Load and parse the transaction dataset."""
    df = pd.read_csv(filepath)
    numeric_cols = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_origin_index(df):
    """Build a mapping from origin account to their outgoing transactions."""
    orig_index = {}
    for _, row in df.iterrows():
        orig = row["nameOrig"]
        if orig not in orig_index:
            orig_index[orig] = []
        orig_index[orig].append(row)
    return orig_index


def detect_repeated_high_value(transactions):
    """Detect repeated high-value transactions from a customer's sequence.

    Returns a score contribution and explanation if the pattern is found.
    """
    high_value_txns = transactions[transactions["amount"] > HIGH_AMOUNT_THRESHOLD]
    count = len(high_value_txns)
    if count >= 2:
        total_amount = high_value_txns["amount"].sum()
        return (
            min(count * 15, 40),
            f"Repeated high-value transactions: {count} transactions "
            f"exceeding {HIGH_AMOUNT_THRESHOLD:,} (total: {total_amount:,.2f})",
        )
    elif count == 1:
        amount = high_value_txns.iloc[0]["amount"]
        if amount > HIGH_AMOUNT_THRESHOLD * 5:
            return (
                20,
                f"Single very high-value transaction: {amount:,.2f} "
                f"(>{HIGH_AMOUNT_THRESHOLD * 5:,})",
            )
    return 0, ""


def detect_transfer_then_cashout(transactions, orig_index):
    """Detect transfer followed by cash-out pattern.

    Checks both within a customer's own sequence and across linked accounts:
    - Direct: customer does TRANSFER then CASH_OUT in their own sequence
    - Chain: customer does TRANSFER to account X, and X does CASH_OUT

    Returns a score contribution and explanation if the pattern is found.
    """
    types_sequence = transactions["type"].tolist()
    steps_sequence = transactions["step"].tolist()
    amounts_sequence = transactions["amount"].tolist()

    patterns_found = []

    # Check within customer's own sequence
    for i in range(len(types_sequence) - 1):
        if types_sequence[i] == "TRANSFER" and types_sequence[i + 1] == "CASH_OUT":
            step_gap = steps_sequence[i + 1] - steps_sequence[i]
            if step_gap <= 1:
                patterns_found.append(
                    f"Direct TRANSFER({amounts_sequence[i]:,.2f}) -> "
                    f"CASH_OUT({amounts_sequence[i + 1]:,.2f}) "
                    f"at steps {steps_sequence[i]}-{steps_sequence[i + 1]}"
                )

    # Check cross-account chains: customer does TRANSFER to X, X does CASH_OUT
    for _, txn in transactions.iterrows():
        if txn["type"] == "TRANSFER":
            dest_account = txn["nameDest"]
            if dest_account in orig_index:
                for linked_txn in orig_index[dest_account]:
                    if linked_txn["type"] == "CASH_OUT":
                        step_gap = linked_txn["step"] - txn["step"]
                        if 0 <= step_gap <= 1:
                            patterns_found.append(
                                f"Chain: TRANSFER({txn['amount']:,.2f}) to "
                                f"{dest_account} -> {dest_account} "
                                f"CASH_OUT({linked_txn['amount']:,.2f})"
                            )

    if patterns_found:
        score = min(len(patterns_found) * 30, 50)
        explanation = (
            "Transfer-then-cash-out pattern: " + "; ".join(patterns_found)
        )
        return score, explanation
    return 0, ""


def detect_sudden_amount_increase(transactions):
    """Detect sudden increase in transaction amounts within a customer's sequence.

    Compares each transaction amount with the rolling average of prior transactions.

    Returns a score contribution and explanation if the pattern is found.
    """
    if len(transactions) < 2:
        return 0, ""

    amounts = transactions["amount"].tolist()
    steps = transactions["step"].tolist()
    spikes = []

    for i in range(1, len(amounts)):
        prior_avg = sum(amounts[:i]) / i
        if prior_avg > 0 and amounts[i] > prior_avg * 3:
            spike_ratio = amounts[i] / prior_avg
            spikes.append(
                f"Amount spike at step {steps[i]}: "
                f"{amounts[i]:,.2f} is {spike_ratio:.1f}x the prior average "
                f"({prior_avg:,.2f})"
            )

    if spikes:
        score = min(len(spikes) * 15, 35)
        explanation = "Sudden amount increase: " + "; ".join(spikes)
        return score, explanation
    return 0, ""


def detect_rapid_transactions(transactions):
    """Detect rapid sequence of transactions from the same account.

    Multiple transactions in the same step or consecutive steps indicate
    potentially automated or suspicious behavior.

    Returns a score contribution and explanation if the pattern is found.
    """
    if len(transactions) < 2:
        return 0, ""

    step_counts = transactions.groupby("step").size()
    rapid_steps = step_counts[step_counts > 1]

    if not rapid_steps.empty:
        max_count = int(rapid_steps.max())
        total_rapid = int(rapid_steps.sum())
        score = min(max_count * 10, 25)
        explanation = (
            f"Rapid transactions: {total_rapid} transactions across "
            f"{len(rapid_steps)} step(s), max {max_count} in a single step"
        )
        return score, explanation
    return 0, ""


def detect_high_risk_type_with_suspicious_indicators(transactions):
    """Detect when a customer's transactions involve high-risk types combined
    with suspicious balance or amount indicators.

    For single-transaction customers: flags high-risk type with account drain,
    overdraft, or high amount.
    For multi-transaction customers: flags high concentration of risky types.

    Returns a score contribution and explanation if the pattern is found.
    """
    total = len(transactions)
    if total == 0:
        return 0, ""

    high_risk_count = transactions[transactions["type"].isin(HIGH_RISK_TYPES)].shape[0]

    # For single-transaction customers, flag if it's a high-risk type with
    # suspicious characteristics
    if total == 1 and high_risk_count == 1:
        txn = transactions.iloc[0]
        score = 15
        flags = [f"High-risk type: {txn['type']}"]

        # Account fully drained
        if txn["oldbalanceOrg"] > 0 and txn["newbalanceOrig"] == 0:
            score += 15
            flags.append("account fully drained")

        # Sending more than balance
        if txn["amount"] > txn["oldbalanceOrg"] and txn["oldbalanceOrg"] > 0:
            score += 10
            flags.append(
                f"amount ({txn['amount']:,.2f}) exceeds balance "
                f"({txn['oldbalanceOrg']:,.2f})"
            )

        # High amount
        if txn["amount"] > HIGH_AMOUNT_THRESHOLD:
            score += 10
            flags.append(f"high amount: {txn['amount']:,.2f}")

        if score > 15:
            explanation = "Suspicious single transaction: " + "; ".join(flags)
            return score, explanation

    # For multi-transaction customers
    if total >= 2:
        ratio = high_risk_count / total
        if high_risk_count >= 2 and ratio >= 0.5:
            score = min(int(ratio * 30), 25)
            explanation = (
                f"High-risk type concentration: {high_risk_count}/{total} "
                f"({ratio * 100:.0f}%) transactions are CASH_OUT or TRANSFER"
            )
            return score, explanation

    return 0, ""


def detect_balance_anomalies(transactions):
    """Detect balance-related anomalies in a customer's transactions.

    Flags patterns like destination balance not reflecting received funds
    (possible mule account indicator).

    Returns a score contribution and explanation if the pattern is found.
    """
    flags = []
    score = 0

    for _, txn in transactions.iterrows():
        # Destination balance unchanged despite receiving funds (mule account)
        if (
            txn["amount"] > 0
            and txn["oldbalanceDest"] > 0
            and txn["newbalanceDest"] == 0
        ):
            score += 10
            flags.append(
                f"Possible mule: {txn['nameDest']} balance zeroed after "
                f"receiving {txn['amount']:,.2f}"
            )

    if flags:
        score = min(score, 20)
        explanation = "Balance anomalies: " + "; ".join(flags)
        return score, explanation
    return 0, ""


def assign_anomaly_level(score):
    """Assign anomaly level based on cumulative anomaly score."""
    if score < LOW_THRESHOLD:
        return "LOW"
    elif score <= HIGH_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"


def build_sequence_pattern(transactions):
    """Build a readable string representing the customer's transaction sequence."""
    parts = []
    for _, txn in transactions.iterrows():
        parts.append(f"{txn['type']}({txn['amount']:,.2f})")
    return " -> ".join(parts)


def analyze_customer_sequences(df):
    """Analyze transaction sequences for each customer and detect anomalies.

    Groups transactions by customer (nameOrig), sorts by step, and runs
    all anomaly detection checks including cross-account chain analysis.

    Returns a list of report rows.
    """
    report_rows = []
    orig_index = build_origin_index(df)

    grouped = df.groupby("nameOrig")
    for customer_id, group in grouped:
        transactions = group.sort_values("step").reset_index(drop=True)

        # Run all anomaly detection checks
        checks = [
            detect_repeated_high_value(transactions),
            detect_transfer_then_cashout(transactions, orig_index),
            detect_sudden_amount_increase(transactions),
            detect_rapid_transactions(transactions),
            detect_high_risk_type_with_suspicious_indicators(transactions),
            detect_balance_anomalies(transactions),
        ]

        total_score = 0
        explanations = []
        for score, explanation in checks:
            total_score += score
            if explanation:
                explanations.append(explanation)

        # Cap score at 100
        total_score = min(total_score, 100)
        anomaly_level = assign_anomaly_level(total_score)
        sequence_pattern = build_sequence_pattern(transactions)

        combined_explanation = "; ".join(explanations) if explanations else "No anomalous patterns detected"

        report_rows.append(
            {
                "customer_id": customer_id,
                "transaction_count": len(transactions),
                "sequence_pattern": sequence_pattern,
                "anomaly_score": total_score,
                "anomaly_level": anomaly_level,
                "explanation": combined_explanation,
            }
        )

    return report_rows


def generate_report(report_rows, output_filepath):
    """Write the anomaly report to a CSV file."""
    report_df = pd.DataFrame(report_rows)
    report_df = report_df.sort_values("anomaly_score", ascending=False).reset_index(
        drop=True
    )
    report_df.to_csv(output_filepath, index=False)
    return report_df


def print_summary(report_df):
    """Print a human-readable summary of the anomaly detection results."""
    total = len(report_df)
    summary = report_df["anomaly_level"].value_counts()

    print("=" * 80)
    print("SEQUENCE ANOMALY DETECTION REPORT SUMMARY")
    print("=" * 80)
    print(f"Total customers analyzed: {total}")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = summary.get(level, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {level:<8} anomaly level: {count:>4} ({pct:.1f}%)")
    print()

    high_anomaly = report_df[report_df["anomaly_level"] == "HIGH"]
    if not high_anomaly.empty:
        print(f"HIGH ANOMALY CUSTOMERS ({len(high_anomaly)}):")
        print("-" * 80)
        print(f"{'Customer':<18} {'Txn Count':>10} {'Score':>8} {'Explanation'}")
        print("-" * 80)
        for _, row in high_anomaly.iterrows():
            explanation_short = row["explanation"][:60] + (
                "..." if len(row["explanation"]) > 60 else ""
            )
            print(
                f"{row['customer_id']:<18} {row['transaction_count']:>10} "
                f"{row['anomaly_score']:>8} {explanation_short}"
            )
    print()

    medium_anomaly = report_df[report_df["anomaly_level"] == "MEDIUM"]
    if not medium_anomaly.empty:
        print(f"MEDIUM ANOMALY CUSTOMERS ({len(medium_anomaly)}):")
        print("-" * 80)
        print(f"{'Customer':<18} {'Txn Count':>10} {'Score':>8} {'Explanation'}")
        print("-" * 80)
        for _, row in medium_anomaly.iterrows():
            explanation_short = row["explanation"][:60] + (
                "..." if len(row["explanation"]) > 60 else ""
            )
            print(
                f"{row['customer_id']:<18} {row['transaction_count']:>10} "
                f"{row['anomaly_score']:>8} {explanation_short}"
            )
    print("=" * 80)


def main():
    print(f"Loading transactions from: {INPUT_FILE}")
    df = load_dataset(INPUT_FILE)
    print(f"Loaded {len(df)} transactions from {df['nameOrig'].nunique()} customers.")

    print("Analyzing transaction sequences for anomalies...")
    report_rows = analyze_customer_sequences(df)

    print(f"Saving anomaly report to: {OUTPUT_FILE}")
    report_df = generate_report(report_rows, OUTPUT_FILE)

    print()
    print_summary(report_df)


if __name__ == "__main__":
    main()
