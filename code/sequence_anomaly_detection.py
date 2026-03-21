"""
Sequence Anomaly Detection System

Detects anomalous transaction sequences for each customer by analyzing
patterns in transaction types, amounts, and timing. Includes both
per-customer sequence analysis and cross-account linked chain detection.

Suspicious Patterns Detected:
1. Repeated high-value transactions - Multiple transactions above threshold
   from the same customer
2. Transfer followed by cash-out - Common fraud pattern where funds are
   transferred then immediately cashed out, detected via balance-linked
   account chains
3. Sudden increase in transaction amounts - Unusual spikes in sequential
   transaction amounts for the same customer

Anomaly Levels:
- LOW: Minor deviations, score < 40
- MEDIUM: Moderate suspicious activity, score between 40 and 70
- HIGH: Strongly anomalous sequences, score > 70
"""

import os

import pandas as pd

INPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "Example1.csv")
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sequence_anomaly_report.csv"
)

HIGH_VALUE_THRESHOLD = 10000
HIGH_RISK_TYPES = {"CASH_OUT", "TRANSFER"}


def load_dataset(filepath):
    """Load and parse the transaction dataset."""
    df = pd.read_csv(filepath)
    df["step"] = df["step"].astype(int)
    df["amount"] = df["amount"].astype(float)
    df["oldbalanceOrg"] = df["oldbalanceOrg"].astype(float)
    df["newbalanceOrig"] = df["newbalanceOrig"].astype(float)
    df["oldbalanceDest"] = df["oldbalanceDest"].astype(float)
    df["newbalanceDest"] = df["newbalanceDest"].astype(float)
    df["isFraud"] = df["isFraud"].astype(int)
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype(int)
    return df


def group_and_sort_transactions(df):
    """Group transactions by customer (nameOrig) and sort by step."""
    df_sorted = df.sort_values(by=["nameOrig", "step"]).reset_index(drop=True)
    grouped = df_sorted.groupby("nameOrig")
    return grouped


def detect_repeated_high_value(transactions):
    """Detect repeated high-value transactions for a customer.

    Returns (detected, count, description).
    Flags customers with 2+ transactions above the threshold.
    """
    high_value_txns = transactions[transactions["amount"] > HIGH_VALUE_THRESHOLD]
    count = len(high_value_txns)
    if count >= 2:
        types_involved = " -> ".join(high_value_txns["type"].tolist())
        amounts = [f"{a:,.2f}" for a in high_value_txns["amount"].tolist()]
        description = (
            f"Repeated high-value transactions ({count}x above "
            f"{HIGH_VALUE_THRESHOLD:,}): types=[{types_involved}], "
            f"amounts=[{', '.join(amounts)}]"
        )
        return True, count, description
    return False, 0, ""


def detect_transfer_then_cashout_in_sequence(transactions):
    """Detect TRANSFER followed by CASH_OUT within a single customer's sequence.

    Returns (detected, count, description).
    """
    types_list = transactions["type"].tolist()
    amounts_list = transactions["amount"].tolist()
    pattern_count = 0
    details = []

    for i in range(len(types_list) - 1):
        if types_list[i] == "TRANSFER" and types_list[i + 1] == "CASH_OUT":
            pattern_count += 1
            details.append(
                f"TRANSFER({amounts_list[i]:,.2f})->CASH_OUT({amounts_list[i+1]:,.2f})"
            )

    if pattern_count > 0:
        description = (
            f"Transfer followed by cash-out detected ({pattern_count}x): "
            f"{'; '.join(details)}"
        )
        return True, pattern_count, description
    return False, 0, ""


def detect_sudden_amount_increase(transactions):
    """Detect sudden increase in transaction amounts.

    Returns (detected, max_ratio, description).
    A ratio > 5x between consecutive transactions is flagged.
    """
    amounts = transactions["amount"].tolist()
    types_list = transactions["type"].tolist()
    if len(amounts) < 2:
        return False, 0.0, ""

    max_ratio = 0.0
    spike_details = []

    for i in range(1, len(amounts)):
        prev_amount = amounts[i - 1]
        curr_amount = amounts[i]
        if prev_amount > 0:
            ratio = curr_amount / prev_amount
            if ratio > 5.0:
                if ratio > max_ratio:
                    max_ratio = ratio
                spike_details.append(
                    f"{types_list[i-1]}({prev_amount:,.2f})->"
                    f"{types_list[i]}({curr_amount:,.2f}) [{ratio:.1f}x increase]"
                )

    if spike_details:
        description = (
            f"Sudden amount increase detected (max {max_ratio:.1f}x): "
            f"{'; '.join(spike_details)}"
        )
        return True, max_ratio, description
    return False, 0.0, ""


def build_balance_linked_chains(df):
    """Build cross-account chains by matching balance flows.

    Detects patterns where one transaction's resulting balance matches another
    transaction's starting balance, linking accounts in a sequence. This
    identifies the TRANSFER->CASH_OUT pattern across different originators.

    Returns a dict mapping customer_id to list of linked chain descriptions.
    """
    chains = {}

    # Index transactions by their newbalanceOrig for quick lookup
    balance_to_txn = {}
    for idx, row in df.iterrows():
        new_bal = row["newbalanceOrig"]
        if new_bal > 0:
            key = round(new_bal, 2)
            if key not in balance_to_txn:
                balance_to_txn[key] = []
            balance_to_txn[key].append(row)

    # Find transactions whose oldbalanceOrg matches another's newbalanceOrig
    for idx, row in df.iterrows():
        old_bal = round(row["oldbalanceOrg"], 2)
        if old_bal > 0 and old_bal in balance_to_txn:
            for prev_txn in balance_to_txn[old_bal]:
                if prev_txn["nameOrig"] != row["nameOrig"]:
                    prev_type = prev_txn["type"]
                    curr_type = row["type"]
                    customer_id = row["nameOrig"]

                    if customer_id not in chains:
                        chains[customer_id] = []

                    chains[customer_id].append({
                        "prev_customer": prev_txn["nameOrig"],
                        "prev_type": prev_type,
                        "prev_amount": prev_txn["amount"],
                        "curr_type": curr_type,
                        "curr_amount": row["amount"],
                        "is_transfer_cashout": (
                            prev_type in {"TRANSFER", "DEBIT"}
                            and curr_type == "CASH_OUT"
                        ),
                        "shared_balance": old_bal,
                    })

    return chains


def detect_repeated_destination_transfers(df):
    """Detect multiple high-value transfers to the same destination account.

    Returns a dict mapping destination account to originator details.
    """
    transfers = df[
        (df["type"] == "TRANSFER") & (df["amount"] > HIGH_VALUE_THRESHOLD)
    ]
    dest_groups = transfers.groupby("nameDest")

    dest_patterns = {}
    for dest, group in dest_groups:
        if len(group) >= 2:
            originators = group["nameOrig"].tolist()
            amounts = group["amount"].tolist()
            total = sum(amounts)
            dest_patterns[dest] = {
                "originators": originators,
                "amounts": amounts,
                "total_amount": total,
                "count": len(group),
            }

    return dest_patterns


def compute_anomaly_score(
    repeated_high_value,
    high_value_count,
    transfer_cashout_seq,
    transfer_cashout_count,
    sudden_increase,
    max_ratio,
    has_linked_chain,
    linked_chain_is_xfer_co,
    is_dest_concentration,
    single_txn_flags,
):
    """Compute anomaly score (0-100) based on detected patterns.

    Scoring breakdown:
    - Repeated high-value transactions: up to 30 points
    - Transfer->cash-out (within sequence): up to 30 points
    - Sudden amount increase: up to 25 points
    - Cross-account linked chain (balance-linked): up to 25 points
    - Destination concentration (repeated transfers to same dest): up to 15 pts
    - Single-transaction flags (high amount + risky type + balance drain): up to 30 pts
    """
    score = 0.0

    # Repeated high-value transactions (up to 30 points)
    if repeated_high_value:
        score += 15 + min((high_value_count - 1) * 10, 15)

    # Transfer followed by cash-out in sequence (up to 30 points)
    if transfer_cashout_seq:
        score += 20 + min((transfer_cashout_count - 1) * 10, 10)

    # Sudden amount increase (up to 25 points)
    if sudden_increase:
        ratio_score = min((max_ratio - 5) * 3, 15)
        score += 10 + max(ratio_score, 0)

    # Cross-account linked chain (up to 25 points)
    if has_linked_chain:
        score += 15
        if linked_chain_is_xfer_co:
            score += 10

    # Destination concentration
    if is_dest_concentration:
        score += 15

    # Single-transaction flags
    if single_txn_flags:
        flag_score = 0
        if single_txn_flags.get("high_amount"):
            flag_score += 10
        if single_txn_flags.get("risky_type"):
            flag_score += 10
        if single_txn_flags.get("balance_drained"):
            flag_score += 10
        if single_txn_flags.get("overdraft"):
            flag_score += 5
        score += min(flag_score, 30)

    return min(score, 100)


def assign_anomaly_level(score):
    """Assign anomaly level based on score thresholds."""
    if score < 40:
        return "LOW"
    elif score <= 70:
        return "MEDIUM"
    else:
        return "HIGH"


def build_sequence_pattern(transactions):
    """Build a string representation of the transaction sequence."""
    parts = []
    for _, row in transactions.iterrows():
        parts.append(f"{row['type']}({row['amount']:,.2f})")
    return " -> ".join(parts)


def get_single_txn_flags(row):
    """Evaluate single-transaction anomaly flags."""
    flags = {}
    if row["amount"] > HIGH_VALUE_THRESHOLD:
        flags["high_amount"] = True
    if row["type"] in HIGH_RISK_TYPES:
        flags["risky_type"] = True
    if row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] == 0:
        flags["balance_drained"] = True
    if row["amount"] > row["oldbalanceOrg"] and row["oldbalanceOrg"] > 0:
        flags["overdraft"] = True
    return flags if flags else None


def analyze_customer_sequences(df, grouped):
    """Analyze transaction sequences for all customers.

    Combines per-customer sequence analysis with cross-account pattern detection.
    Returns a list of anomaly report rows.
    """
    # Build cross-account analyses
    linked_chains = build_balance_linked_chains(df)
    dest_patterns = detect_repeated_destination_transfers(df)

    # Build a set of customers involved in destination concentration
    dest_concentration_customers = set()
    for dest_info in dest_patterns.values():
        for orig in dest_info["originators"]:
            dest_concentration_customers.add(orig)

    report_rows = []

    for customer_id, transactions in grouped:
        transactions = transactions.sort_values("step").reset_index(drop=True)
        num_txns = len(transactions)

        # Per-customer multi-transaction analysis
        rhv_detected, rhv_count, rhv_desc = detect_repeated_high_value(transactions)
        tc_detected, tc_count, tc_desc = detect_transfer_then_cashout_in_sequence(
            transactions
        )
        si_detected, si_ratio, si_desc = detect_sudden_amount_increase(transactions)

        # Cross-account linked chain analysis
        has_linked_chain = customer_id in linked_chains
        linked_chain_is_xfer_co = False
        chain_desc = ""
        if has_linked_chain:
            chain_info = linked_chains[customer_id]
            chain_parts = []
            for c in chain_info:
                chain_parts.append(
                    f"{c['prev_customer']}:{c['prev_type']}({c['prev_amount']:,.2f})"
                    f"->{customer_id}:{c['curr_type']}({c['curr_amount']:,.2f})"
                    f" [shared balance: {c['shared_balance']:,.2f}]"
                )
                if c["is_transfer_cashout"]:
                    linked_chain_is_xfer_co = True
            chain_desc = (
                f"Balance-linked chain detected: {'; '.join(chain_parts)}"
            )

        # Destination concentration
        is_dest_concentration = customer_id in dest_concentration_customers
        dest_conc_desc = ""
        if is_dest_concentration:
            for dest, info in dest_patterns.items():
                if customer_id in info["originators"]:
                    dest_conc_desc = (
                        f"Part of repeated high-value transfers to {dest} "
                        f"({info['count']}x, total {info['total_amount']:,.2f})"
                    )
                    break

        # Single-transaction flags
        single_txn_flags = None
        single_flag_desc = ""
        if num_txns == 1:
            row = transactions.iloc[0]
            single_txn_flags = get_single_txn_flags(row)
            if single_txn_flags:
                flag_parts = []
                if single_txn_flags.get("high_amount"):
                    flag_parts.append(
                        f"high amount ({row['amount']:,.2f} > {HIGH_VALUE_THRESHOLD:,})"
                    )
                if single_txn_flags.get("risky_type"):
                    flag_parts.append(f"risky transaction type ({row['type']})")
                if single_txn_flags.get("balance_drained"):
                    flag_parts.append("account fully drained")
                if single_txn_flags.get("overdraft"):
                    flag_parts.append("amount exceeds account balance")
                single_flag_desc = (
                    f"Single-transaction flags: {', '.join(flag_parts)}"
                )

        # Check if any anomaly was detected
        any_anomaly = (
            rhv_detected
            or tc_detected
            or si_detected
            or has_linked_chain
            or is_dest_concentration
            or (single_txn_flags is not None)
        )

        if not any_anomaly:
            continue

        # Compute score
        score = compute_anomaly_score(
            rhv_detected,
            rhv_count,
            tc_detected,
            tc_count,
            si_detected,
            si_ratio,
            has_linked_chain,
            linked_chain_is_xfer_co,
            is_dest_concentration,
            single_txn_flags,
        )

        anomaly_level = assign_anomaly_level(score)

        # Build explanation
        explanations = []
        if rhv_detected:
            explanations.append(rhv_desc)
        if tc_detected:
            explanations.append(tc_desc)
        if si_detected:
            explanations.append(si_desc)
        if chain_desc:
            explanations.append(chain_desc)
        if dest_conc_desc:
            explanations.append(dest_conc_desc)
        if single_flag_desc:
            explanations.append(single_flag_desc)

        sequence_pattern = build_sequence_pattern(transactions)

        report_rows.append({
            "customer_id": customer_id,
            "num_transactions": num_txns,
            "sequence_pattern": sequence_pattern,
            "anomaly_score": round(score, 2),
            "anomaly_level": anomaly_level,
            "repeated_high_value": rhv_detected,
            "transfer_then_cashout": tc_detected or linked_chain_is_xfer_co,
            "sudden_amount_increase": si_detected,
            "balance_linked_chain": has_linked_chain,
            "destination_concentration": is_dest_concentration,
            "explanation": " | ".join(explanations),
        })

    return report_rows


def generate_report(report_rows, output_filepath):
    """Generate the sequence anomaly report as a CSV file."""
    columns = [
        "customer_id",
        "num_transactions",
        "sequence_pattern",
        "anomaly_score",
        "anomaly_level",
        "repeated_high_value",
        "transfer_then_cashout",
        "sudden_amount_increase",
        "balance_linked_chain",
        "destination_concentration",
        "explanation",
    ]

    if not report_rows:
        print("No anomalous sequences detected.")
        report_df = pd.DataFrame(columns=columns)
    else:
        report_df = pd.DataFrame(report_rows)
        report_df = report_df.sort_values(
            "anomaly_score", ascending=False
        ).reset_index(drop=True)

    report_df.to_csv(output_filepath, index=False)
    return report_df


def print_summary(report_df):
    """Print a human-readable summary of the anomaly detection results."""
    total = len(report_df)
    print("=" * 80)
    print("SEQUENCE ANOMALY DETECTION REPORT SUMMARY")
    print("=" * 80)
    print(f"Total customers with anomalous sequences: {total}")

    if total == 0:
        print("No anomalous sequences detected.")
        print("=" * 80)
        return

    level_counts = report_df["anomaly_level"].value_counts()
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = level_counts.get(level, 0)
        pct = count / total * 100
        print(f"  {level:<8} anomaly level: {count:>4} ({pct:.1f}%)")
    print()

    # Pattern breakdown
    rhv_count = report_df["repeated_high_value"].sum()
    tc_count = report_df["transfer_then_cashout"].sum()
    si_count = report_df["sudden_amount_increase"].sum()
    blc_count = report_df["balance_linked_chain"].sum()
    dc_count = report_df["destination_concentration"].sum()
    print("Pattern Breakdown:")
    print(f"  Repeated high-value transactions:     {rhv_count}")
    print(f"  Transfer followed by cash-out:        {tc_count}")
    print(f"  Sudden amount increase:               {si_count}")
    print(f"  Balance-linked cross-account chains:  {blc_count}")
    print(f"  Destination concentration:            {dc_count}")
    print()

    # Show HIGH anomaly customers
    high_anomaly = report_df[report_df["anomaly_level"] == "HIGH"]
    if not high_anomaly.empty:
        print(f"HIGH ANOMALY CUSTOMERS ({len(high_anomaly)}):")
        print("-" * 80)
        print(
            f"{'Customer':<18} {'Txns':>5} {'Score':>7} {'Level':<8} "
            f"{'HighVal':>8} {'Xfer->CO':>9} {'Chain':>6} {'DestConc':>9}"
        )
        print("-" * 80)
        for _, row in high_anomaly.iterrows():
            print(
                f"{row['customer_id']:<18} {row['num_transactions']:>5} "
                f"{row['anomaly_score']:>7.1f} {row['anomaly_level']:<8} "
                f"{'YES' if row['repeated_high_value'] else 'NO':>8} "
                f"{'YES' if row['transfer_then_cashout'] else 'NO':>9} "
                f"{'YES' if row['balance_linked_chain'] else 'NO':>6} "
                f"{'YES' if row['destination_concentration'] else 'NO':>9}"
            )

    # Show MEDIUM anomaly customers
    medium_anomaly = report_df[report_df["anomaly_level"] == "MEDIUM"]
    if not medium_anomaly.empty:
        print()
        print(f"MEDIUM ANOMALY CUSTOMERS ({len(medium_anomaly)}):")
        print("-" * 80)
        print(
            f"{'Customer':<18} {'Txns':>5} {'Score':>7} {'Level':<8} "
            f"{'HighVal':>8} {'Xfer->CO':>9} {'Chain':>6} {'DestConc':>9}"
        )
        print("-" * 80)
        for _, row in medium_anomaly.iterrows():
            print(
                f"{row['customer_id']:<18} {row['num_transactions']:>5} "
                f"{row['anomaly_score']:>7.1f} {row['anomaly_level']:<8} "
                f"{'YES' if row['repeated_high_value'] else 'NO':>8} "
                f"{'YES' if row['transfer_then_cashout'] else 'NO':>9} "
                f"{'YES' if row['balance_linked_chain'] else 'NO':>6} "
                f"{'YES' if row['destination_concentration'] else 'NO':>9}"
            )

    print("=" * 80)


def main():
    print(f"Loading transactions from: {INPUT_FILE}")
    df = load_dataset(INPUT_FILE)
    print(
        f"Loaded {len(df)} transactions from "
        f"{df['nameOrig'].nunique()} unique customers."
    )

    print("Grouping transactions by customer and sorting by step...")
    grouped = group_and_sort_transactions(df)

    print("Analyzing transaction sequences for anomalies...")
    report_rows = analyze_customer_sequences(df, grouped)

    print(f"Generating anomaly report to: {OUTPUT_FILE}")
    report_df = generate_report(report_rows, OUTPUT_FILE)

    print()
    print_summary(report_df)


if __name__ == "__main__":
    main()
