import re
import pandas as pd
# import ace_tools as tools

def parse_graph_transactions(filename):
    transactions = {}
    current_transaction = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Detect new transaction (e.g., "t # 0")
            if line.startswith('t #'):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "#":
                    try:
                        current_transaction = int(parts[2])  # Extract transaction number safely
                        transactions[current_transaction] = {"vertices": set(), "edges": 0, "total_prob": 0, "prob_count": 0}
                    except ValueError:
                        print(f"Warning: Skipping malformed transaction line: {line}")
                        continue  # Skip malformed lines

            # Extract vertices (e.g., "v 1497 DL")
            elif line.startswith('v') and current_transaction is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        vertex_id = int(parts[1])
                        transactions[current_transaction]["vertices"].add(vertex_id)
                    except ValueError:
                        print(f"Warning: Skipping malformed vertex line: {line}")
                        continue

            # Extract edges (e.g., "e 381 382 PPI 0.821")
            elif line.startswith('e') and current_transaction is not None:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        node1, node2, label, probability = int(parts[1]), int(parts[2]), parts[3], float(parts[4])
                        transactions[current_transaction]["edges"] += 1
                        transactions[current_transaction]["total_prob"] += probability
                        transactions[current_transaction]["prob_count"] += 1
                    except ValueError:
                        print(f"Warning: Skipping malformed edge line: {line}")
                        continue

    # Convert transactions into structured data
    transaction_stats = []
    for t_id, data in transactions.items():
        vertex_count = len(data["vertices"])
        edge_count = data["edges"]
        avg_prob = data["total_prob"] / data["prob_count"] if data["prob_count"] > 0 else 0
        
        transaction_stats.append({
            "Transaction ID": t_id,
            "Total Vertices": vertex_count,
            "Total Edges": edge_count,
            "Average Probability": avg_prob
        })

    return transaction_stats



# Example usage
filename = "../data/uncertain_dataset_cog_new_filtered.txt"  # Replace with your actual file path
stats = parse_graph_transactions(filename)

print(stats)

# df = pd.DataFrame([stats])
# tools.display_dataframe_to_user(name="Graph Statistics", dataframe=df)
