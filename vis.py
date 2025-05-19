import matplotlib.pyplot as plt
import networkx as nx

def create_nx_graph_from_edges(edges):
    """
    Convert a list of DFSCode edges into a NetworkX graph.
    Each edge has the format: (v1_id, v2_id, label1, edge_label, label2).
    
    We'll use 'v1_id' and 'v2_id' as the graph's node identifiers,
    and store 'label1'/'label2' in a dict so we can display them.
    
    We ignore 'edge_label' (like 'PPI') per request.
    """
    G = nx.Graph()
    node_labels = {}
    
    for (v1, v2, lbl1, _, lbl2) in edges:
        # Ensure both v1 and v2 are in the graph with their labels
        if v1 not in node_labels:
            node_labels[v1] = lbl1
        # If there's a conflict, pick one or trust they match (usually they do).
        
        if v2 not in node_labels:
            node_labels[v2] = lbl2
        
        # Add the edge (v1, v2) ignoring edge label
        G.add_edge(v1, v2)
    
    return G, node_labels

def draw_subgraph(ax, edges, title):
    """
    Create a NetworkX graph from the edges, then plot on the given Axes 'ax'
    with node labels. 'title' is used to label the subplot.
    """
    G, node_labels = create_nx_graph_from_edges(edges)
    
    # Layout for the graph
    pos = nx.spring_layout(G, seed=42)  # fixed seed for reproducible layout
    
    # Draw nodes and edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=1500)
    
    # Create a dict: {node: "Label"} for labeling
    # node_labels[node_id] = actual_label (like 'MDT', 'O', etc.)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_size=10)
    
    ax.set_title(title)
    ax.axis('off')

def main():
    # 1) Single edge: MDT -- O (very frequent)
    edges_1 = [(0, 1, 'MDT', 'PPI', 'O')]

    # 2) Single edge: C -- O
    edges_2 = [(0, 1, 'C', 'PPI', 'O')]

    # 3) Three edges, all O
    edges_3 = [
        (0, 1, 'O', 'PPI', 'O'),
        (1, 2, 'O', 'PPI', 'O'),
        (1, 3, 'O', 'PPI', 'O')
    ]

    # 4) Three edges, with O, T, R
    edges_4 = [
        (0, 1, 'O', 'PPI', 'T'),
        (1, 2, 'T', 'PPI', 'R'),
        (1, 3, 'T', 'PPI', 'T')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Flatten the Axes array for easier indexing
    ax_list = axes.flatten()
    
    draw_subgraph(ax_list[0], edges_1, "Single Edge (MDT -- O)")
    draw_subgraph(ax_list[1], edges_2, "Single Edge (C -- O)")
    draw_subgraph(ax_list[2], edges_3, "3-Edge (All O)")
    draw_subgraph(ax_list[3], edges_4, "3-Edge (O, T, R)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
