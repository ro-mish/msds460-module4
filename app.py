import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from itertools import combinations

# Set page config
st.set_page_config(
    page_title="Shortest Path Optimizer",
    page_icon="🗺️",
    layout="wide"
)

# Initialize session state for storing the graph data
if 'distances' not in st.session_state:
    st.session_state.distances = {
        ('Origin', 'A'): 40, ('Origin', 'B'): 60, ('Origin', 'C'): 50,
        ('A', 'B'): 10, ('B', 'C'): 20, ('B', 'D'): 55, ('B', 'E'): 40,
        ('C', 'D'): 70, ('C', 'E'): 50, ('D', 'E'): 10, ('D', 'Destination'): 60,
        ('E', 'Destination'): 80
    }

def create_graph(distances):
    G = nx.Graph()
    for (source, target), distance in distances.items():
        G.add_edge(source, target, weight=distance)
    return G

def find_shortest_path(G, start='Origin', end='Destination'):
    try:
        path = nx.shortest_path(G, start, end, weight='weight')
        distance = nx.shortest_path_length(G, start, end, weight='weight')
        return path, distance
    except nx.NetworkXNoPath:
        return None, float('inf')

def create_network_plot(G, path=None):
    # Get position layout for nodes
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edges trace
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = G[edge[0]][edge[1]]['weight']
        edge_text.append(f"{edge[0]} to {edge[1]}: {weight}")
    
    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')

    # Create nodes trace
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    nodes_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            size=30,
            line_width=2))

    # Create highlighted path trace if path exists
    path_trace = None
    if path:
        path_x = []
        path_y = []
        for i in range(len(path)-1):
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i+1]]
            path_x.extend([x0, x1, None])
            path_y.extend([y0, y1, None])
        
        path_trace = go.Scatter(
            x=path_x, y=path_y,
            line=dict(width=3, color='red'),
            hoverinfo='none',
            mode='lines')

    # Create the figure
    fig = go.Figure(
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    # Add all traces
    fig.add_trace(edges_trace)
    fig.add_trace(nodes_trace)
    if path_trace:
        fig.add_trace(path_trace)
    
    return fig

def main():
    st.title("🗺️ Interactive Shortest Path Optimizer")
    st.write("""
    This application helps you find the optimal route between towns based on different metrics.
    You can modify the distances between towns and see how it affects the optimal path.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Network Configuration")
        st.subheader("Modify Distances")
        
        # Get all nodes
        nodes = sorted(list(set([node for edge in st.session_state.distances.keys() for node in edge])))
        
        # Create input fields for each possible connection
        new_distances = {}
        for source, target in combinations(nodes, 2):
            current_value = st.session_state.distances.get((source, target)) or \
                          st.session_state.distances.get((target, source))
            if current_value is not None:
                value = st.number_input(
                    f"{source} to {target}",
                    min_value=0.0,
                    value=float(current_value),
                    step=1.0,
                    key=f"{source}_{target}"
                )
                if value > 0:
                    new_distances[(source, target)] = value
        
        # Update button
        if st.button("Update Network"):
            st.session_state.distances = new_distances
    
    # Metric selection
    metric = st.selectbox(
        "Select optimization metric",
        ["Distance (miles)", "Cost (dollars)", "Time (minutes)"]
    )
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Network Visualization", "Distance Matrix", "Results"])
    
    # Get the graph based on the selected metric
    G = create_graph(st.session_state.distances)
    path, total_distance = find_shortest_path(G)
    
    with tab1:
        st.subheader("Network Visualization")
        fig = create_network_plot(G, path)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Distance Matrix")
        # Convert the distance dictionary to a DataFrame for better visualization
        nodes = sorted(list(set([node for edge in st.session_state.distances.keys() for node in edge])))
        matrix = pd.DataFrame(index=nodes, columns=nodes)
        
        for (source, target), distance in st.session_state.distances.items():
            matrix.loc[source, target] = distance
            matrix.loc[target, source] = distance
        
        # Create a styled dataframe - fixed styling to avoid the error
        st.dataframe(matrix.fillna('-'), use_container_width=True)
        
    with tab3:
        st.subheader("Shortest Path Results")
        if path:
            # Create a summary card using st.container
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Distance", f"{total_distance:.1f} {metric.split()[0].lower()}")
                with col2:
                    st.metric("Number of Stops", len(path) - 2)  # Excluding origin and destination
            
            # Path visualization
            st.write("### Optimal Route")
            st.write(f"**Path**: {' → '.join(path)}")
            
            # Detailed breakdown
            st.write("### Segment Breakdown")
            segments_df = pd.DataFrame(columns=["From", "To", metric])
            for i in range(len(path)-1):
                start, end = path[i], path[i+1]
                segment_distance = G[start][end]['weight']
                segments_df.loc[i] = [start, end, segment_distance]
            
            # Fixed styling - use basic formatting without complex styling
            st.dataframe(segments_df, use_container_width=True)
        else:
            st.error("No valid path found!")

if __name__ == "__main__":
    main() 