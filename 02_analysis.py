import osmnx as ox
import networkx as nx
import superblockify.superblockify as sb
from preprocess_graph import preprocess_graph
import momepy as mp
import pandas as pd
import geopandas as gpd
import os
from libpysal import graph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cities_to_process = os.listdir("data/results/")

# Directories
output_dir = "data/cross_city_analysis"
os.makedirs(output_dir, exist_ok=True)
maps_dir = os.path.join(output_dir, "maps")
os.makedirs(maps_dir, exist_ok=True)
geodata_dir = os.path.join(output_dir, "geodata")
os.makedirs(geodata_dir, exist_ok=True)

# =============================================================================
# PART 1: DATA EXTRACTION
# Process graph partitioning, extract geographic enclosures and LTN intersections,
# and save them as Geopackages.
# =============================================================================

for city in cities_to_process:
    city_name = city.replace("_residential_test", "")
    expanded_path = os.path.join(geodata_dir, f"{city_name}_enclosures_expanded.gpkg")
    filtered_path = os.path.join(geodata_dir, f"{city_name}_filtered_enclosures.gpkg")
    
    if os.path.exists(expanded_path) and os.path.exists(filtered_path):
        print(f"Extraction for {city_name} already exists. Skipping graph processing...")
        continue
        
    print(f"Processing and Extracting {city_name} (from {city})...")
    
    part = sb.ResidentialPartitioner.load(f"{city}")
    for u, v, key in part.graph.edges(keys=True):
        part.graph[u][v][key].pop("residential", None)

    part.run()

    parts = part.components if part.components else part.partitions

    for _, mpart in enumerate(parts):
        nx.set_edge_attributes(mpart["subgraph"], mpart["name"], "classification")
    nx.set_edge_attributes(part.sparsified, "SPARSE", "classification")
    nodes, edges = ox.graph_to_gdfs(part.graph, nodes=True, fill_edge_geometry=True)

    sparse_graph = part.sparsified
    boundary = part.graph.graph['boundary']
    sparse_gdf = ox.graph_to_gdfs(sparse_graph.to_undirected(), nodes=False, edges=True)

    residential_edges = edges[edges["classification"] != "SPARSE"]
    
    lnt_areas = residential_edges.union_all().buffer(100)

    enclosures = mp.enclosures(sparse_gdf, lnt_areas)
    enclosures["area"] = enclosures.geometry.area * 1e-6  
    enclosures["perimeter"] = enclosures.geometry.length * 1e-3

    diss_edges = residential_edges.dissolve(by="classification")
    dissolved_edges = gpd.GeoDataFrame(geometry=diss_edges.centroid)

    ltn_data = pd.DataFrame(part.get_ltns()).set_index("name")
    ltn_centroids = dissolved_edges.join(ltn_data)

    joined_ltn_nodes = ltn_centroids.sjoin(enclosures[["eID","geometry"]], how="left", predicate="within")
    enclosure_summary = joined_ltn_nodes.reset_index().groupby("eID").agg(
        ltn_count = ("classification", "count"),
        ltn_pop = ("population", "sum"),
        ltn_length = ("length_total", "sum")
    )

    enclosures_expanded = enclosures.join(enclosure_summary)
    # enclosures_expanded = enclosures_expanded[enclosures_expanded.area < max(enclosures_expanded.area)]
    
    enclosures_expanded["ltn"] = enclosures_expanded["ltn_count"] > 0
    enclosures_expanded["pop_dens_ha"] = enclosures_expanded["ltn_pop"] / (enclosures_expanded["area"] / 10000)

    filtered_enclosures = enclosures_expanded[enclosures_expanded["ltn"]].copy()

    # Save extraction outputs
    enclosures_expanded.to_file(expanded_path, driver="GPKG")
    filtered_enclosures.to_file(filtered_path, driver="GPKG")


# =============================================================================
# PART 2: PLOTTING & DATA ANALYSIS
# Load the pre-processed Geopackages, compute network contiguity models,
# generate static maps, and build the cross-city comparison arrays.
# =============================================================================

all_filtered_enclosures = []
all_component_stats = []

for city in cities_to_process:
    city_name = city.replace("_residential_test", "")
    expanded_path = os.path.join(geodata_dir, f"{city_name}_enclosures_expanded.gpkg")
    filtered_path = os.path.join(geodata_dir, f"{city_name}_filtered_enclosures.gpkg")
    if not (os.path.exists(expanded_path) and os.path.exists(filtered_path)):
        continue
    
    print(f"Analyzing {city_name}...")
    enclosures_expanded = gpd.read_file(expanded_path)
    filtered_enclosures = gpd.read_file(filtered_path)
    
    # 1. Contiguity & Connectivity Graph
    rook = graph.Graph.build_contiguity(filtered_enclosures)
    G_nx = rook.to_networkx()

    filtered_enclosures["connectivity_degree"] = filtered_enclosures.index.map(dict(G_nx.degree()))
    components = list(nx.connected_components(G_nx))

    id_to_component = {}
    for comp_id, component in enumerate(components):
        for node_id in component:
            id_to_component[node_id] = comp_id

    filtered_enclosures["component"] = filtered_enclosures.index.map(id_to_component)
    filtered_enclosures["city"] = city_name
    
    # 2. Static Mapping: Number of LTNs per enclosure
    fig, ax = plt.subplots(figsize=(10, 10))
    enclosures_expanded.plot(
        ax=ax, column="ltn_count", cmap="Reds", edgecolor="black", 
        linewidth=0.5, legend=True, missing_kwds={"color": "white", "edgecolor": "lightgrey"}
    )
    plt.title(f"{city_name} - Number of LTNs per Enclosure")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(maps_dir, f"{city_name}_ltn_count_static.png"), dpi=300)
    plt.close()

    # 3. Static Mapping: Contiguity Graph over Filtered Enclosures
    fig, ax = plt.subplots(figsize=(10, 10))
    enclosures_expanded.plot(ax=ax, color="whitesmoke", edgecolor="lightgrey", linewidth=0.5)
    filtered_enclosures.plot(ax=ax, column="component", cmap="tab20", alpha=0.7, edgecolor="black", linewidth=0.5)
    rook.plot(filtered_enclosures, ax=ax, edge_kws=dict(color="black", linewidth=1.5, alpha=0.8), node_kws=dict(s=0))
    plt.title(f"{city_name} - LTN Enclosures Contiguity Analysis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(maps_dir, f"{city_name}_contiguity_static.png"), dpi=300)
    plt.close()

    # 4. Component Stats Extraction
    component_stats = filtered_enclosures.groupby("component").agg(
        total_area=("area", "sum"),
        mean_area=("area", "mean"),
        sd_area=("area", "std"),
        total_population=("ltn_pop", "sum"),
        mean_population=("ltn_pop", "mean"),
        sd_population=("ltn_pop", "std"),
        component_size=("component", "count")
    ).reset_index()
    component_stats["city"] = city_name

    all_filtered_enclosures.append(pd.DataFrame(filtered_enclosures.drop(columns=["geometry"])))
    all_component_stats.append(component_stats)

# Concatenate all city data
all_enclosures_df = pd.concat(all_filtered_enclosures, ignore_index=True)
all_component_stats_df = pd.concat(all_component_stats, ignore_index=True)

# Generate Cross-City Boxplots/Violin Plots

# Save tables before plotting
all_enclosures_df.to_csv(f"{output_dir}/all_enclosures.csv", index=False)
all_component_stats_df.to_csv(f"{output_dir}/all_component_stats.csv", index=False)

plt.figure(figsize=(12, 6))
sns.boxplot(data=all_enclosures_df, x="city", y="perimeter", color="lightblue")
plt.title("Distribution of Enclosure Perimeter Across Cities")
plt.xlabel("City")
plt.ylabel("Perimeter (km)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/boxplot_perimeter.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=all_enclosures_df, x="city", y="area", color="lightgreen")
plt.title("Distribution of Enclosure Area Across Cities")
plt.xlabel("City")
plt.ylabel("Area (sq km)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/boxplot_area.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=all_enclosures_df, x="city", y="connectivity_degree", color="lightcoral")
plt.title("Distribution of Connectivity Degree (# Neighbors) Across Cities")
plt.xlabel("City")
plt.ylabel("Degree (Number of Neighbors)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/boxplot_degree.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=all_component_stats_df, x="city", y="component_size", color="thistle")
plt.title("Distribution of Connected Component Sizes Across Cities")
plt.xlabel("City")
plt.ylabel("Component Size (Number of Enclosures)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{output_dir}/boxplot_component_size.png", dpi=300)
plt.close()

# Exploratory Pattern Finding

# Area vs Connectivity Degree
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_enclosures_df, x="area", y="connectivity_degree", hue="city", alpha=0.6)
plt.title("Pattern Exploration: Enclosure Area vs Connectivity Degree")
plt.xlabel("Enclosure Area (sq km)")
plt.ylabel("Connectivity Degree (# Neighbors)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_area_vs_degree.png", dpi=300)
plt.close()

# Enclosure Area vs Population
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_enclosures_df, x="area", y="ltn_pop", hue="city", alpha=0.6)
plt.title("Pattern Exploration: Enclosure Area vs Population")
plt.xlabel("Enclosure Area (sq km)")
plt.ylabel("Total Population in LTNs")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_area_vs_pop.png", dpi=300)
plt.close()

# Component Size vs Total Area
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_component_stats_df, x="component_size", y="total_area", hue="city", size="total_population", sizes=(20, 500), alpha=0.6)
plt.title("Pattern Exploration: Component Size vs Total Area")
plt.xlabel("Component Size (Number of Enclosures)")
plt.ylabel("Total Area of Component (sq km)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_compsize_vs_totarea.png", dpi=300)
plt.close()

