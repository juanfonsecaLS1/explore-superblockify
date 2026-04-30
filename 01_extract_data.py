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


cities = [
    {"name": "Lima", "lat": -12.077948042115647, "lon": -77.05096688507561},
    {"name": "Brussels", "lat": 50.856117793777145, "lon": 4.381718424984715},
    {"name": "Rome", "lat": 41.89900799198937, "lon": 12.512797548177907},
    {"name": "Cairo", "lat": 30.036716997454857, "lon": 31.23947439245143},
    {"name": "Salt Lake City", "lat": 40.75984495218752, "lon": -111.88769690322184},
    {"name": "Bogota", "lat": 4.691389013888271, "lon": -74.06823532051276},
    {"name": "Leeds", "lat": 53.81303429075668,  "lon": -1.508881353201933},
    {"name": "Milton Keynes", "lat": 52.0367171540023, "lon": -0.7324001661371712},
    {"name": "Lisbon", "lat": 38.72778849555562, "lon": -9.162267565552924},
    {"name": "New York", "lat": 40.7589664816669, "lon": -73.96121069380527},
    {"name": "Brasilia", "lat": -15.76133509896713, "lon": -47.88377302616466},
    {"name": "Sao Paulo", "lat": -23.54468352821393, "lon": -46.67047368527096},

]

BUFFER_M = 3500 
NETWORK_TYPE = "drive"
UNIT = "time"

for city_info in cities:
    
    city_name = city_info["name"]
              
    graph_file = f"raw_graphs/{city_name}.graphml"
    if not os.path.exists(graph_file):
         graph = ox.graph_from_point((city_info["lat"], city_info["lon"]),
         dist= BUFFER_M,
         network_type=NETWORK_TYPE,
         simplify=True,
         custom_filter=None)
         ox.save_graphml(graph, graph_file)


for city_info in cities:
    city_name = city_info["name"]

    graph_file = f"raw_graphs/{city_name}.graphml"
    G_raw = ox.load_graphml(graph_file)

    if not os.path.exists(f"data/results/{city_name}_residential_test/{city_name}_residential_test.partitioner"):
    
        G_processed = preprocess_graph(G_raw)
    
        # Running the partitioner on the graph 
        part = sb.ResidentialPartitioner(
            name=f"{city_name}_residential_test",
            city_name=city_name,
            unit="time",
            graph=G_processed
            )
    
        part.run(
            calculate_metrics=True,
            make_plots=False,
            replace_max_speeds=False
            )
    
        part.save(save_graph_copy=True)
        del part
