#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
2. Route Optimization
In the atta
1. Determine the shortest route in each area that covers all customers.
ched file named "routes_task.xlsx", you will find location data for different customers, their collection amounts, and the areas they belong to.

Task:
2. Identify the shortest route in each area that minimizes the distance while maximizing the collected amount. Note that the maximum collected amount per route (per area) is limited to 300,000.
3. Identify the shortest route in each area that minimizes the distance while maximizing customers per route. Note that the maximum customers amount per route (per area) is limited to 30. (bonus)

Deliverable:
Submit a Jupyter Notebook that includes your data exploration, analysis, and route optimization approach. Feel free to use different ready packages in python.

The duration for completing this case study is 7 days from the date of receiving this email.
"""


# In[ ]:


Remember Tooooooooooooo: 
    Convert total_distance unit to km


# ## Loading Packages 

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from functools import lru_cache
from geopy.distance import geodesic


# ## EDA

# ### Loading File 

# In[2]:


file_path = "C:\\Users\\yymahmoudali\\Downloads\\routes_task.xlsx"
df = pd.read_excel(file_path, sheet_name="Query result")
# Display basic info
display(df.head())


# ### Data Overview

# In[3]:


display(df.info())


# In[4]:


# No type conversion needed!


# In[5]:


summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)


# In[6]:


missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)


# In[7]:


# Number of areas exist in thhe dataset
num_areas = df["area_id"].nunique()
print("\nNumber of Existing Areas:", num_areas)


# ## Determine the shortest route in each area that covers all customers

# In[9]:


# 5. Customer Locations with Amount Collected
plt.figure(figsize=(8, 5))
sc = plt.scatter(df["longitude_"], df["latitude_"], c=df["amount_"], cmap="viridis", alpha=0.7)
plt.title("Customer Locations with Amount Collected")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(sc, label="Amount Collected")
plt.show()


# ### - Assuming the collecter needs to return back to the office (starting point) to return the money

# In[10]:


"""
Task No.Approach: 
- This is a Travelling Salesman Problem (TSP)
- Find the shortest possible route that visits all locations in an area exactly once and returns to the starting point
- Using the dynamic programming (Held-Karp) solution complexity of  O(n² * 2ⁿ)

Steps: 
1. Grouping by area_id to get the best route for each area
2. Building a distance matrix for each area
3. Applying the TSP dynamic programming approach
4. Returning the best route for each area 
"""


# In[8]:


#pip install ortools


# In[15]:


def compute_distance_km(coord1, coord2):
    return geodesic(coord1, coord2).kilometers


# In[16]:


# Enhance an initial route by eliminating crossing edges using the 2-opt algorithm
def two_opt(route, distance_matrix):
    def total_distance(route):
        return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

    def swap_2opt(route, i, k):
        return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

    best_route = route
    best_distance = total_distance(route)
    improvement = True

    while improvement:
        improvement = False
        for i in range(1, len(route) - 2):
            for k in range(i + 1, len(route) - 1):
                new_route = swap_2opt(best_route, i, k)
                new_distance = total_distance(new_route)
                if new_distance < best_distance:
                    best_route, best_distance = new_route, new_distance
                    improvement = True

    return best_route, best_distance


# In[21]:


# Calculate the optimized TSP routes with geodesic distance
def calculate_optimized_tsp_routes(dataframe):
    optimized_results = []

    for area_id, group_data in dataframe.groupby("area_id"):
        coordinates = group_data[["longitude_", "latitude_"]].values

        # Compute the distance matrix using geodesic distances
        dist_matrix = []
        for i in range(len(coordinates)):
            row = []
            for j in range(len(coordinates)):
                distance = compute_distance_km(coordinates[i], coordinates[j])
                row.append(distance)
            dist_matrix.append(row)

        dist_matrix = squareform(pdist(coordinates, metric='euclidean'))  # Optional for checking results, will use geodesic now

        # Create an initial route using the Nearest Neighbor Heuristic
        initial_path = nearest_neighbor_heuristic(dist_matrix)

        # Optimize the route using the 2-Opt algorithm
        best_route, route_distance = two_opt(initial_path, dist_matrix)

        # Map route indices to customer IDs
        customer_order = group_data.iloc[best_route].customer_id_.tolist()

        optimized_results.append({"area_id": area_id, "route": customer_order, "total_distance (km)": route_distance})

    return pd.DataFrame(optimized_results)


# In[22]:


optimized_routes_df = calculate_optimized_tsp_routes(df)
optimized_routes_df.to_excel("C:\\Users\\yymahmoudali\\Downloads\\optimized_routes.xlsx", index=False)
print("Saved to optimized_routes.xlsx Succesfully!")


# ## Identify the shortest route in each area that minimizes the distance while maximizing the collected amount. Note that the maximum collected amount per route (per area) is limited to 300,000.

# In[62]:


#pip install geopy


# In[23]:


def compute_distance_matrix(locations):
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = compute_distance_km(locations[i], locations[j])
    return distance_matrix


# In[24]:


def nearest_neighbor_heuristic(distance_matrix, amounts, max_amount=300000):
    n = len(distance_matrix)
    unvisited = set(range(1, n))  # Start at node 0
    route = [0]
    collected_amount = amounts[0]

    while unvisited:
        last = route[-1]
        nearest = min(unvisited, key=lambda x: distance_matrix[last, x])
        if collected_amount + amounts[nearest] <= max_amount:
            route.append(nearest)
            unvisited.remove(nearest)
            collected_amount += amounts[nearest]
        else:
            break
    route.append(0) 
    return route


# In[25]:


def two_opt(route, distance_matrix):
    def compute_distance(route):
        return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

    best_route = route
    best_distance = compute_distance(route)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                new_distance = compute_distance(new_route)
                if new_distance < best_distance:
                    best_route, best_distance = new_route, new_distance
                    improved = True

    return best_route, best_distance


# In[26]:


def compute_optimized_tsp_routes(df):
    results = []

    for area, group in df.groupby("area_id"):
        locations = group[["latitude_", "longitude_"]].values
        amounts = group["amount_"].values

        # Compute pairwise geodesic distance matrix in km
        distance_matrix = compute_distance_matrix(locations)

        # Generate initial route using Nearest Neighbor Heuristic with amount constraint
        initial_route = nearest_neighbor_heuristic(distance_matrix, amounts)

        # Improve the route using the 2-Opt heuristic
        optimized_route, total_distance = two_opt(initial_route, distance_matrix)

        # Convert route indices to customer IDs
        ordered_customers = group.iloc[optimized_route].customer_id_.tolist()
        collected_amount = sum(group.iloc[optimized_route].amount_)

        results.append({"area_id": area, "route": ordered_customers, "total_distance (km)": total_distance, "collected_amount": collected_amount})

    return pd.DataFrame(results)


# In[27]:


optimized_tsp_routes_df = compute_optimized_tsp_routes(df)
optimized_tsp_routes_df.to_excel("C:\\Users\\yymahmoudali\\Downloads\\optimized_routes_30k.xlsx", index=False)
print("Saved to optimized_routes_30k.xlsx Succesfully!")


# ## Identify the shortest route in each area that minimizes the distance while maximizing customers per route. Note that the maximum customers amount per route (per area) is limited to 30. (bonus)

# In[28]:


def compute_distance_matrix(locations):
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = compute_distance_km(locations[i], locations[j])
    return distance_matrix

def nearest_neighbor_heuristic(distance_matrix, max_customers=30):
    n = len(distance_matrix)
    unvisited = set(range(1, n))  # Start at node 0
    route = [0]

    while unvisited and len(route) < max_customers:
        last = route[-1]
        nearest = min(unvisited, key=lambda x: distance_matrix[last, x])
        route.append(nearest)
        unvisited.remove(nearest)
    
    route.append(0)  # Return to start
    return route

def two_opt(route, distance_matrix):
    def compute_distance(route):
        return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

    best_route = route
    best_distance = compute_distance(route)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                new_distance = compute_distance(new_route)
                if new_distance < best_distance:
                    best_route, best_distance = new_route, new_distance
                    improved = True

    return best_route, best_distance

def compute_optimized_tsp_routes(df):
    results = []

    for area, group in df.groupby("area_id"):
        locations = group[["latitude_", "longitude_"]].values

        # Compute pairwise geodesic distance matrix in km
        distance_matrix = compute_distance_matrix(locations)

        # Generate initial route using Nearest Neighbor Heuristic with customer limit
        initial_route = nearest_neighbor_heuristic(distance_matrix)

        # Improve the route using the 2-Opt heuristic
        optimized_route, total_distance = two_opt(initial_route, distance_matrix)

        # Convert route indices to customer IDs
        ordered_customers = group.iloc[optimized_route].customer_id_.tolist()

        results.append({"area_id": area, "route": ordered_customers, "total_distance (km)": total_distance, "customers_count": len(ordered_customers) - 1})

    return pd.DataFrame(results)


# In[29]:


optimized_tsp_routes_df = compute_optimized_tsp_routes(df)
optimized_tsp_routes_df.to_excel("C:\\Users\\yymahmoudali\\Downloads\\optimized_routes_customers_pt3.xlsx", index=False)
print("Saved to optimized_routes_customers_pt3.xlsx Succesfully!")


# In[ ]:





# In[ ]:




