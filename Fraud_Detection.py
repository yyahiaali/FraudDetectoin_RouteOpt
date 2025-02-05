#!/usr/bin/env python
# coding: utf-8

# In[117]:


"""
Approach: 
There are some cases to identify fraud
1. Frequent Failed Updates:A high number of failures within a short perio, indicating tryingmultiple times to authenticate with
wrong credentials
2. Multiple Device Usage: Using differnt devices to access wallets
3. Multiple Customer_ID for the same wallet
"""


# ## Loading Packages

# In[83]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from functools import lru_cache


# ## Data Overview

# In[84]:


file_path = "C:\\Users\\yymahmoudali\\Downloads\\wallet_fraud_task.xlsx"
df = pd.read_excel(file_path, sheet_name="Query result")


# In[85]:


# Display basic  info
df.head()


# In[86]:


df.info()


# In[87]:


##df.describe()


# In[88]:


df.isnull().sum()


# In[89]:


# Drop null rows if exist
initial_rows = df.shape[0]
df_cleaned = df.dropna()
final_rows = df_cleaned.shape[0]
rows_dropped = initial_rows - final_rows
print(f"Number of null rows dropped: {rows_dropped}")


# ### Case No.1: Failing many times in short period

# In[90]:


# Showing teh istribution of Wallet Statuses
plt.figure(figsize=(5,3))
df['STATUS_'].value_counts().plot(kind='bar')
plt.title("Wallet Statuses Distribution")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()


# In[91]:


# Matrix for status transitions
transition_matrix = pd.crosstab(df['STATUS_'], df['STATUS_'].shift(-1), rownames=['From'], colnames=['To'])
plt.figure(figsize=(6, 4))
sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Status Heatmap")
plt.show()


# In[92]:


data_sorted = df.sort_values(by=['customer_id_', 'created_at_'])
data_sorted.head()


# In[109]:


# Sort data by wallet number and creation date
sorted_data = data_sorted[data_sorted['STATUS_'] == 'FAILED'].sort_values(by=['wallet_number_', 'created_at_'])

# Count failed attempts per wallet
failed_attempts_count = sorted_data.groupby('wallet_number_').size()

# Identify wallets with more than 2 failures
multiple_failures = failed_attempts_count[failed_attempts_count > 2]

# Calculate time difference from the first failure for each wallet
sorted_data['time_since_first_failure'] = sorted_data.groupby('wallet_number_')['created_at_'].transform(lambda x: x - x.iloc[0])

# Format the duration into days, hours, and minutes
sorted_data['formatted_duration'] = sorted_data['time_since_first_failure'].apply(
    lambda x: f"{x.days} days"
)

# Summarize the data with wallet number, number of failures, and duration from the first failure
summary = sorted_data.groupby('wallet_number_').agg(
    total_failures=('STATUS_', 'size'),
    max_duration=('time_since_first_failure', 'max')
).reset_index()

# Filter for wallets with more than 2 failures
summary = summary[summary['total_failures'] > 2]

# Add formatted duration to the summary
summary['formatted_duration'] = summary['max_duration'].apply(
    lambda x: f"{x.days} days"
)

# Reset index and display the final summary
final_summary = summary.reset_index()
print(final_summary[['wallet_number_', 'total_failures', 'formatted_duration']])


# ### Case No.2: Multiple Device Usage

# In[111]:


# Count the number of unique devices used by customers
device_counts = df.groupby('customer_id_')['device_id_'].nunique()
plt.figure(figsize=(8, 6))
device_counts.value_counts().plot(kind='bar')
plt.title("Distribution of Number of Devices per Customer")
plt.xlabel("Number of Devices")
plt.ylabel("Count of Customers")
plt.show()


# In[114]:


# Identify wallets with multiple device usage
device_usage = data_sorted.groupby(['customer_id_', 'wallet_number_'])['device_id_'].nunique()

# Filter wallets that have been accessed by more than one device
# usually it will be installed on one phone
wallets_with_multiple_devices = device_usage[device_usage > 1]

# Display the results
print("\nWallets with Multiple Device Usage: ")
print(wallets_with_multiple_devices)


# ### Case No.3 Multiple Customer_ID for the same wallet
# 

# In[132]:


# Count unique customer IDs for each wallet
unique_customers_per_wallet = data_sorted.groupby('wallet_number_')['customer_id_'].nunique()

# Identify wallets linked to more than one customer ID
suspicious_wallets = unique_customers_per_wallet[unique_customers_per_wallet > 1]

# Create a DataFrame and rename columns for better understanding
suspicious_wallets_df = suspicious_wallets.reset_index()
suspicious_wallets_df.columns = ['wallet_number_', 'num_unique_customers']

# Display the first 30 results
print("Wallets Associated with Multiple Customer IDs:")
print(suspicious_wallets_df)


# In[ ]:




