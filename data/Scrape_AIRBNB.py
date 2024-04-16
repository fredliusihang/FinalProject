#%%

import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

#%%
# URL of the webpage containing the links to CSV files
url = "http://insideairbnb.com/get-the-data/"

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the webpage
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all links in the webpage
    links = soup.find_all("a")

    # List to store all the listings for USA.
    usa_listings = []

    # Iterate through the links
    for link in links:
        href = link.get("href")
        if href and ("united-states" in href) and href.endswith("listings.csv.gz"):
            usa_listings.append(href)
else:
    print("Failed to fetch webpage")

#%%
    
len(usa_listings) # Links for listings data for 34 cities in USA.
usa_listings[:5] # Links where the files are stored.

#%%

# Load the data for cities.
city_data = []

for listing in tqdm(usa_listings):
  df = pd.read_csv(listing)
  df["link"] = listing
  city_data.append(df)
#%%
  
# Concatenate all the cities data to create a single dataframe.
usa_data = pd.concat(city_data)
display(usa_data.head())
print(f"Dataset Shape: {usa_data.shape}")

#%%
listings = usa_data.copy()
listings.reset_index(drop=True, inplace=True)
listings

#%%
listings.info()
#%%
listings.columns
#%%
listings.iloc[0].values