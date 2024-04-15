#%%
"""
Text Analysis: Analyzing text data from descriptions, reviews, or host information to gauge sentiment or extract themes.
Predictive Modeling: Developing models to predict prices, review scores, or host response times based on various predictors.
Cluster Analysis: Grouping listings or hosts based on similarities in features to identify patterns or segments in the market.
"""
"""
Data Cleaning: Check for missing values, inconsistent entries, and duplicates.
Exploratory Data Analysis (EDA): Analyze distributions of key variables like price, minimum nights, number of reviews, and room types.
Geographical Analysis: Explore how listings are distributed geographically, possibly mapping them or analyzing location-based trends.
Temporal Analysis: Examine trends over time, especially in reviews and availability.
Price Analysis: Analyze pricing strategies across different neighbourhoods and room types.
Host Analysis: Study host activity, such as multi-listing practices and their impact on ratings.
"""

#%%
#import listings.csv
import pandas as pd
listing = pd.read_csv('listings.csv')
print(listing.shape)
print(listing.info())
print(listing.head())
# %%
#Calculate percent of missing values in each column
count_missing_value = listing.isna().sum()
total_rows = len(listing)
percent_missing = (count_missing_value / total_rows) * 100
print(percent_missing)
#Remove column with more than 20% missing values
keep_col = percent_missing[percent_missing < 20].index
listing_filter = listing[keep_col]
col_removed = len(listing.columns) - len(listing_filter.columns)
print(f"The number of features removed: {col_removed}")
#%%
#Remove columns that contain "url" in their names
all_columns = listing_filter.columns.tolist()
columns_to_remove = [col for col in all_columns if 'url' in col]
listing_filter.drop(columns_to_remove, axis=1, inplace=True)
print(listing_filter.head())

# %%
"""
Change data types: 
host_since to datetime, 
host_response_rate to float,
host_is_superhost object to boolean,
host_has_profile_pic to boolean,
host_identity_verified to boolean,
price to float,
has_availability to boolean,
instant_bookable to boolean,
"""