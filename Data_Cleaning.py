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
#%%
listing_filter.columns.to_list()
new_df = pd.DataFrame(columns=listing_filter.columns, index=None)
pd.set_option('display.max_columns', None)
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
listing_filter["host_response_time"].value_counts()
# change host_response_time from categorical to numerical. Mapping  1, 2, 3, 4 - from fastest response to late response.
response_time_map = {"within an hour": 1,
                     "within a few hours": 2,
                     "within a day": 3,
                     "a few days or more": 4}

listing_filter["host_response_time"] = listing_filter["host_response_time"].map(response_time_map).astype("category")

# Convert host_response_rate, host_acceptance_rate columns from strings to numerical.

listing_filter["host_response_rate"] = listing_filter["host_response_rate"].apply(lambda x: int(x[:-1]) if isinstance(x, str) else x)
listing_filter["host_acceptance_rate"] = listing_filter["host_acceptance_rate"].apply(lambda x: int(x[:-1]) if isinstance(x, str) else x)
listing_filter["host_is_superhost"] = listing_filter["host_is_superhost"].apply(lambda x: 0 if x == 'f' else 1).astype("category")
listing_filter["host_identity_verified"] = listing_filter["host_identity_verified"].apply(lambda x: 0 if x == 'f' else 1).astype("category")
listing_filter["bathrooms"] = listing_filter["bathrooms_text"].replace(['Half-bath', "Private half-bath", "Shared half-bath"], 0.5).apply(lambda x: float(x.split()[0]) if isinstance(x, str) else x)
listing_filter["amenities"] = listing_filter["amenities"].apply(lambda x: len(eval(x)))
listing_filter["price"] = listing_filter["price"].apply(lambda x: float(''.join(x[1:].split(','))) if isinstance(x, str) else x)
listing_filter["instant_bookable"] = listing_filter["instant_bookable"].apply(lambda x: 0 if x == 'f' else 1).astype("category")
listing_filter["neighbourhood_cleansed"] = listing_filter["neighbourhood_cleansed"].astype("category")
listing_filter["room_type"] = listing_filter["room_type"].astype("category")
listing_filter["City"] = listing_filter["City"].astype("category")
listing_filter["State"] = listing_filter["State"].astype("category")
listing_filter.drop(["bedrooms", "bathrooms_text"], axis = 1, inplace = True)
listing_filter.info()

#%%
#Print names of numerical and categorical variables.
num_cols = listing_filter.select_dtypes(include=['float64', 'int64']).columns
cat_cols = listing_filter.select_dtypes(include=['category']).columns
print(num_cols, cat_cols, len(num_cols), len(cat_cols))
print(listing_filter[num_cols].describe())

# %%
