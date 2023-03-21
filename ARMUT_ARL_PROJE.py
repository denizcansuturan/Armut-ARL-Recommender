#########################
# Business Problem
#########################

# Armut, the largest online service platform in Turkey, brings together service providers and those who want to receive
# services. It enables easy access to services such as cleaning, renovation, and transportation with just a few touches
# on a computer or smartphone. Using the dataset that includes users who have received services and the categories of
# these services, an association rule learning based product recommendation system is aimed to be created.

#########################
# Dataset
#########################
# The dataset consists of the services purchased by the customers and their categories.
# It includes the date and time information of each service received.

# -UserId

# -ServiceId: Services for categories (Ex :sofa cleaning service under the cleaning category)
# A ServiceId can be found under different categories and represents different services under different categories.
# The service with CategoryId 7 and ServiceId 4 is radiator cleaning,
# while the service with CategoryId 2 and ServiceId 4 is furniture assembly.

# -CategoryId: cleaning, transportation, renovation categories

# -CreateDate


#########################
# Task 1: Data Preparation
#########################
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# ensures that the output is on a single line
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: read armut_data.csv file

df = pd.read_csv("location")

# Step 2: Each ServiceID represents a different service for each CategoryID.
# Create a new variable representing the services by combining ServiceID and CategoryID with an underscore.

df.head()
df["Combined"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

# Step 3: The data set consists of the date and time when the service was received, without any basket definition
# (invoice, etc.) In order to apply Association Rule Learning, a basket (invoice, etc.) definition needs to be created.
# Here, the basket definition is the services each customer receives monthly. For example, customer with ID 7256 has a
# basket consisting of the services 9_4 and 46_4 they received in August 2017, and another basket consisting of the
# services 9_4 and 38_4 they received in October 2017. The baskets should be identified with a unique ID. To do this,
# first create a new date variable that only includes the year and month. Combine the UserID and the new date variable
# with "_" and assign it to a new variable named ID.

df['CreateDate'].dtype
df.head()
df['CreateDate'] = pd.to_datetime(df['CreateDate'])

df['New_Date'] = pd.DatetimeIndex(df['CreateDate']).year.astype(str) + "-" + pd.DatetimeIndex(
    df['CreateDate']).month.astype(str)
df['CartID'] = df['UserId'].astype(str) + "_" + df['New_Date']
#########################
# Task 2: Generate Association Rules.
#########################

# Step 1: Create a pivot table for the basket service as shown below.

# Combined         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# CartID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

df.isnull().sum()
df.describe().T


def create_invoice_product_df(dataframe, size=6, sizeneeded=False):
    if sizeneeded:
        return dataframe.groupby(['CartID', 'Combined']) \
                   ['CategoryId'].count() \
                   .unstack() \
                   .fillna(0) \
                   .applymap(lambda x: 1 if x > 0 else 0).iloc[0:size, 0:size]
    else:
        return dataframe.groupby(['CartID', 'Combined']) \
            ['CategoryId'].count() \
            .unstack() \
            .fillna(0) \
            .applymap(lambda x: 1 if x > 0 else 0)


inv_pro_df = create_invoice_product_df(df)

# everything looks normal, we can continue with generating Association Rules

# Step 2: Generate Association Rules.


frequent_itemsets = apriori(inv_pro_df.astype('bool'),
                            min_support=0.015,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False).head()

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules.head()


# Step 3: Using the arl_recommender function, recommend a service to a user who last received the 2_0 service.


def arl_recommender(rules_df, service, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, serv in enumerate(sorted_rules["antecedents"]):
        for j in list(serv):
            if j == service:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]


arl_recommender(rules, "2_0", 1)
arl_recommender(rules, "2_0", 2)
arl_recommender(rules, "2_0", 3)
