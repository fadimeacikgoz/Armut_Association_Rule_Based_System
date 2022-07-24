############################################
# Armut Association Rule Based Recommender System
############################################

#############################################
# Business Problem
############################################
# Armut, Turkey's largest online service platform, brings together service providers and those who want to receive service.
# It provides easy access to services such as cleaning, modification and transportation with a few touches on your computer
# or smart phone. It is desired to create a product recommendation system with Association Rule Learning by using the data set
# containing the service users and the services and categories these users have received.




############################################
#  Dataset Story
############################################
# The data set consists of the services customers receive and the categories of these services.
# It contains the date and time information of each service received.
#

############################################
# Variables
############################################

# *UserId : Customer ID
# *ServiceId : They are anonymized services belonging to each category. (Example: Upholstery washing service under cleaning) A ServiceId can be found under different categories and refers to different services under different categories. (Example: Furniture assembly with CategoryId 2 ServiceId 4 during service core cleaning with CategoryId 7 ServiceId 4)
# *CategoryId :They are anonymized categories. (Example: Cleaning, transportation, renovation category)
# *CreateDate :The date the service was purchased


# Task 1 :Data Preparation

# !pip install mlxtend
import pandas as pd


# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Step1 :Read the armut_data.csv dataset.
df_ = pd.read_csv("Recommender_Systems/ödev/armut_data.csv")
df = df_.copy()
df.head()
df.isnull().sum()
df.describe().T
df.info()

# Step 2: Outlier suppression
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    replace_with_thresholds(dataframe, "ServiceId")
    replace_with_thresholds(dataframe, "CategoryId")
    replace_with_thresholds(dataframe, "UserId")

    return dataframe

df = retail_data_prep(df)
df.describe().T

df["ServiceId"] = df["ServiceId"].astype("int64")
df["CategoryId"] = df["CategoryId"].astype("int64")

# Step 2: ServiceID represents a different service for each Category ID.
# Combine ServiceID and CategoryID with "_" to create a new representation
# to represent these services

df.info()
df.head()
df.values

df["ServiceId"] = df["ServiceId"].astype("int64")
df["CategoryId"] = df["CategoryId"].astype("int64")
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")

df["ServiceId"] = df["ServiceId"].astype("str")
df["CategoryId"] = df["CategoryId"].astype("str")
df["hizmet"] = ["_".join(col) for col in (df[["ServiceId", "CategoryId"]].values)]
# df["hizmet"] = [row[1] + "_" + row[2] for row in df.values]


# Step 3: The data set consists of the date and time the services were received,
# there is no basket definition (invoice etc.). In order to apply Association Rule Learning,
# a basket (invoice, etc.) definition must be created. Here, the definition of basket is the services
# that each customer receives monthly. For example; A basket of 4_5, 48_5, 6_7, 47_7 services received by the
# customer with id 25446 in the 8th month of 2017; 17_5, 147 services received in the 9th month of 2017 represent
# another basket. Baskets must be identified with a unique ID. To do this, first create a new date containing only the
# year and month. Create the variable. Combine UserID and the newly created date variable with "" to a new variable named ID.


####
df["UserId"]= df["UserId"].astype("str")
df["NEW_DATE"] = df["NEW_DATE"].astype("str")
df["SepetID"]= ["_".join(col) for col in (df[["UserId", "NEW_DATE"]].values)]


#Task 2: Create an Association Rule Based Recommender System and make a recommendation

df.head()

#
df_t = df.groupby(['SepetID', 'hizmet'])["CategoryId"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
#     df_t= common_movies.pivot_table(index=["SepetID"], columns=["hizmet"], values="CategoryId")
df_t.shape

############################################
# Step 1:Create Association Rules
############################################

frequent_itemsets = apriori(df_t,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.01) & (rules["confidence"]>0.1) & (rules["lift"]>1)]. \
sort_values("confidence", ascending=False)

#Step 3: Using the arl_recommender function, recommend a service to a user
# who has received the 2_0 service in the last 1 month.
def arl_recommender(rules, product_id, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, "2_0", 1)
arl_recommender(rules, "2_0", 2)
arl_recommender(rules, "2_0", 3)



