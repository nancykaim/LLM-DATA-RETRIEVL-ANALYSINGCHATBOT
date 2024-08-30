import pandas as  pd 
Superstore_df = pd.read_csv('Sample_Superstore.csv',encoding='latin1')
Customer_df = pd.read_csv('Customers.csv')
Customer_df = Customer_df.head(100)
Superstore_df = Superstore_df.head(100)
valmap={
    'Standard Class' : 0,
    'First Class' : 1,
    'Second Class' : 2,
}
Superstore_df['Ship Mode'] = Superstore_df['Ship Mode'].replace(valmap)

unique_superstore_df = Superstore_df.drop_duplicates(subset='Customer Name', keep='first').copy()
unique_superstore_df['CustomerID'] = range(1, len(unique_superstore_df) + 1)

data_to_add = Customer_df[['CustomerID','Annual Income','Spending Score','Family Size','Age']]

merge_df = pd.merge(unique_superstore_df,data_to_add,on='CustomerID')
merge_df.to_csv('testfile.csv')
