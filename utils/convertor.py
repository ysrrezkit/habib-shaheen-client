import pandas as pd
import os 

df = pd.read_excel('sales_db.xlsx')
df.to_csv('sales_db.csv', index=False)
os.remove('sales_db.xlsx')

print("Excel file converted to CSV and original file removed.")