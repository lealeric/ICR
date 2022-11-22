import os
import pandas as pd

dfData = pd.read_excel(os.path.join(os.getcwd(), "NaturalDisasters.xlsx"))

print(dfData.head(20))
