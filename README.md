# EXNO-3-DS

## Name : STEFFI J
## Reg No : 212224220107

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

<img width="687" height="546" alt="image" src="https://github.com/user-attachments/assets/f60be0f2-1c87-4590-ae00-9933f852dd05" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="771" height="365" alt="image" src="https://github.com/user-attachments/assets/6cf5b752-0c9e-4a41-82fa-cf80e9c8945f" />

```
 df['bo2']=e1.fit_transform(df[["ord_2"]])
 df
```

<img width="546" height="517" alt="image" src="https://github.com/user-attachments/assets/dca725bc-8d7b-4bdf-9667-96febaa25c70" />

```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```

<img width="515" height="562" alt="image" src="https://github.com/user-attachments/assets/ebd4f2f3-90a5-49af-a282-763e0c775bd9" />

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="677" height="656" alt="image" src="https://github.com/user-attachments/assets/89c34f83-c07d-48fa-98cf-eb142ef19ba7" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="873" height="501" alt="image" src="https://github.com/user-attachments/assets/ee83c57b-ab60-493e-90f6-f06805d86936" />

```
pip install --upgrade category_encoders
```

<img width="1587" height="467" alt="image" src="https://github.com/user-attachments/assets/6ecebdd1-66aa-4360-9190-cfdc0f7516ea" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

<img width="672" height="545" alt="image" src="https://github.com/user-attachments/assets/56760ec9-adac-463f-916a-cdcfcdd7e38f" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

<img width="946" height="596" alt="image" src="https://github.com/user-attachments/assets/be76965c-d45f-4b88-8a0a-e632b1116651" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="898" height="516" alt="image" src="https://github.com/user-attachments/assets/baa4afc2-b15e-4b82-86ae-0f0893c69dc8" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="803" height="615" alt="image" src="https://github.com/user-attachments/assets/38b1afd5-27aa-4cef-9205-ef44efe05fe6" />

```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```

<img width="993" height="658" alt="image" src="https://github.com/user-attachments/assets/d8f036de-aeb2-4876-9e44-f0d6759ffb36" />

```
 df.skew()
```

<img width="502" height="303" alt="image" src="https://github.com/user-attachments/assets/b4ec8637-72c2-46e7-bb3d-246c34eb9b71" />

```
 np.log(df["Highly Positive Skew"])
```

<img width="522" height="610" alt="image" src="https://github.com/user-attachments/assets/2e3debd5-7762-4187-9bc1-038cf99a33f7" />

```
 np.reciprocal(df["Moderate Positive Skew"])
```

<img width="595" height="607" alt="image" src="https://github.com/user-attachments/assets/bf2e4c6e-a23a-4c7b-b3df-8b09e8d45e4a" />

```
 np.sqrt(df["Highly Positive Skew"])
```

<img width="475" height="612" alt="image" src="https://github.com/user-attachments/assets/9288033e-a1a2-4eea-aab9-3fa1836f780c" />

```
 np.square(df["Highly Positive Skew"])
```

<img width="530" height="616" alt="image" src="https://github.com/user-attachments/assets/9f1c41bd-63a9-4c5e-9b71-13b233492bf1" />

```
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```

<img width="1347" height="586" alt="image" src="https://github.com/user-attachments/assets/56329bcf-b71f-437a-a5d7-1441294f221e" />

```
df.skew()
```

<img width="491" height="347" alt="image" src="https://github.com/user-attachments/assets/b7414416-09dc-484b-bd1e-00d9a3025779" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1327" height="590" alt="image" src="https://github.com/user-attachments/assets/c9d3dda3-869c-4891-a328-0609efb1abb7" />


```
df.skew()
```

<img width="602" height="387" alt="image" src="https://github.com/user-attachments/assets/e9528b92-a156-43ba-92c2-403cd97e8ed8" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```

<img width="1523" height="636" alt="image" src="https://github.com/user-attachments/assets/d65e812c-e3bd-4fdb-9d42-ac48eb1db1f5" />

```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```

<img width="877" height="688" alt="image" src="https://github.com/user-attachments/assets/5d4aa75c-048f-41fe-ab08-f145c990ea31" />

```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()
```

<img width="1001" height="621" alt="image" src="https://github.com/user-attachments/assets/170b0ad2-9008-4314-8bad-4cd400178b55" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```
```
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```
```
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```

<img width="962" height="776" alt="image" src="https://github.com/user-attachments/assets/a45fa7bc-2a0c-4534-9601-32ed986bea40" />

```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```

<img width="861" height="647" alt="image" src="https://github.com/user-attachments/assets/1c7cf577-a14a-44b5-8d9f-9d7645174d1e" />

```
 dt=pd.read_csv("titanic_dataset.csv")
 dt
```

<img width="1492" height="591" alt="image" src="https://github.com/user-attachments/assets/6ed4976c-0530-4ea9-8460-0056325e8d9f" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```

<img width="952" height="686" alt="image" src="https://github.com/user-attachments/assets/086737d8-410f-4c8c-91f9-7ae7f30fb49a" />


```
 sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()S
```

<img width="892" height="616" alt="image" src="https://github.com/user-attachments/assets/808b0252-3573-407e-b3bd-2747e10f83cc" />

# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
