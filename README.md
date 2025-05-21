## EXNO-3-DS
# Name:Sree Niveditaa Saravanan
# Reg.no:212223230213

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
![image](https://github.com/user-attachments/assets/5ec31a5a-c83e-405b-816f-8f603db28de1)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/3b07cb57-601c-413d-b932-acabe1a148ea)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/56b5b965-539d-489e-b1f5-1d9e3aa53237)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/c895d1fd-cb06-480e-bb98-4d2063c54526)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/c2277e6a-c543-4d22-8169-193b863adf0d)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/8eb08bce-55be-4e3f-8964-12c1eaf96127)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/7dd3f373-add4-4b46-8a39-ce314686506c)

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/7af14088-d6de-4603-853b-9e3455797876)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/2aaf0d41-b4e8-48e2-a2a0-d716b44a5563)

```
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/57392653-6625-4773-adc4-df18e1133923)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/e6b2444f-a3cd-4321-9d65-9c637243b20d)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7bae1473-e3ba-480e-9437-ffb0ea4db249)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/35d6d278-d941-4762-9ce9-612f189f66b0)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f7b4ac03-ac0d-4c08-8c9b-25eb1691c354)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/13287e2f-81fd-460b-92a2-531dc72a473e)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/a3cef940-8b2b-425e-8b2b-500e056cc879)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/d5504a6d-6963-4edc-bcf6-3f5f26dea404)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/da2b6581-0cec-4232-a39e-37383fd32541)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/93f9be16-0805-4d9e-ab95-333f3c587625)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/78be949d-033f-4619-b7d9-fbec5595ec36)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/35592df2-e252-45bc-9535-19ba2681b567)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9c8a92c6-e1bd-47e1-90a8-2cb944ecf316)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c47720df-1c12-4cf0-8252-08fce5f5ccb5)

```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![image](https://github.com/user-attachments/assets/4aa92f3e-cb5a-4a48-a139-fcc8f2e4120d)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/user-attachments/assets/3db6ef8d-c9b0-46db-802d-886e11e6c567)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/249280b8-aca1-45f6-a40e-da21dbc2d264)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
