
# coding: utf-8

# In[1]:


from fastai import *
from fastai.tabular import *


# In[2]:


path = Path('data/Kaggle/House_Prices')


# In[4]:


path.ls()


# In[5]:


df_train = pd.read_csv(path/'Kaggle_House_Prices_train.csv')
df_test = pd.read_csv(path/'Kaggle_House_Prices_test.csv')


# In[6]:


# Fill empty test values with the mean
df_test.fillna(value = df_test.mean(), inplace=True)


# In[7]:


df_train.head()


# In[10]:


cat_names = df_train.select_dtypes(include=['object']).columns.tolist()
len(cat_names)


# In[11]:


cat_names1 = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
             'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
             'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
             'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
             'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition',
             'MSSubClass', 'OverallQual', 'OverallCond','BsmtFullBath',
              'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
              'Fireplaces','GarageCars','YrSold' , 'MoSold', 'LowQualFinSF' , 'PoolArea', 
             'YearBuilt', 'YearRemodAdd' , 'MiscVal', '3SsnPorch']
len(cat_names1)


# In[16]:


cont_names = df_train.select_dtypes(include=[np.number]).columns.tolist()
cont_names.remove('SalePrice')
cont_names.remove('Id')
len(cont_names)


# In[17]:


cont_names1 = [  'LotFrontage', 'LotArea',  
               'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
              'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea',
              'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']
len(cont_names1)


# In[18]:


dep_var = 'SalePrice'


# In[19]:


procs = [FillMissing, Categorify, Normalize]


# In[20]:


test = TabularList.from_df(df_test, cat_names=cat_names, cont_names=cont_names, procs=procs)


# In[22]:


data = (TabularList.from_df(df_train, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                        .split_by_rand_pct(valid_pct = 0.2, seed = 42)
                        .label_from_df(cols = dep_var, label_cls = FloatList, log = True)
                        .add_test(test)
                        .databunch())


# In[23]:


learn = tabular_learner(data, layers=[200,100], metrics=rmse)


# In[24]:


learn.lr_find()


# In[25]:


learn.recorder.plot()


# In[26]:


learn.save('Pre-Learn')


# In[27]:


learn.fit_one_cycle(15, max_lr =5e-01)


# In[28]:


learn.load('Pre-Learn')


# In[29]:


learn.fit_one_cycle(15, max_lr =1e-01)


# In[30]:


learn.save('200,100 - 15 - 1e-1')


# In[31]:


learn.fit_one_cycle(5, max_lr =1e-01)


# In[32]:


learn.fit_one_cycle(15, max_lr =1e-01)


# In[33]:


learn.save('200,100 - 15,5,15 - 1e-1')


# In[35]:


test_id = df_test['Id']
preds, targets = learn.get_preds(DatasetType.Test)
labels = [np.exp(p[0].data.item()) for p in preds]

submission = pd.DataFrame({'Id': test_id, 'SalePrice': labels})
submission.to_csv('Kaggle_House_Price-1.csv', index=False)
submission.head()

