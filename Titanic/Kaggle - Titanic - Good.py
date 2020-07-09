# In[1]:
from fastai.tabular import *

# In[2]:
path = Path('data/Kaggle/Titanic')

# In[4]:
df_train = pd.read_csv(path/'train.csv')
df_test = pd.read_csv(path/'test.csv')

# In[5]:
df_test["Fare"] = df_test["Fare"].fillna(value =0)

# In[6]:
dep_var = 'Survived'
cat_names = ['Pclass','Sex', 'Ticket', 'Cabin', 'Embarked']
cont_names = ['Age', 'SibSp','Parch', 'Fare']
procs = [FillMissing, Categorify, Normalize]

# In[7]:
df_train.head()

# In[8]:
test = TabularList.from_df(df_test, cat_names=cat_names, cont_names=cont_names, procs=procs)

# In[9]:
data = (TabularList.from_df(df_train, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                            .split_by_idx(list(range(0,200)))
                            .label_from_df(cols=dep_var)
                            .add_test(test, label=0)
                            .databunch())

# In[10]:
data.show_batch(rows=10)

# In[11]:
learn = tabular_learner(data, layers=[500,200,15], metrics=accuracy)

# In[12]:
learn.lr_find()

# In[13]:
learn.recorder.plot()

# In[14]:
learn.fit_one_cycle(5, max_lr=slice(1e-3))

# In[15]:
learn.fit_one_cycle(5, 1e-02)

# In[16]:
learn.save('stage-1')

# In[17]:
learn.unfreeze()

# In[18]:
learn.lr_find()

# In[19]:
learn.recorder.plot()

# In[20]:
learn.fit_one_cycle(5, 3e-3)

# In[ ]:
learn.load('stage-1')

# In[25]:
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)

# In[42]:
test_id=df_test['PassengerId']

# In[43]:
submission = pd.DataFrame({'PassengerId': test_id, 'Survived': labels})
submission.to_csv('submission.csv', index=False)
submission.head()
