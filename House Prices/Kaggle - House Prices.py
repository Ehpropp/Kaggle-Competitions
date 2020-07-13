from fastai import *
from fastai.tabular import *

path = Path('data/Kaggle/House_Prices')
path.ls()

df_train = pd.read_csv(path/'Kaggle_House_Prices_train.csv')
df_test = pd.read_csv(path/'Kaggle_House_Prices_test.csv')

# Fill empty test values with the mean for that column
df_test.fillna(value = df_test.mean(), inplace=True)

df_train.head()

dep_var = 'SalePrice'
cat_names = df_train.select_dtypes(include=['object']).columns.tolist()
len(cat_names)
cont_names = df_train.select_dtypes(include=[np.number]).columns.tolist()
cont_names.remove('SalePrice') # Remove SalePrice and Id from cont_names as they aren't features
cont_names.remove('Id')
len(cont_names)
procs = [FillMissing, Categorify, Normalize]

test = TabularList.from_df(df_test, cat_names=cat_names, cont_names=cont_names, procs=procs)

data = (TabularList.from_df(df_train, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                        .split_by_rand_pct(valid_pct = 0.2, seed = 42)
                        .label_from_df(cols = dep_var, label_cls = FloatList, log = True)
                        .add_test(test)
                        .databunch())

learn = tabular_learner(data, layers=[200,100], metrics=rmse)

learn.lr_find()
learn.recorder.plot()

learn.save('Pre-Learn')

learn.fit_one_cycle(15, max_lr =1e-01)
learn.save('200,100 - 15 - 1e-1')

learn.fit_one_cycle(5, max_lr =1e-01)

learn.fit_one_cycle(15, max_lr =1e-01)
learn.save('200,100 - 15,5,15 - 1e-1')

test_id = df_test['Id']
preds, targets = learn.get_preds(DatasetType.Test)
labels = [np.exp(p[0].data.item()) for p in preds]

submission = pd.DataFrame({'Id': test_id, 'SalePrice': labels})
submission.to_csv('Kaggle_House_Price-1.csv', index=False)
submission.head()
