
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from fastai.vision import *
from fastai.metrics import error_rate
import pandas as pd
import numpy as np
import re


# In[3]:


bs = 64


# In[5]:


path = Path('data/Kaggle/Digit_Recognizer')
path.ls()


# In[6]:


df_train = pd.read_csv(path/'train.csv')
df_train['fn'] = df_train.index
df_train.head()


# In[7]:


class PixelImageItemList(ImageList):
    def open(self,fn):
        regex = re.compile(r'\d+')
        fn = re.findall(regex,fn)
        df = self.inner_df[self.inner_df.fn.values == int(fn[0])]
        df_fn = df[df.fn.values == int(fn[0])]
        img_pixel = df_fn.drop(labels=['label','fn'],axis=1).values
        img_pixel = img_pixel.reshape(28,28)
        img_pixel = np.stack((img_pixel,)*3,axis=-1)
        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))


# In[8]:


src = (PixelImageItemList.from_df(df_train,path=path,cols='fn')
      .split_by_rand_pct(valid_pct = 0.2, seed = 42)
      .label_from_df(cols='label'))


# In[10]:


tfms = get_transforms(max_rotate=10., max_zoom=1.05, do_flip=False)


# In[12]:


data = (src.transform(tfms=tfms)
       .databunch(num_workers=2,bs=bs)
       .normalize(imagenet_stats))


# In[13]:


data.show_batch(rows=3, figsize=(6,6))


# In[14]:


learn = cnn_learner(data,models.resnet50,metrics=accuracy)


# In[15]:


learn.lr_find()


# In[16]:


learn.recorder.plot()


# In[17]:


learn.save('Pre-learn1')


# In[18]:


learn.fit_one_cycle(5, max_lr=slice(1e-3))


# In[20]:


learn.save('50-stage1')


# In[21]:


learn.load('Pre-learn1')


# In[22]:


learn.fit_one_cycle(5, max_lr=slice(1e-2))


# In[23]:


learn.save('50-stage1')


# In[24]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(6,6))


# In[25]:


learn.unfreeze()


# In[26]:


learn.lr_find()


# In[27]:


learn.recorder.plot()


# In[28]:


learn.fit_one_cycle(7, max_lr=slice(3e-6,8e-5))


# In[29]:


learn.save('50-stage2')


# In[30]:


learn.fit_one_cycle(3, max_lr=slice(3e-6,8e-5))


# In[31]:


learn.fit_one_cycle(5, max_lr=slice(3e-6,8e-5))


# In[32]:


learn.load('50-stage2')


# In[33]:


learn.fit_one_cycle(6, max_lr=slice(3e-6,8e-5))


# In[36]:


learn.save('50-stage3')


# In[37]:


learn.fit_one_cycle(1, max_lr=slice(3e-6,8e-5))


# In[38]:


learn.save('50-stage4')


# In[39]:


learn.fit_one_cycle(2, max_lr=slice(3e-6,8e-5))


# In[40]:


learn.load('50-stage4')


# In[41]:


learn.fit_one_cycle(3, max_lr=slice(3e-6,8e-5))


# In[42]:


learn.load('50-stage4')


# In[43]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(6,6))


# In[45]:


interp.plot_confusion_matrix()


# In[47]:


learn.show_results()


# In[48]:


df_test = pd.read_csv(path/'test.csv')
df_test['label'] = 0
df_test['fn'] = df_test.index
df_test.head()


# In[49]:


learn.data.add_test(PixelImageItemList.from_df(df_test,path=path,cols='fn'))


# In[50]:


pred_test = learn.get_preds(ds_type=DatasetType.Test)


# In[51]:


test_result = torch.argmax(pred_test[0],dim=1)
result = test_result.numpy()


# In[55]:


test_result.shape


# In[57]:


final = pd.Series(result,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)
submission.to_csv('Kaggle-Digit_Recognizer-1.csv',index=False)


# In[58]:


submission.head()


# In[59]:


learn.load('50-stage3')


# In[60]:


learn.data.add_test(PixelImageItemList.from_df(df_test,path=path,cols='fn'))


# In[61]:


pred_test1 = learn.get_preds(ds_type=DatasetType.Test)


# In[62]:


test_result1 = torch.argmax(pred_test1[0],dim=1)
result1 = test_result1.numpy()


# In[63]:


final1 = pd.Series(result1,name='Label')
submission1 = pd.concat([pd.Series(range(1,28001),name='ImageId'),final1],axis=1)
submission1.to_csv('Kaggle-Digit_Recognizer-2.csv',index=False)

