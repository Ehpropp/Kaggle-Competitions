get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *
from fastai.metrics import error_rate
import pandas as pd
import numpy as np
import re

bs = 64

path = Path('data/Kaggle/Digit_Recognizer')
path.ls()

df_train = pd.read_csv(path/'train.csv')
df_train['fn'] = df_train.index
df_train.head()

# This class was developed in the follwing Kaggle notebook:
# https://www.kaggle.com/heye0507/fastai-1-0-with-customized-itemlist
# It was developed to have the fastai library read in the pixel data from the csv correctly.
# Check out the Kaggle notebook mentioned above for a detailed description
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

# Split train.csv into 80/20 train/valid sets
# Create a seed to keep any future valid sets the same
src = (PixelImageItemList.from_df(df_train,path=path,cols='fn')
      .split_by_rand_pct(valid_pct = 0.2, seed = 42)
      .label_from_df(cols='label'))

# Create transforms of the images
# max_rotate as numbers can be written on angles
# max_zoom to make images larger and smaller
# do_flip = False as numbers are not written backwards
tfms = get_transforms(max_rotate=10., max_zoom=1.05, do_flip=False)

data = (src.transform(tfms=tfms)
       .databunch(num_workers=2,bs=bs)
       .normalize(imagenet_stats))

# View the images in the dataset
data.show_batch(rows=3, figsize=(6,6))

# Create a learner with ResNet50
learn = cnn_learner(data,models.resnet50,metrics=accuracy)

# Find and plot learning rate data
learn.lr_find()
learn.recorder.plot()
learn.save('Pre-learn1')

learn.fit_one_cycle(5, max_lr=slice(1e-2))

learn.save('50-stage1')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(6,6))

# Unfreeze and train the entire model (not just the last layers which were solely trained before)
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(7, max_lr=slice(3e-6,8e-5))
learn.save('50-stage2')

learn.fit_one_cycle(3, max_lr=slice(3e-6,8e-5))
learn.fit_one_cycle(5, max_lr=slice(3e-6,8e-5))

learn.load('50-stage2')

learn.fit_one_cycle(6, max_lr=slice(3e-6,8e-5))

learn.save('50-stage3')

learn.fit_one_cycle(1, max_lr=slice(3e-6,8e-5))

learn.save('50-stage4')

learn.fit_one_cycle(2, max_lr=slice(3e-6,8e-5))

learn.load('50-stage4')

learn.fit_one_cycle(3, max_lr=slice(3e-6,8e-5))

learn.load('50-stage4')

# Look at the top mistakes (they seemed reasonable when looking at the images)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(6,6))
interp.plot_confusion_matrix()
learn.show_results()

# 50-stage3 resulted in my hisgest score on Kaggle
# It had the same accuracy as 50-stage4 but lower train and valid losses
learn.load('50-stage3')

# Load test data
df_test = pd.read_csv(path/'test.csv')
df_test['label'] = 0
df_test['fn'] = df_test.index
df_test.head()

learn.data.add_test(PixelImageItemList.from_df(df_test,path=path,cols='fn'))

# Get test predictions
pred_test = learn.get_preds(ds_type=DatasetType.Test)

test_result = torch.argmax(pred_test[0],dim=1)
result = test_result.numpy()
test_result.shape

# Create a csv file
final = pd.Series(result,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)
submission.to_csv('Kaggle-Digit_Recognizer-2.csv',index=False)
submission.head()
