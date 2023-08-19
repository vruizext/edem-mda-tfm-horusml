from sklearn.metrics import mean_absolute_error, mean_squared_error
import gc
import time
from datetime import datetime
from fastai.vision.all import *

from .config import *


class ImageTuple(fastuple):
    # esta clase nos permite manejar un conjunto de imágenes como una tupla
    def show(self, ctx=None, **kwargs):
        n = len(self)
        img0, img1, img2 = self[0], self[n // 2], self[n - 1]
        if not isinstance(img1, Tensor):
            t0, t1, t2 = tensor(img0), tensor(img1), tensor(img2)
            t0, t1, t2 = t0.permute(2, 0, 1), t1.permute(2, 0, 1), t2.permute(2, 0, 1)
        else:
            t0, t1, t2 = img0, img1, img2
        return show_image(torch.cat([t0, t1, t2], dim=2), ctx=ctx, **kwargs)


class ImageTupleTfm(Transform):
    # esta clase obtiene las frames de un video y las devuelve dentro de una ImageTuple
    def __init__(self, start_frame=0, seq_len=20):
        store_attr()

    def encodes(self, path: Path):
        frames = path.ls_sorted()
        n_frames = len(frames)

        # nos quedamos con max_frames, empezando en start_frame
        if n_frames < (self.start_frame + self.seq_len):
            self.start_frame = max(0, n_frames - self.seq_len)

        # si no tenemos suficientes frames, replicamos
        if n_frames < self.seq_len:
            new_frames = self.seq_len - n_frames
            for n in range(0, new_frames):
                frames.append(frames[n % n_frames])

        s = slice(self.start_frame, self.start_frame + self.seq_len)
        return ImageTuple(tuple(PILImage.create(f) for f in frames[s]))


def get_regression_dls(df, splits, sf, sq, bs=8, shuffle_batch=False, batch_transforms=[]):
    # Esta función le dice al dataset donde buscar las X (frames de los videos)
    def get_x(row): return path_frames / row['FileName']

    # Y esta le dice donde buscar la Y, el target de cada X
    def get_y(row): return row['target']

    # Definimos dataset a partir del dataframe
    ds = Datasets(df,
                  tfms=[[get_x, ImageTupleTfm(start_frame=sf, seq_len=sq)],
                        [get_y, RegressionSetup]],
                  splits=splits(df))

    # Creamos dataloader a partir del dataset
    dls = ds.dataloaders(bs=bs, after_item=[Resize(112), ToTensor],
                         shuffle=shuffle_batch,
                         after_batch=[IntToFloatTensor(), *batch_transforms, Normalize.from_stats(*imagenet_stats)])

    return dls


def regression_metrics(learner, dls, valid_df, test_df):
    preds, targets = learner.get_preds()
    valid_mae = mean_absolute_error(preds.flatten(), targets)
    valid_mse = mean_squared_error(preds.flatten(), targets)
    # valid_df = train_df.iloc[valid_idx].copy()
    valid_df['error'] = np.abs(preds.flatten() - targets)

    # test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    test_data = dls.test_dl(test_df)
    preds, _ = learner.tta(dl=test_data)
    targets = test_df['target']
    test_mae = mean_absolute_error(preds.flatten(), targets)
    test_mse = mean_squared_error(preds.flatten(), targets)
    test_df['error'] = np.abs(preds.flatten() - targets)

    print("\nValid MAE por banda de FEVI")
    print(valid_df.groupby('FEVI10')['error'].mean())

    print("\nTest MAE por banda de FEVI")
    print(test_df.groupby('FEVI10')['error'].mean())

    return valid_mae, valid_mse, test_mae, test_mse


def regression_train(df, train_df, valid_idx, test_idx, model_class, model_arch, splitter, dls, epochs):
    # bs = train_params['bs']
    # sq = train_params['sq']
    # sf = train_params['sf']
    # epochs = train_params['epochs']
    # model_arch = train_params['model_arch']
    # shuffle_batch = train_params['shuffle_batch']

    t1 = time.time()

    # Creamos dataloader a partir del dataset
    # Definimos dataset a partir del dataframe
    # dls = get_regression_dls(df, sf, sq, bs, shuffle_batch, batch_transforms)

    model_obj = model_class(model_arch).cuda()
    learn = Learner(dls, model_obj, metrics=[mae, mse], splitter=splitter, loss_func=MSELossFlat()).to_fp16()

    lrs = learn.lr_find(suggest_funcs=(steep, valley, slide))
    lr = lrs.valley
    plt.show()

    print(f"LR: {lr}")

    cbs = [
        SaveModelCallback(monitor='mse', comp=np.less, fname="best-mse", every_epoch=False, reset_on_fit=False),
    ]

    learn.fit_one_cycle(epochs, lr, cbs=cbs)
    learn.recorder.plot_loss()
    plt.show()

    learn.load("best-mse")

    valid_df = train_df.iloc[valid_idx].copy()
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    valid_mae, valid_mse, test_mae, test_mse = regression_metrics(learn, dls, valid_df, test_df)

    iter_time = (time.time() - t1) / 60

    return valid_mae, valid_mse, test_mae, test_mse, iter_time, lr