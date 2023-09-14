import cv2
import albumentations as A
from fastai.vision.all import *
import pandas as pd

from .config import *


# Balanced Contrast Enhancement Technique (BCET)
def bcet(image, ex=110, low=0, high=255):
    s = np.mean(np.power(image, 2)) # mean squared
    e = np.mean(image)
    l = np.min(image)
    h = np.max(image)

    L = low # output minimum
    H = high # output maximum
    E = ex # # output mean

    # Find b
    b_nom = ((h**2)*(E-L))-(s*(H-L))+((l**2)*(H-E))
    b_den = 2*((h*(E-L))-(e*(H-L))+(l*(H-E)))

    b = b_nom/b_den

    # Find a
    a1 = H-L
    a2 = h-l
    a3 = h+l-(2*b)

    a = a1/(a2*a3)

    # Find c
    c = L-(a*(l-b)**2)

    # Apply BCET
    new_image = a*((image - b)**2) + c
    return new_image


def extract_frames(video_path, max_frames=125):
    """
    Extrae las frames de un vídeo de forma secuencial
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error opening video file")

    frames = []
    frame_idx = -1

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # extraer frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.copy())
        frame_idx += 1

    return frames


def avi2frames(video_path, max_frames=50, use_bcet=False):
    """
    Extrae las frames de un video y las guarda como imágenes
    """

    dest_path = path_frames/video_path.stem

    # creamos el directorio para guardar las frames
    dest_path.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(video_path, max_frames)
    for i, frame in frames:
        if use_bcet == True:  # aplicamos ajuste de contraste
            frame = bcet(np.array(frame), low=0, ex=50, high=255)
        else:
            frame = np.array(frame)
        # guardamos cada frame, numerada, dentro del directorio del video
        cv2.imwrite(str(dest_path / f'{i}.png'), frame)


def extract_diff_frames(video_path, max_frames=125, lag_ms=100, thresh=10):
    """
    Primero calcula la diferencia absoluta entre frames y luego aplica una máscara
    para diferenciar las zonas que no se han movido de las que lo han hecho.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error opening video file")

    # calculamos el número de frames necesario para obtener lag_ms
    fps = cap.get(cv2.CAP_PROP_FPS)
    lag_frames = int(lag_ms * fps // 1000)

    frames = []
    diff_frames = []
    frame_idx = -1

    while frame_idx < (max_frames + lag_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # extraer frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.copy())
        frame_idx += 1

        if frame_idx < lag_frames:
            # necesitamos al menos lag frames para empezar
            continue

        # calculamos diferencia entre este frame y el frame que está lag_frames por detrás
        diff_frame = cv2.absdiff(frames[frame_idx - lag_frames], frame)
        # creamos una máscara para diferenciar zonas oscuras (cavidades) de las activas (tejido)
        diff_frame = cv2.threshold(src=diff_frame, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)[1]
        diff_frames.append(diff_frame.copy())

    return diff_frames


def avi2diff_frames(video_path, start_frame=0, max_frames=100, lag_ms=100, thresh=15):
    dest_path = path_frames/video_path.stem
    dest_path.mkdir(parents=True, exist_ok=True)

    frames = extract_diff_frames(video_path, max_frames=(start_frame + max_frames), lag_ms=lag_ms, thresh=thresh)
    for i, frame in enumerate(frames[start_frame:(start_frame + max_frames)]):
        cv2.imwrite(str(dest_path / f'{i}.png'), frame)



def oversample_video(video_id, new_video_id, transform):
    # creamos directorio para el nuevo video
    video_path = path_frames / new_video_id
    if not video_path.exists():
        video_path.mkdir()

    for img_path in (path_frames / video_id).ls_sorted():
        # leemos un frame
        new_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        # aplicamos las transformaciones
        new_img = transform(image=new_img)['image']
        # guardamos imagen
        new_img_path = (video_path / img_path.stem).with_suffix('.png')
        if not cv2.imwrite(str(new_img_path), new_img):
            print(f"error writing file {new_img_path}")
            return False

    return True


def oversampling(df, num_img, counts, transform):
    new_df = pd.DataFrame()
    tot_img = 0

    for k, img_count in counts.items():
        if img_count >= num_img:
            continue
        tmp_df = df[df['FEVI10'] == k].reset_index(drop=True)
        # cuantos nuevos videos vamos a generar para cada subclase
        num_new = num_img - img_count
        idx = 0

        print(f"\nGenerating {num_new} videos for class {k}\n")

        while num_new > 0:
            row = tmp_df.iloc[idx]
            video_id = row['FileName']
            new_video_id = f"{video_id}_{new_df.shape[0]}"
            oversample_video(video_id, new_video_id, transform)

            new_df = new_df.append({ 'FileName': new_video_id, 'target': row['target'], 'FEVI10': k }, ignore_index=True)
            # indice del siguiente video
            idx = (idx + 1) % img_count
            # actualizamos numero de videos restantes
            num_new -= 1

    return new_df

# esta clase nos permite manejar un conjunto de imágenes como una tupla
class ImageTuple(fastuple):
    "A tuple of PILImages"
    def show(self, ctx=None, **kwargs):
        n = len(self)
        img0, img1, img2= self[0], self[n//2], self[n-1]
        if not isinstance(img1, Tensor):
            t0, t1,t2 = tensor(img0), tensor(img1),tensor(img2)
            t0, t1,t2 = t0.permute(2,0,1), t1.permute(2,0,1),t2.permute(2,0,1)
        else: t0, t1,t2 = img0, img1,img2
        return show_image(torch.cat([t0,t1,t2], dim=2), ctx=ctx, **kwargs)


# esta clase obtiene las frames de un video y las devuelve en forma de tupla
class ImageTupleTfm(Transform):
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

