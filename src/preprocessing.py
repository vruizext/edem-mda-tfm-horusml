import cv2


def extract_diff_frames(video_path, max_frames=125, lag_ms=100, thresh=10):
    """
    Primero aplica una máscara para diferenciar las zonas activas/tejidos de las
    que no se mueven, y luego calcula la diferencia entre estas máscaras, para
    ver que zonas se mueven.
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
        # creamos una máscara para diferenciar zonas oscuras (cavidades) de las activas (tejido)
        frame = cv2.threshold(src=frame, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)[1]
        frames.append(frame.copy())
        frame_idx += 1

        if frame_idx < lag_frames:
            # necesitamos al menos lag frames para empezar
            continue

        # calculamos diferencia entre este frame y el frame que está lag_frames por detrás
        diff_frame = cv2.absdiff(frames[frame_idx - lag_frames], frame)
        diff_frames.append(diff_frame.copy())

    return diff_frames


def extract_diff_frames_inv(video_path, max_frames=125, lag_ms=100, thresh=10):
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


def avi2diff_frames(video_path, path_frames, start_frame=0, max_frames=100, lag_ms=100, thresh=15):
    dest_path = path_frames / video_path.stem
    dest_path.mkdir(parents=True, exist_ok=True)

    frames = extract_diff_frames(video_path, max_frames=(start_frame + max_frames), lag_ms=lag_ms, thresh=thresh)
    for i, frame in enumerate(frames[start_frame:(start_frame + max_frames)]):
        cv2.imwrite(str(dest_path / f'{i}.png'), frame)


def avi2diff_frames_inv(video_path, path_frames, start_frame=0, max_frames=100, lag_ms=100, thresh=15):
    dest_path = path_frames / video_path.stem
    dest_path.mkdir(parents=True, exist_ok=True)

    frames = extract_diff_frames_inv(video_path, max_frames=(start_frame + max_frames), lag_ms=lag_ms, thresh=thresh)
    for i, frame in enumerate(frames[start_frame:(start_frame + max_frames)]):
        cv2.imwrite(str(dest_path / f'{i}.png'), frame)
