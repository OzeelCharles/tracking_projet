import cv2
import numpy as np


def get_mean_background(frames: list[np.ndarray]) -> np.ndarray:
    """Calcule l'image moyenne (background) à partir d'une liste de frames."""
    if not frames:
        raise ValueError("La liste de frames est vide.")
    frames_array = np.array(frames, dtype=np.double)
    mu = frames_array.mean(axis=0)
    return np.uint8(mu)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def get_background_difference(
    frame_gray: np.ndarray, bg_gray: np.ndarray, blur_size: int = 15
) -> np.ndarray:
    """Applique un flou et retourne la différence absolue avec le fond."""
    f_blur = cv2.blur(frame_gray, (blur_size, blur_size))
    bg_blur = cv2.blur(bg_gray, (blur_size, blur_size))
    return cv2.absdiff(f_blur, bg_blur)


def thresh_frame(frame, lvl=25):
    """_summary_

    Args:
        frame (_type_): _description_
        lvl (int, optional): _description_. Defaults to 40.
    """
    thresh = np.zeros_like(frame, dtype=np.uint8)
    thresh[frame > lvl] = 255
    return thresh


def morpho_frame(frame, k1=(40, 40), k2=(20, 20)):
    kernel1 = np.ones(k1, dtype=np.uint8)
    dilated = cv2.dilate(frame, kernel1)
    kernel2 = np.ones(k2, dtype=np.uint8)
    return cv2.erode(dilated, kernel2)
