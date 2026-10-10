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
    frame_gray: np.ndarray, bg_gray: np.ndarray, blur_size: int = 7
) -> np.ndarray:
    """Applique un flou et retourne la différence absolue avec le fond."""
    f_blur = cv2.GaussianBlur(frame_gray, (blur_size, blur_size), 0)
    bg_blur = cv2.GaussianBlur(bg_gray, (blur_size, blur_size), 0)
    return cv2.absdiff(f_blur, bg_blur)


def thresh_frame(frame, lvl: int = 30):
    """_summary_

    Args:
        frame (_type_): _description_
        lvl (int, optional): _description_. Defaults to 40.
    """
    _, thresh = cv2.threshold(frame, lvl, 255, cv2.THRESH_BINARY)
    return thresh


def morpho_frame(frame, k_erode: int = 3, k_dilate: int = 11 ) -> np.ndarray:
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
    eroded = cv2.erode(frame, kernel_erode, iterations=1)
    dilated = cv2.dilate(eroded, kernel_dilate, iterations=2)
    return dilated