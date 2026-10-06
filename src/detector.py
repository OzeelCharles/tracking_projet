import cv2
import numpy as np
from utils.image_processing import (
    to_grayscale,
    get_background_difference,
    morpho_frame,
    thresh_frame,
)


class MotionDetector:
    def __init__(self, background_frame: np.ndarray, min_area: int = 1100):
        self.bg_gray = to_grayscale(background_frame)
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Retourne une liste de dictionnaires contenant bbox et centroid des objets détectés.
        """
        frame_gray = to_grayscale(frame)
        diff = get_background_difference(frame_gray, self.bg_gray)
        thresh = thresh_frame(diff)
        morph = morpho_frame(thresh)

        # On utilise connectedComponentsWithStats (plus moderne que findContours pour avoir les centroids direct)
        nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(morph)

        detections = []
        for i in range(1, nlabels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                detections.append(
                    {
                        "centroid": centroids[i],
                        "bbox": (
                            stats[i, cv2.CC_STAT_LEFT],
                            stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT],
                        ),
                    }
                )
        return detections
