import cv2
import numpy as np


def draw_tracking_results(
    frame: np.ndarray, detections: list[dict], tracked_objects: dict
) -> np.ndarray:
    """
    Dessine les bounding boxes des détections et les IDs des objets suivis.
    """
    output = frame.copy()

    for det in detections:
        x, y, w, h = det["bbox"]
        cv2.rectangle(
            output, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
        )

    for obj_id, centroid in tracked_objects.items():
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

        # Afficher le texte de l'ID juste au-dessus du centroïde
        text = f"ID: {obj_id}"
        cv2.putText(
            output,
            text,
            (cx - 10, cy - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return output
