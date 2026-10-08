# src/main.py
import cv2
import os
import glob
from dotenv import load_dotenv
import numpy as np
from src.config import *
from src.detector import MotionDetector
from src.tracker import CentroidTracker
from utils.image_processing import get_mean_background
from utils.visualization import draw_tracking_results


def get_video_source() -> str | int:
    """
    Renvoie un fichier .mp4 dans le dossier data/
    Sinon renvoie l'url dans le fichier env.txt
    """
    local_videos = glob.glob("data/*.mp4") + glob.glob("data/*.avi")
    if local_videos:
        print(f"[INFO] Vidéo locale trouvée : {local_videos[0]}")
        return local_videos[0]
    load_dotenv(os.path.join("venv.txt"))
    url = os.getenv("URL")
    if url:
        return url
    raise FileNotFoundError("Aucune vidéo locale trouvée et env.txt vide ou absent.")


def init_background(cap: cv2.VideoCapture, num_frames: int) -> np.ndarray:
    """
    Capture les N premières frames pour calculer l'image moyenne de fond
    """
    frames = []
    print(f"Capture de {num_frames} frames pour initialiser le fond...")
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return get_mean_background(frames)


def main():
    try:
        source = get_video_source()
    except FileNotFoundError as e:
        print(e)
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Impossible d'ouvrir le flux vidéo.")
        return

    bg_frame = init_background(cap, INIT_FRAMES_COUNT)
    if bg_frame is None:
        print("Impossible de générer le fond.")
        return

    detector = MotionDetector(background_frame=bg_frame, min_area=MIN_AREA_CONTOUR)
    tracker = CentroidTracker(
        max_distance=MAX_DISTANCE, max_disappeared=MAX_DISAPPEARED
    )

    print("Démarrage du tracking (Appuyez sur 'q' pour quitter)")
    total_crossings = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin du flux vidéo")
            break

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        for obj_id, centroid in tracked_objects.items():
            crossed = tracker.check_line_crossing(obj_id, centroid, COUNT_LINE)
            if crossed:
                total_crossings += 1
                print(
                    f"[EVENT] L'objet {obj_id} a franchi la ligne ! Total : {total_crossings}"
                )

        if DISPLAY:
            output_frame = draw_tracking_results(frame, detections, tracked_objects)
            cv2.line(
                output_frame,
                (int(COUNT_LINE[0]), int(COUNT_LINE[1])),
                (int(COUNT_LINE[2]), int(COUNT_LINE[3])),
                (0, 255, 255),
                2,
            )
        else:
            output_frame = frame.copy()

        cv2.putText(
            output_frame,
            f"Comptage: {total_crossings}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Tracking CitySkyline", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
