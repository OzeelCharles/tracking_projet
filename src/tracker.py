import numpy as np
from utils.geometry import calculate_distance, side


class CentroidTracker:
    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 30):
        self.next_object_id = 0
        self.objects = {}  # Dict {id: centroid} actuel
        self.previous_centers = {}  # Mémoire pour le franchissement de ligne
        self.disappeared = {}  # On gère le cas de disparition de centroïdes
        self.counted_ids = set()  # IDs déjà comptés pour ne pas les compter en double
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def _deregister(self, obj_id: int):
        """Supprime proprement un objet de la mémoire."""
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        if obj_id in self.previous_centers:
            del self.previous_centers[obj_id]

    def update(self, detections: list[dict]) -> dict:
        """Associe les nouvelles détections aux IDs existants."""
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] >= self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects

        # Si aucun objet n'est suivi, on enregistre tout comme nouveau
        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
            return self.objects

        # MATCHING LOGIC (Association spatiale)
        used_objects = set()
        used_detections = set()

        for det_idx, det in enumerate(detections):
            best_id = None
            best_dist = float("inf")

            for obj_id, old_centroid in self.objects.items():
                if obj_id in used_objects:
                    continue

                dist = calculate_distance(det["centroid"], old_centroid)

                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_id = obj_id

            # Si on a trouvé une correspondance valide
            if best_id is not None:
                det["id"] = best_id
                self.objects[best_id] = det["centroid"]
                self.disappeared[best_id] = 0
                used_objects.add(best_id)
                used_detections.add(det_idx)

        # Ici on gère le cas des objets qui disparaissent
        for obj_id in list(self.objects.keys()):
            if obj_id not in used_objects:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] >= self.max_disappeared:
                    self._deregister(obj_id)

        # Enregistrement des détections restantes comme NOUVEAUX objets
        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections:
                self._register(det)

        return self.objects

    def _register(self, det: dict):
        """Enregistre un nouvel objet et lui assigne un ID."""
        det["id"] = self.next_object_id
        self.disappeared[det["id"]] = 0
        self.objects[self.next_object_id] = det["centroid"]
        self.next_object_id += 1

    def check_line_crossing(
        self,
        obj_id: int,
        current_centroid: np.ndarray,
        line: tuple[float, float, float, float],
    ) -> bool:
        """Vérifie si un objet vient de franchir la ligne de comptage."""
        x1, y1, x2, y2 = line

        # Première apparition de l'objet pour la vérification de ligne
        if obj_id not in self.previous_centers:
            self.previous_centers[obj_id] = current_centroid
            return False

        prev = self.previous_centers[obj_id]

        s1 = side(prev, x1, y1, x2, y2)
        s2 = side(current_centroid, x1, y1, x2, y2)

        # Mise à jour de l'historique
        self.previous_centers[obj_id] = current_centroid

        if s1 == 0 or s2 == 0:
            return False

        if s1 * s2 < 0 and obj_id not in self.counted_ids:
            self.counted_ids.add(obj_id)
            return True

        return False
