import numpy as np
import pytest

from src.tracker import CentroidTracker


def test_tracker_register_and_match():
    """Test 1: Vérifie que le tracker enregistre un objet et le suit à la frame suivante."""
    tracker = CentroidTracker(max_distance=50.0)
    det_frame1 = [{"centroid": np.array([10, 10]), "bbox": (5, 5, 10, 10)}]
    objects_f1 = tracker.update(det_frame1)

    assert len(objects_f1) == 1
    assert 0 in objects_f1
    assert np.array_equal(objects_f1[0], np.array([10, 10]))

    det_frame2 = [{"centroid": np.array([12, 12]), "bbox": (7, 7, 10, 10)}]
    objects_f2 = tracker.update(det_frame2)
    assert len(objects_f2) == 1
    assert 0 in objects_f2
    assert np.array_equal(objects_f2[0], np.array([12, 12]))


def test_tracker_deregister_lost_objects():
    """Test 2: Vérifie que le tracker oublie un objet après X frames d'absence."""
    tracker = CentroidTracker(max_disappeared=2)
    tracker.update([{"centroid": np.array([10, 10]), "bbox": (5, 5, 10, 10)}])
    assert len(tracker.objects) == 1

    tracker.update([])
    assert len(tracker.objects) == 1

    tracker.update([])
    assert len(tracker.objects) == 0


def test_line_crossing_logic():
    """Test 3: Vérifie la logique de franchissement de ligne (comptage)."""
    tracker = CentroidTracker()
    ligne_virtuelle = (50.0, 0.0, 50.0, 100.0)
    obj_id = 1

    crossed = tracker.check_line_crossing(obj_id, np.array([40, 50]), ligne_virtuelle)
    assert crossed is False

    crossed = tracker.check_line_crossing(obj_id, np.array([60, 50]), ligne_virtuelle)
    assert crossed is True

    crossed = tracker.check_line_crossing(obj_id, np.array([70, 50]), ligne_virtuelle)
    assert crossed is False


def test_tracker_max_distance_exceeded():
    """Test 4: Un objet qui bouge trop vite ne doit pas être matché (nouvel ID assigné)."""
    tracker = CentroidTracker(max_distance=20.0)
    tracker.update([{"centroid": np.array([10, 10])}])
    objects_f2 = tracker.update([{"centroid": np.array([100, 100])}])

    assert 1 in objects_f2
    assert np.array_equal(objects_f2[1], np.array([100, 100]))
    assert tracker.disappeared[0] == 1  # L'objet 0 est considéré comme disparu


def test_tracker_reappearance_resets_disappeared():
    """Test 5: La réapparition d'un objet avant max_disappeared remet son compteur à 0."""
    tracker = CentroidTracker(max_disappeared=3)
    tracker.update([{"centroid": np.array([10, 10])}])
    assert tracker.disappeared[0] == 0
    tracker.update([])  # L'objet disparaît 1 frame
    assert tracker.disappeared[0] == 1
    tracker.update([{"centroid": np.array([12, 12])}])  # L'objet réapparaît
    assert tracker.disappeared[0] == 0


def test_tracker_multiple_objects_matching():
    """Test 6: Vérifie que plusieurs objets sont correctement associés simultanément."""
    tracker = CentroidTracker()
    f1 = [{"centroid": np.array([10, 10])}, {"centroid": np.array([50, 50])}]
    tracker.update(f1)
    f2 = [{"centroid": np.array([52, 52])}, {"centroid": np.array([11, 11])}]
    objects = tracker.update(f2)
    assert np.array_equal(objects[0], np.array([11, 11]))
    assert np.array_equal(objects[1], np.array([52, 52]))


def test_line_crossing_double_count_prevention():
    """Test 7: Un même objet ne doit pas être compté deux fois s'il re-franchit la ligne."""
    tracker = CentroidTracker()
    ligne = (50.0, 0.0, 50.0, 100.0)
    tracker.check_line_crossing(0, np.array([40, 50]), ligne)
    assert tracker.check_line_crossing(0, np.array([60, 50]), ligne) is True
    assert tracker.check_line_crossing(0, np.array([40, 50]), ligne) is False


def test_line_crossing_exactly_on_line():
    """Test 8: Cas limite où le centroïde atterrit pile sur la ligne."""
    tracker = CentroidTracker()
    ligne = (50.0, 0.0, 50.0, 100.0)
    tracker.check_line_crossing(0, np.array([40, 50]), ligne)
    crossed = tracker.check_line_crossing(0, np.array([50.0, 50]), ligne)

    assert (
        crossed is False
    )  # Le code ignore s1=0 ou s2=0, ce qui est le comportement attendu


def test_tracker_missing_centroid_key_raises_error():
    """Test 9: Vérifie qu'une détection mal formatée lève bien une exception claire."""
    tracker = CentroidTracker()
    bad_detection = [{"bbox": (0, 0, 10, 10)}]

    with pytest.raises(KeyError) as exc_info:
        tracker.update(bad_detection)

    assert "centroid" in str(exc_info.value)
