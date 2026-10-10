# Paramètres de Détection
MIN_AREA_CONTOUR = 2500
BLUR_SIZE = 7
THRESH_LEVEL = 25

# Paramètres de Tracking
MAX_DISTANCE = 40
MAX_DISAPPEARED = 30

# Paramètres Métier (Comptage)
# COUNT_LINE = (400, 600, 800, 600) # Road mp4

COUNT_LINE = (1574, 864, 1731, 824)  # Pour le cas de la ville allemande

# Paramètres d'Initialisation
INIT_FRAMES_COUNT = 30

# Paramètre d'affichage final

DISPLAY = True

# Learning rate du background glissant
LR = 0.2

# Region of Interest (ROI)
ROI_VERTICES = [(1811, 1047), (1091, 558), (1420, 457), (1800, 837)]