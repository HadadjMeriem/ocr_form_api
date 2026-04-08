import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

TEMPLATE_PATH = str(BASE_DIR / "doc2.png")
DEBUG_DIR = str(BASE_DIR / "debug_outputs")

DATE_OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789./-'

CANONICAL_W = 1448
CANONICAL_H = 2048

EXCLUSIVE_MIN_CONF = 0.35
EXCLUSIVE_MARGIN = 0.08

EXPECTED_CHECKBOX_ORDER = [
    "sexe_masculin", "sexe_feminin",
    "age_0_4", "age_5_14", "age_15_24", "age_25_44", "age_45_64", "age_65p",
    "varicelle", "syndromes_grippaux",
    "varicelle_debut_brutal", "varicelle_fievre_moderee", "grippe_fievre_39",
    "varicelle_eruption", "grippe_debut_brutal", "varicelle_prurit",
    "grippe_myalgies", "varicelle_duree", "grippe_signes_respiratoires",
    "varicelle_dessiccation", "diarrhee_aigue", "ira", "ira_apparition_brutale",
    "diarrhee_3_selles", "ira_fievre", "diarrhee_moins_14j",
    "diarrhee_motif_consultation", "ira_signes_respiratoires",
]

# =========================================================
# CACHE GLOBAL
# =========================================================

_TEMPLATE = None
_TEMPLATE_FEATURES = None
_TEMPLATE_BOXES = None
_ORB = None

# =========================================================
# OUTILS
# =========================================================

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def resize_keep_ratio(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is not None:
        scale = width / w
        return cv2.resize(img, (width, int(h * scale)))
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))

def preprocess_for_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return sharp

def normalize_local_illumination(gray):
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
    return cv2.divide(gray, bg, scale=255)

# =========================================================
# TEMPLATE INIT
# =========================================================

def init_template_cache():
    global _TEMPLATE, _TEMPLATE_FEATURES, _TEMPLATE_BOXES, _ORB

    if _TEMPLATE is not None:
        return

    template = cv2.imread(str(TEMPLATE_PATH))
    if template is None:
        raise ValueError(f"Impossible de charger la template : {TEMPLATE_PATH}")

    template = cv2.resize(template, (CANONICAL_W, CANONICAL_H))

    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=6,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=12
    )

    tpl_feat = preprocess_for_features(template)
    kp, des = orb.detectAndCompute(tpl_feat, None)
    if des is None:
        raise ValueError("Impossible de calculer les descripteurs ORB de la template.")

    _TEMPLATE = template
    _TEMPLATE_FEATURES = (kp, des)
    _TEMPLATE_BOXES = detect_checkboxes_on_template(template)
    _ORB = orb

# =========================================================
# ALIGNEMENT
# =========================================================

def align_to_template_fast(image):
    init_template_cache()

    tpl = _TEMPLATE
    kp2, des2 = _TEMPLATE_FEATURES
    orb = _ORB

    img_feat = preprocess_for_features(image)
    kp1, des1 = orb.detectAndCompute(img_feat, None)

    if des1 is None or des2 is None:
        raise ValueError("Impossible de calculer les descripteurs ORB.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.78 * n.distance:
            good.append(m)

    if len(good) < 25:
        raise ValueError(f"Pas assez de matches pour aligner l'image ({len(good)}).")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homographie introuvable.")

    h, w = tpl.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)

    return aligned

# =========================================================
# DETECTION DES CASES SUR LA TEMPLATE
# =========================================================

def detect_checkboxes_on_template(template_bgr):
    gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_local_illumination(gray)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )

    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri < 20:
            continue

        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        ar = w / float(h)

        if not (20 <= w <= 60 and 20 <= h <= 60):
            continue
        if not (0.75 <= ar <= 1.25):
            continue
        if area < 350 or area > 3200:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        candidates.append((x, y, w, h, cx, cy))

    filtered = []
    for cand in sorted(candidates, key=lambda t: (t[1], t[0])):
        x, y, w, h, cx, cy = cand
        keep = True
        for fx, fy, fw, fh, fcx, fcy in filtered:
            if abs(cx - fcx) < 8 and abs(cy - fcy) < 8:
                keep = False
                break
        if keep:
            filtered.append(cand)

    filtered.sort(key=lambda b: (round(b[1] / 80), b[0]))

    if len(filtered) != len(EXPECTED_CHECKBOX_ORDER):
        raise ValueError(
            f"Nombre de cases détectées = {len(filtered)} "
            f"au lieu de {len(EXPECTED_CHECKBOX_ORDER)}."
        )

    fields = {}
    for name, (x, y, w, h, _, _) in zip(EXPECTED_CHECKBOX_ORDER, filtered):
        mask = np.zeros((h, w), dtype=np.uint8)
        margin = max(4, int(min(h, w) * 0.18))
        cv2.rectangle(mask, (margin, margin), (w - margin - 1, h - margin - 1), 255, -1)

        fields[name] = {
            "bbox": (x, y, w, h),
            "inner_mask": mask
        }

    return fields

# =========================================================
# CHECKBOX
# =========================================================

def score_to_confidence(fill_ratio, total_area, diag_score, score):
    c_fill = clamp((fill_ratio - 0.015) / (0.12 - 0.015))
    c_area = clamp((total_area - 20) / (220 - 20))
    c_diag = clamp((diag_score - 0.03) / (0.30 - 0.03))
    c_score = clamp((score - 0.20) / (1.60 - 0.20))

    conf = (
        0.30 * c_fill +
        0.20 * c_area +
        0.20 * c_diag +
        0.30 * c_score
    )
    return round(float(clamp(conf)), 4)

def analyze_checkbox(aligned_gray, bbox, inner_mask):
    x, y, w, h = bbox
    roi = aligned_gray[y:y + h, x:x + w]

    if roi.size == 0:
        return {"checked": False, "confidence": 0.0, "score": 0.0}

    roi_big = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    mask_big = cv2.resize(inner_mask, (roi_big.shape[1], roi_big.shape[0]), interpolation=cv2.INTER_NEAREST)

    roi_big = normalize_local_illumination(roi_big)
    roi_big = cv2.GaussianBlur(roi_big, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        roi_big,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    ink = cv2.bitwise_and(th, th, mask=mask_big)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)

    mask_area = max(1, cv2.countNonZero(mask_big))
    ink_area = cv2.countNonZero(ink)
    fill_ratio = ink_area / float(mask_area)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    component_areas = []
    component_boxes = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        w0 = stats[i, cv2.CC_STAT_WIDTH]
        h0 = stats[i, cv2.CC_STAT_HEIGHT]

        if area >= 12:
            component_areas.append(area)
            component_boxes.append((x0, y0, w0, h0))

    n_components = len(component_areas)
    max_component = max(component_areas) if component_areas else 0
    total_area = sum(component_areas)

    ink_bin = (ink > 0).astype(np.uint8)
    diag1 = float(np.mean(np.diag(ink_bin)))
    diag2 = float(np.mean(np.diag(np.fliplr(ink_bin))))
    diag_score = max(diag1, diag2)

    proj_v = float(np.mean(np.sum(ink_bin, axis=0) > 0))
    proj_h = float(np.mean(np.sum(ink_bin, axis=1) > 0))

    elongated_bonus = 0.0
    for (_, _, bw, bh), area in zip(component_boxes, component_areas):
        ar = max(bw / max(1.0, bh), bh / max(1.0, bw))
        if ar > 1.6 and area > 20:
            elongated_bonus = max(elongated_bonus, 0.12)

    score = (
        3.0 * fill_ratio
        + 0.004 * min(total_area, 2500)
        + 0.010 * min(max_component, 500)
        + 0.22 * min(n_components, 6)
        + 1.6 * diag_score
        + 0.35 * proj_v
        + 0.35 * proj_h
        + elongated_bonus
    )

    confidence = score_to_confidence(fill_ratio, total_area, diag_score, score)

    checked = bool(
        confidence >= 0.50
        or fill_ratio > 0.065
        or (diag_score > 0.16 and total_area > 110)
    )

    return {
        "checked": checked,
        "confidence": confidence,
        "score": round(float(score), 4),
        "fill_ratio": round(float(fill_ratio), 4),
        "total_area": int(total_area),
        "diag_score": round(float(diag_score), 4),
    }

# =========================================================
# RESOLUTION
# =========================================================

def resolve_exclusive_group(raw, mapping, min_conf=0.35, margin=0.10):
    scored = []
    for key, label in mapping.items():
        conf = raw.get(key, {}).get("confidence", 0.0)
        scored.append((key, label, conf))

    scored.sort(key=lambda x: x[2], reverse=True)

    best_key, best_label, best_conf = scored[0]
    second_conf = scored[1][2] if len(scored) > 1 else 0.0

    if best_conf < min_conf:
        return {
            "value": None,
            "confidence": round(float(best_conf), 4),
            "selected_key": None,
            "ambiguous": False,
        }

    ambiguous = (best_conf - second_conf) < margin

    return {
        "value": best_label,
        "confidence": round(float(best_conf), 4),
        "selected_key": best_key,
        "ambiguous": bool(ambiguous),
    }

def resolve_multiple_symptoms(raw, keys,
                              min_conf=0.60,
                              min_score=4.5,
                              min_fill=0.020,
                              min_area=180,
                              min_diag=0.020):
    out = []

    for key, label in keys.items():
        item = raw.get(key, {})

        checked = bool(item.get("checked", False))
        conf = float(item.get("confidence", 0.0))
        score = float(item.get("score", 0.0))
        fill_ratio = float(item.get("fill_ratio", 0.0))
        total_area = float(item.get("total_area", 0.0))
        diag_score = float(item.get("diag_score", 0.0))

        accepted = bool(
            checked
            and conf >= min_conf
            and score >= min_score
            and (
                (fill_ratio >= min_fill and total_area >= min_area)
                or (diag_score >= min_diag and total_area >= min_area)
            )
        )

        if accepted:
            out.append(label)

    return out

# =========================================================
# MAPPING FRONT
# =========================================================

def map_sex(sexe):
    if not sexe:
        return "M"
    s = sexe.lower().strip()
    if "fém" in s:
        return "F"
    return "M"

def map_age_group(age):
    if not age:
        return "0-4 ans"
    s = age.lower().strip()
    if s in {"0-4 ans", "5-14 ans", "15-24 ans", "25-44 ans", "45-64 ans"}:
        return age
    if "65" in s:
        return "plus de 65 ans"
    return "0-4 ans"

def map_pathology(s):
    if not s:
        return "Varicelle"
    v = s.lower().strip()
    if "varicelle" in v:
        return "Varicelle"
    if "gripp" in v:
        return "Syndromes grippaux"
    if "diarrh" in v:
        return "Diarrhée aiguë"
    if "infection respiratoire" in v or v == "ira":
        return "IRA"
    return "Varicelle"

# =========================================================
# EXTRACTION PRINCIPALE
# =========================================================

def analyze_form_v2(image_path, include_date=False, minimal=True,debug=False):
    init_template_cache()

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")

    if img.shape[1] > 1600:
        img = resize_keep_ratio(img, width=1600)

    aligned = align_to_template_fast(img)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    raw = {}

    for name in EXPECTED_CHECKBOX_ORDER:
        bbox = _TEMPLATE_BOXES[name]["bbox"]
        inner_mask = _TEMPLATE_BOXES[name]["inner_mask"]
        raw[name] = analyze_checkbox(aligned_gray, bbox, inner_mask)

    sexe_resolution = resolve_exclusive_group(
        raw,
        {
            "sexe_masculin": "Masculin",
            "sexe_feminin": "Féminin"
        },
        min_conf=EXCLUSIVE_MIN_CONF,
        margin=EXCLUSIVE_MARGIN
    )

    age_resolution = resolve_exclusive_group(
        raw,
        {
            "age_0_4": "0-4 ans",
            "age_5_14": "5-14 ans",
            "age_15_24": "15-24 ans",
            "age_25_44": "25-44 ans",
            "age_45_64": "45-64 ans",
            "age_65p": "+65 ans"
        },
        min_conf=0.32,
        margin=EXCLUSIVE_MARGIN
    )

    disease_resolution = resolve_exclusive_group(
        raw,
        {
            "varicelle": "Varicelle",
            "syndromes_grippaux": "Syndromes grippaux",
            "diarrhee_aigue": "Diarrhée aiguë",
            "ira": "Infection respiratoire aiguë (IRA)"
        },
        min_conf=EXCLUSIVE_MIN_CONF,
        margin=EXCLUSIVE_MARGIN
    )

    selected_disease = disease_resolution["selected_key"]
    symptoms = []

    if selected_disease == "varicelle":
        symptoms = resolve_multiple_symptoms(
            raw,
            {
                "varicelle_debut_brutal": "Début brutal de l’éruption",
                "varicelle_fievre_moderee": "Fièvre modérée",
                "varicelle_eruption": "Éruption érythémato-vésiculeuse",
                "varicelle_prurit": "Prurit",
                "varicelle_duree": "Durée 3 à 4 jours",
                "varicelle_dessiccation": "Phase de dessiccation"
            }
        )
    elif selected_disease == "syndromes_grippaux":
        symptoms = resolve_multiple_symptoms(
            raw,
            {
                "grippe_fievre_39": "Fièvre > 39°C",
                "grippe_debut_brutal": "Début brutal des symptômes",
                "grippe_myalgies": "Myalgies",
                "grippe_signes_respiratoires": "Signes respiratoires"
            }
        )
    elif selected_disease == "diarrhee_aigue":
        symptoms = resolve_multiple_symptoms(
            raw,
            {
                "diarrhee_3_selles": "Au moins 3 selles liquides ou molles par jour",
                "diarrhee_moins_14j": "Évolution depuis moins de 14 jours",
                "diarrhee_motif_consultation": "Motif de consultation lié à ces symptômes"
            }
        )
    elif selected_disease == "ira":
        symptoms = resolve_multiple_symptoms(
            raw,
            {
                "ira_apparition_brutale": "Apparition brutale des symptômes",
                "ira_fievre": "Présence de fièvre ou sensation de fièvre",
                "ira_signes_respiratoires": "Présence de signes respiratoires"
            }
        )

    result = {
        "sex": map_sex(sexe_resolution["value"]),
        "ageGroup": map_age_group(age_resolution["value"]),
        "pathology": map_pathology(disease_resolution["value"]),
        "symptoms": symptoms,
        "needs_review": bool(
            sexe_resolution["value"] is None
            or age_resolution["value"] is None
            or disease_resolution["value"] is None
            or sexe_resolution["ambiguous"]
            or age_resolution["ambiguous"]
            or disease_resolution["ambiguous"]
        )
    }

    if include_date:
        result["date"] = None  # active seulement si tu veux vraiment Tesseract

    if minimal:
        return result

    return {
        "sex": result["sex"],
        "ageGroup": result["ageGroup"],
        "pathology": result["pathology"],
        "symptoms": result["symptoms"],
        "needs_review": result["needs_review"],
        "debug_raw": raw
    }
