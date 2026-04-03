import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Chemin Linux pour Render / Docker
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

TEMPLATE_PATH = str(BASE_DIR / "doc2.png")
DEBUG_DIR = str(BASE_DIR / "debug_outputs")

DATE_OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789./-'

CANONICAL_W = 1448
CANONICAL_H = 2048

# Seuils globaux
EXCLUSIVE_MIN_CONF = 0.35
EXCLUSIVE_MARGIN = 0.08
SYMPTOM_MIN_CONF = 0.42

# Ordre attendu des cases sur la template
EXPECTED_CHECKBOX_ORDER = [
    "sexe_masculin",
    "sexe_feminin",

    "age_0_4",
    "age_5_14",
    "age_15_24",
    "age_25_44",
    "age_45_64",
    "age_65p",

    "varicelle",
    "syndromes_grippaux",

    "varicelle_debut_brutal",
    "varicelle_fievre_moderee",
    "grippe_fievre_39",
    "varicelle_eruption",
    "grippe_debut_brutal",
    "varicelle_prurit",
    "grippe_myalgies",
    "varicelle_duree",
    "grippe_signes_respiratoires",
    "varicelle_dessiccation",
    "diarrhee_aigue",
    "ira",
    "ira_apparition_brutale",
    "diarrhee_3_selles",
    "ira_fievre",
    "diarrhee_moins_14j",
    "diarrhee_motif_consultation",
    "ira_signes_respiratoires",
]


# =========================================================
# OUTILS
# =========================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def resize_keep_ratio(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is not None:
        scale = width / w
        return cv2.resize(img, (width, int(h * scale)))
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def preprocess_for_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    return sharp


def normalize_local_illumination(gray):
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def roi_from_pixels(img, box):
    x, y, w, h = box
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img.shape[1], int(x + w))
    y2 = min(img.shape[0], int(y + h))
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


# =========================================================
# ALIGNEMENT
# =========================================================

def align_to_template(image, template):
    img_feat = preprocess_for_features(image)
    tpl_feat = preprocess_for_features(template)

    orb = cv2.ORB_create(
        nfeatures=8000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=7
    )

    kp1, des1 = orb.detectAndCompute(img_feat, None)
    kp2, des2 = orb.detectAndCompute(tpl_feat, None)

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

    if len(good) < 30:
        raise ValueError(f"Pas assez de bons matches pour aligner l'image ({len(good)}).")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homographie introuvable.")

    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_CUBIC)

    inliers = int(mask.sum()) if mask is not None else 0
    return aligned, H, len(good), inliers


# =========================================================
# DETECTION DES CASES SUR LA TEMPLATE
# =========================================================

def detect_checkboxes_on_template(template_bgr, debug=False):
    gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    gray = normalize_local_illumination(gray)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
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
            f"Nombre de cases détectées sur la template = {len(filtered)} "
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

    if debug:
        ensure_dir(DEBUG_DIR)
        dbg = template_bgr.copy()
        for name, item in fields.items():
            x, y, w, h = item["bbox"]
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                dbg,
                name,
                (x, max(18, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )
        cv2.imwrite(str(Path(DEBUG_DIR) / "00_template_detected_boxes.jpg"), dbg)

    return fields


# =========================================================
# SCORE / CONFIANCE CASE
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
        return {
            "checked": False,
            "confidence": 0.0,
            "fill_ratio": 0.0,
            "n_components": 0,
            "max_component": 0,
            "total_area": 0,
            "diag_score": 0.0,
            "proj_v": 0.0,
            "proj_h": 0.0,
            "score": 0.0,
            "ink_mask": None
        }

    roi_big = cv2.resize(roi, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    mask_big = cv2.resize(
        inner_mask,
        (roi_big.shape[1], roi_big.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

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

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel_open)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel_close)

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

    confidence = score_to_confidence(
        fill_ratio=fill_ratio,
        total_area=total_area,
        diag_score=diag_score,
        score=score
    )

    checked = bool(
        confidence >= 0.50
        or fill_ratio > 0.065
        or (diag_score > 0.16 and total_area > 110)
    )

    return {
        "checked": checked,
        "confidence": confidence,
        "fill_ratio": round(float(fill_ratio), 4),
        "n_components": int(n_components),
        "max_component": int(max_component),
        "total_area": int(total_area),
        "diag_score": round(float(diag_score), 4),
        "proj_v": round(float(proj_v), 4),
        "proj_h": round(float(proj_h), 4),
        "score": round(float(score), 4),
        "ink_mask": ink
    }


# =========================================================
# RESOLUTION METIER
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
            "ranking": [{"key": k, "label": l, "confidence": c} for k, l, c in scored]
        }

    ambiguous = (best_conf - second_conf) < margin

    return {
        "value": best_label,
        "confidence": round(float(best_conf), 4),
        "selected_key": best_key,
        "ambiguous": bool(ambiguous),
        "ranking": [{"key": k, "label": l, "confidence": c} for k, l, c in scored]
    }


def resolve_multiple_symptoms(raw, checkbox_debug, keys,
                              min_conf=0.60,
                              min_score=4.5,
                              min_fill=0.020,
                              min_area=180,
                              min_diag=0.020):
    out = []
    ranking = []

    for key, label in keys.items():
        raw_item = raw.get(key, {})
        dbg = checkbox_debug.get(key, {})

        checked = bool(raw_item.get("checked", False))
        conf = float(raw_item.get("confidence", 0.0))
        score = float(raw_item.get("score", 0.0))

        fill_ratio = float(dbg.get("fill_ratio", 0.0))
        total_area = float(dbg.get("total_area", 0.0))
        diag_score = float(dbg.get("diag_score", 0.0))

        accepted = bool(
            checked
            and conf >= min_conf
            and score >= min_score
            and (
                (fill_ratio >= min_fill and total_area >= min_area)
                or (diag_score >= min_diag and total_area >= min_area)
            )
        )

        ranking.append({
            "key": key,
            "label": label,
            "checked": checked,
            "confidence": round(conf, 4),
            "score": round(score, 4),
            "fill_ratio": round(fill_ratio, 4),
            "total_area": int(total_area),
            "diag_score": round(diag_score, 4),
            "accepted": accepted
        })

        if accepted:
            out.append(label)

    ranking.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "values": out,
        "ranking": ranking
    }


# =========================================================
# OCR DATE
# =========================================================

def normalize_date_text(text):
    text = text.strip()
    text = text.replace(" ", "")
    text = text.replace(",", ".")
    text = text.replace("_", "")
    text = re.sub(r"[^0-9./-]", "", text)
    text = text.replace(".", "/").replace("-", "/")

    m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", text)
    if m:
        return m.group(1)
    return text if text else None


def extract_date(aligned_img):
    gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

    x, y, w, h = (1020, 20, 360, 70)
    roi = gray[y:y + h, x:x + w]

    roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    roi = normalize_local_illumination(roi)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    roi = clahe.apply(roi)

    roi_bin = cv2.adaptiveThreshold(
        roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    text = pytesseract.image_to_string(roi_bin, lang="eng", config=DATE_OCR_CONFIG)
    return normalize_date_text(text), roi_bin, (x, y, x + w, y + h)


# =========================================================
# EXTRACTION PRINCIPALE
# =========================================================

def analyze_form(image_path, template_path=TEMPLATE_PATH, debug=False):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")

    template = cv2.imread(str(template_path))
    if template is None:
        raise ValueError(f"Impossible de charger la template : {template_path}")

    template = cv2.resize(template, (CANONICAL_W, CANONICAL_H))

    if img.shape[1] > 1800:
        img = resize_keep_ratio(img, width=1800)

    blur = blur_score(img)

    template_boxes = detect_checkboxes_on_template(template, debug=debug)
    aligned, H, n_good_matches, n_inliers = align_to_template(img, template)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    result = {
        "date": None,
        "sexe": None,
        "sexe_confidence": 0.0,
        "sexe_ambiguous": False,
        "age": None,
        "age_confidence": 0.0,
        "age_ambiguous": False,
        "syndrome_principal": None,
        "syndrome_principal_confidence": 0.0,
        "syndrome_principal_ambiguous": False,
        "details": {},
        "raw_checkboxes": {},
        "quality": {
            "blur_score": float(blur),
            "good_matches": int(n_good_matches),
            "inliers": int(n_inliers),
            "homography_found": bool(H is not None),
            "checkbox_debug": {}
        }
    }

    ensure_dir(DEBUG_DIR) if debug else None
    debug_img = aligned.copy() if debug else None

    date_text, date_roi, date_coords = extract_date(aligned)
    result["date"] = date_text

    if debug:
        masks_dir = Path(DEBUG_DIR) / "checkbox_masks"
        ensure_dir(masks_dir)

    for name in EXPECTED_CHECKBOX_ORDER:
        bbox = template_boxes[name]["bbox"]
        inner_mask = template_boxes[name]["inner_mask"]

        analysis = analyze_checkbox(aligned_gray, bbox, inner_mask)

        result["raw_checkboxes"][name] = {
            "checked": bool(analysis["checked"]),
            "confidence": float(analysis["confidence"]),
            "score": float(analysis["score"])
        }

        details_to_store = {k: v for k, v in analysis.items() if k != "ink_mask"}
        result["quality"]["checkbox_debug"][name] = details_to_store

        if debug:
            x, y, w, h = bbox
            checked = analysis["checked"]
            confidence = analysis["confidence"]

            color = (0, 200, 0) if checked else (0, 0, 255)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                debug_img,
                f"{'X' if checked else '-'} {confidence:.2f}",
                (x, max(18, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA
            )

            ink_mask = analysis.get("ink_mask")
            if ink_mask is not None:
                cv2.imwrite(
                    str(masks_dir / f"{name}_{'checked' if checked else 'empty'}.png"),
                    ink_mask
                )

    sexe_resolution = resolve_exclusive_group(
        result["raw_checkboxes"],
        {
            "sexe_masculin": "Masculin",
            "sexe_feminin": "Féminin"
        },
        min_conf=EXCLUSIVE_MIN_CONF,
        margin=EXCLUSIVE_MARGIN
    )

    result["sexe"] = sexe_resolution["value"]
    result["sexe_confidence"] = sexe_resolution["confidence"]
    result["sexe_ambiguous"] = sexe_resolution["ambiguous"]
    result["quality"]["sexe_ranking"] = sexe_resolution["ranking"]

    age_resolution = resolve_exclusive_group(
        result["raw_checkboxes"],
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

    result["age"] = age_resolution["value"]
    result["age_confidence"] = age_resolution["confidence"]
    result["age_ambiguous"] = age_resolution["ambiguous"]
    result["quality"]["age_ranking"] = age_resolution["ranking"]

    disease_resolution = resolve_exclusive_group(
        result["raw_checkboxes"],
        {
            "varicelle": "Varicelle",
            "syndromes_grippaux": "Syndromes grippaux",
            "diarrhee_aigue": "Diarrhée aiguë",
            "ira": "Infection respiratoire aiguë (IRA)"
        },
        min_conf=EXCLUSIVE_MIN_CONF,
        margin=EXCLUSIVE_MARGIN
    )

    result["syndrome_principal"] = disease_resolution["value"]
    result["syndrome_principal_confidence"] = disease_resolution["confidence"]
    result["syndrome_principal_ambiguous"] = disease_resolution["ambiguous"]
    result["quality"]["disease_ranking"] = disease_resolution["ranking"]

    selected_disease = disease_resolution["selected_key"]

    details = {
        "varicelle": {
            "selectionnee": selected_disease == "varicelle",
            "symptomes": []
        },
        "syndromes_grippaux": {
            "selectionnee": selected_disease == "syndromes_grippaux",
            "symptomes": []
        },
        "diarrhee_aigue": {
            "selectionnee": selected_disease == "diarrhee_aigue",
            "symptomes": []
        },
        "ira": {
            "selectionnee": selected_disease == "ira",
            "symptomes": []
        }
    }

    if selected_disease == "varicelle":
        symptoms = resolve_multiple_symptoms(
            result["raw_checkboxes"],
            result["quality"]["checkbox_debug"],
            {
                "varicelle_debut_brutal": "Début brutal de l’éruption",
                "varicelle_fievre_moderee": "Fièvre modérée",
                "varicelle_eruption": "Éruption érythémato-vésiculeuse",
                "varicelle_prurit": "Prurit",
                "varicelle_duree": "Durée 3 à 4 jours",
                "varicelle_dessiccation": "Phase de dessiccation"
            },
            min_conf=0.60,
            min_score=4.5,
            min_fill=0.020,
            min_area=180,
            min_diag=0.020
        )
        details["varicelle"]["symptomes"] = symptoms["values"]
        details["varicelle"]["ranking"] = symptoms["ranking"]

    elif selected_disease == "syndromes_grippaux":
        symptoms = resolve_multiple_symptoms(
            result["raw_checkboxes"],
            result["quality"]["checkbox_debug"],
            {
                "grippe_fievre_39": "Fièvre > 39°C",
                "grippe_debut_brutal": "Début brutal des symptômes",
                "grippe_myalgies": "Myalgies",
                "grippe_signes_respiratoires": "Signes respiratoires"
            },
            min_conf=0.60,
            min_score=4.5,
            min_fill=0.020,
            min_area=180,
            min_diag=0.020
        )
        details["syndromes_grippaux"]["symptomes"] = symptoms["values"]
        details["syndromes_grippaux"]["ranking"] = symptoms["ranking"]

    elif selected_disease == "diarrhee_aigue":
        symptoms = resolve_multiple_symptoms(
            result["raw_checkboxes"],
            result["quality"]["checkbox_debug"],
            {
                "diarrhee_3_selles": "Au moins 3 selles liquides ou molles par jour",
                "diarrhee_moins_14j": "Évolution depuis moins de 14 jours",
                "diarrhee_motif_consultation": "Motif de consultation lié à ces symptômes"
            },
            min_conf=0.60,
            min_score=4.5,
            min_fill=0.020,
            min_area=180,
            min_diag=0.020
        )
        details["diarrhee_aigue"]["symptomes"] = symptoms["values"]
        details["diarrhee_aigue"]["ranking"] = symptoms["ranking"]

    elif selected_disease == "ira":
        symptoms = resolve_multiple_symptoms(
            result["raw_checkboxes"],
            result["quality"]["checkbox_debug"],
            {
                "ira_apparition_brutale": "Apparition brutale des symptômes",
                "ira_fievre": "Présence de fièvre ou sensation de fièvre",
                "ira_signes_respiratoires": "Présence de signes respiratoires"
            },
            min_conf=0.60,
            min_score=4.5,
            min_fill=0.020,
            min_area=180,
            min_diag=0.020
        )
        details["ira"]["symptomes"] = symptoms["values"]
        details["ira"]["ranking"] = symptoms["ranking"]

    result["details"] = details

    result["needs_review"] = bool(
        result["sexe"] is None
        or result["age"] is None
        or result["syndrome_principal"] is None
        or result["sexe_ambiguous"]
        or result["age_ambiguous"]
        or result["syndrome_principal_ambiguous"]
    )

    if debug:
        cv2.imwrite(str(Path(DEBUG_DIR) / "01_aligned.jpg"), aligned)

        x1, y1, x2, y2 = date_coords
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imwrite(str(Path(DEBUG_DIR) / "02_debug_boxes.jpg"), debug_img)
        cv2.imwrite(str(Path(DEBUG_DIR) / "03_date_roi.jpg"), date_roi)

    return convert_numpy_types(result)


# =========================================================
# BATCH
# =========================================================

def analyze_folder(folder_path, template_path=TEMPLATE_PATH, debug=False):
    folder = Path(folder_path)
    results = []

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in valid_ext:
            continue

        try:
            data = analyze_form(str(img_path), template_path=template_path, debug=debug)
            results.append({
                "file_name": img_path.name,
                "success": True,
                "data": data
            })
            print(f"[OK] {img_path.name}")
        except Exception as e:
            results.append({
                "file_name": img_path.name,
                "success": False,
                "error": str(e)
            })
            print(f"[ERREUR] {img_path.name}: {e}")

    return results