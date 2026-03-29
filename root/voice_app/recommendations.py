"""YouTube practice recommendations based on GRBAS scores and neck effort."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlencode, urlparse


# =========================
# Course base URLs
# =========================
COURSE_1_BASE_URL = "https://www.youtube.com/watch?v=dgGyow4nn3Q"
COURSE_2_BASE_URL = "https://www.youtube.com/watch?v=fIwaDTqaQ94"
COURSE_3_BASE_URL = "https://youtu.be/_rPOXZDUq30"


def yt_time_url(base_url: str, seconds: int) -> str:
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    query = parse_qs(parsed.query)

    if "youtu.be" in host:
        video_id = parsed.path.strip("/")
        params = {"v": video_id, "t": f"{seconds}s"}
        return f"https://www.youtube.com/watch?{urlencode(params)}"

    if "youtube.com" in host and parsed.path == "/watch":
        params = {key: values[-1] for key, values in query.items() if values}
        params["t"] = f"{seconds}s"
        return f"https://www.youtube.com/watch?{urlencode(params)}"

    if "youtube.com" in host and parsed.path == "/playlist":
        playlist_id = query.get("list", [""])[-1]
        if playlist_id:
            params = {
                "list": playlist_id,
                "start": str(seconds),
                "autoplay": "1",
            }
            return f"https://www.youtube.com/embed/videoseries?{urlencode(params)}"

    sep = "&" if parsed.query else "?"
    return f"{base_url}{sep}t={seconds}s"


@dataclass
class Resource:
    course: str
    section: str
    start_seconds: int
    url: str
    why: str


@dataclass
class Recommendation:
    measure: str
    score: int
    priority_rank: int
    resources: List[Resource]


# =========================
# Course 1 sections
# =========================
COURSE_1 = [
    Resource("קורס 1", "יציבה", 174, yt_time_url(COURSE_1_BASE_URL, 174), "בסיס ארגוני לקול"),
    Resource("קורס 1", "נשימה", 280, yt_time_url(COURSE_1_BASE_URL, 280), "תמיכה נשימתית"),
    Resource("קורס 1", "קוליות", 316, yt_time_url(COURSE_1_BASE_URL, 316), "הפקה קולית כללית"),
    Resource("קורס 1", "הקסם בסנכרון", 327, yt_time_url(COURSE_1_BASE_URL, 327), "תיאום גוף-נשימה-קול"),
    Resource("קורס 1", "יציבה - רקע", 423, yt_time_url(COURSE_1_BASE_URL, 423), "שחרור וארגון הגוף"),
    Resource("קורס 1", "יציבה - תרגיל מטאטא", 486, yt_time_url(COURSE_1_BASE_URL, 486), "הפחתת מאמץ וארגון יציבה"),
    Resource("קורס 1", "נשימה - חלק נוסף", 532, yt_time_url(COURSE_1_BASE_URL, 532), "שליטה בזרם האוויר"),
    Resource("קורס 1", "נשימה סרעפתית", 635, yt_time_url(COURSE_1_BASE_URL, 635), "חיזוק תמיכה נשימתית"),
    Resource("קורס 1", "נשימת אף", 737, yt_time_url(COURSE_1_BASE_URL, 737), "ויסות אוויר והרגעה"),
    Resource("קורס 1", "קוליות - תהודה", 793, yt_time_url(COURSE_1_BASE_URL, 793), "עידון איכות הקול"),
    Resource("קורס 1", "קוליות - זמזום", 878, yt_time_url(COURSE_1_BASE_URL, 878), "שחרור ועידון קול"),
    Resource("קורס 1", "תרגיל פיצוצים", 955, yt_time_url(COURSE_1_BASE_URL, 955), "הפעלה קולית והגאים"),
]


# =========================
# Course 2 sections
# =========================
COURSE_2 = [
    Resource("קורס 2", "דגשים לעמידה וישיבה", 0, yt_time_url(COURSE_2_BASE_URL, 0), "ארגון כללי והפחתת עומס"),
    Resource("קורס 2", "יציבה - כתפיים", 115, yt_time_url(COURSE_2_BASE_URL, 115), "שחרור חגורת כתפיים"),
    Resource("קורס 2", "נשימה ושכמות", 225, yt_time_url(COURSE_2_BASE_URL, 225), "תיאום נשימה-גב עליון"),
    Resource("קורס 2", "תמיכה נשמתית", 250, yt_time_url(COURSE_2_BASE_URL, 250), "תמיכה יציבה לקול"),
    Resource("קורס 2", "אורך נשימה", 343, yt_time_url(COURSE_2_BASE_URL, 343), "שליטה בזמן נשיפה"),
    Resource("קורס 2", "תרגיל קש", 390, yt_time_url(COURSE_2_BASE_URL, 390), "שחרור לחץ ושיפור זרימת אוויר"),
    Resource("קורס 2", "קוליות - הגאים", 496, yt_time_url(COURSE_2_BASE_URL, 496), "דיוק הפקה והגאים"),
    Resource("קורס 2", "זריקת הכדור", 528, yt_time_url(COURSE_2_BASE_URL, 528), "הקרנה ושחרור"),
    Resource("קורס 2", "השלכת קול", 544, yt_time_url(COURSE_2_BASE_URL, 544), "הקרנת קול יעילה"),
    Resource("קורס 2", "מציאת טון בסיס", 579, yt_time_url(COURSE_2_BASE_URL, 579), "ייצוב טון ונוחות קולית"),
]


# =========================
# Course 3 sections
# =========================
COURSE_3 = [
    Resource("קורס 3", "בסיס הלשון", 0, yt_time_url(COURSE_3_BASE_URL, 0), "שחרור עומס בבסיס הלשון"),
    Resource("קורס 3", "צוואר", 137, yt_time_url(COURSE_3_BASE_URL, 137), "שחרור מאמץ צווארי"),
    Resource("קורס 3", "נשימה - סטופר", 305, yt_time_url(COURSE_3_BASE_URL, 305), "בקרה על נשיפה"),
    Resource("קורס 3", "אגרוף", 407, yt_time_url(COURSE_3_BASE_URL, 407), "הפעלה ותמיכה"),
    Resource("קורס 3", "הגאים קוליים", 568, yt_time_url(COURSE_3_BASE_URL, 568), "דיוק בהפקה קולית"),
    Resource("קורס 3", "תרגילים לסוף יום עבודה", 648, yt_time_url(COURSE_3_BASE_URL, 648), "הרפיה והורדת עומס"),
    Resource("קורס 3", "הנחיה לתרגול עצמי", 776, yt_time_url(COURSE_3_BASE_URL, 776), "שימור תרגול"),
]

ALL_RESOURCES = COURSE_1 + COURSE_2 + COURSE_3


# =========================
# Fixed 3 exercises per GRBAS measure
# =========================
FIXED_RESOURCES: Dict[str, List[Resource]] = {
    "G": [
        Resource("קורס 1", "יציבה", 174, yt_time_url(COURSE_1_BASE_URL, 174), "בסיס ארגוני לקול"),
        Resource("קורס 1", "נשימה", 280, yt_time_url(COURSE_1_BASE_URL, 280), "תמיכה נשימתית"),
        Resource("קורס 2", "מציאת טון בסיס", 579, yt_time_url(COURSE_2_BASE_URL, 579), "ייצוב טון ונוחות קולית"),
    ],
    "R": [
        Resource("קורס 1", "קוליות - תהודה", 793, yt_time_url(COURSE_1_BASE_URL, 793), "עידון איכות הקול"),
        Resource("קורס 1", "קוליות - זמזום", 878, yt_time_url(COURSE_1_BASE_URL, 878), "שחרור ועידון קול"),
        Resource("קורס 3", "הגאים קוליים", 568, yt_time_url(COURSE_3_BASE_URL, 568), "דיוק בהפקה קולית"),
    ],
    "B": [
        Resource("קורס 1", "נשימה סרעפתית", 635, yt_time_url(COURSE_1_BASE_URL, 635), "חיזוק תמיכה נשימתית"),
        Resource("קורס 2", "תרגיל קש", 390, yt_time_url(COURSE_2_BASE_URL, 390), "שחרור לחץ ושיפור זרימת אוויר"),
        Resource("קורס 3", "נשימה - סטופר", 305, yt_time_url(COURSE_3_BASE_URL, 305), "בקרה על נשיפה"),
    ],
    "A": [
        Resource("קורס 1", "נשימה סרעפתית", 635, yt_time_url(COURSE_1_BASE_URL, 635), "חיזוק תמיכה נשימתית"),
        Resource("קורס 2", "תמיכה נשמתית", 250, yt_time_url(COURSE_2_BASE_URL, 250), "תמיכה יציבה לקול"),
        Resource("קורס 3", "אגרוף", 407, yt_time_url(COURSE_3_BASE_URL, 407), "הפעלה ותמיכה"),
    ],
    "S": [
        Resource("קורס 1", "יציבה", 174, yt_time_url(COURSE_1_BASE_URL, 174), "בסיס ארגוני לקול"),
        Resource("קורס 3", "צוואר", 137, yt_time_url(COURSE_3_BASE_URL, 137), "שחרור מאמץ צווארי"),
        Resource("קורס 3", "תרגילים לסוף יום עבודה", 648, yt_time_url(COURSE_3_BASE_URL, 648), "הרפיה והורדת עומס"),
    ],
    "N": [
        Resource("קורס 1", "יציבה - תרגיל מטאטא", 486, yt_time_url(COURSE_1_BASE_URL, 486), "ארגון הראש והצוואר והפחתת מאמץ"),
        Resource("קורס 2", "יציבה - כתפיים", 115, yt_time_url(COURSE_2_BASE_URL, 115), "שחרור חגורת הכתפיים והורדת עומס מהצוואר"),
        Resource("קורס 3", "צוואר", 137, yt_time_url(COURSE_3_BASE_URL, 137), "שחרור מאמץ צווארי ישיר"),
    ],
}


def recommend_by_grbas(
    scores: Dict[str, int],
    include_only_bad_scores: bool = True,
) -> List[Recommendation]:
    normalized = {k.upper(): int(v) for k, v in scores.items()}
    valid_measures = ["G", "R", "B", "A", "S"]

    sorted_measures: List[Tuple[str, int]] = sorted(
        [(m, normalized.get(m, 0)) for m in valid_measures],
        key=lambda x: (-x[1], valid_measures.index(x[0])),
    )

    recommendations: List[Recommendation] = []
    rank = 1

    for measure, score in sorted_measures:
        if include_only_bad_scores and score < 1:
            continue
        resources = FIXED_RESOURCES.get(measure, [])
        recommendations.append(
            Recommendation(
                measure=measure,
                score=score,
                priority_rank=rank,
                resources=resources,
            )
        )
        rank += 1

    # Fallback: if all scores are 0, show general maintenance exercises
    if not recommendations:
        recommendations.append(
            Recommendation(
                measure="G",
                score=0,
                priority_rank=1,
                resources=[
                    FIXED_RESOURCES["G"][0],  # יציבה
                    FIXED_RESOURCES["G"][1],  # נשימה
                    FIXED_RESOURCES["G"][2],  # מציאת טון בסיס
                ],
            )
        )

    return recommendations


def recommend_with_context(
    scores: Dict[str, int],
    neck_effort: Dict | None = None,
    include_only_bad_scores: bool = True,
) -> List[Recommendation]:
    recommendations = recommend_by_grbas(scores, include_only_bad_scores=include_only_bad_scores)
    rank = len(recommendations) + 1

    if neck_effort:
        neck_score = neck_effort.get("score")
        neck_confidence = float(neck_effort.get("confidence") or 0.0)
        neck_calibrated = bool(neck_effort.get("calibrated"))

        if isinstance(neck_score, int) and neck_score >= 1 and neck_confidence >= 0.45 and neck_calibrated:
            recommendations.append(
                Recommendation(
                    measure="N",
                    score=neck_score,
                    priority_rank=rank,
                    resources=FIXED_RESOURCES["N"],
                )
            )

    return recommendations


def flatten_recommendations(recommendations: List[Recommendation]) -> List[Dict]:
    return [
        {
            "measure": rec.measure,
            "score": rec.score,
            "priority_rank": rec.priority_rank,
            "resources": [asdict(r) for r in rec.resources],
        }
        for rec in recommendations
    ]
