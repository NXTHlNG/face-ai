from __future__ import annotations

from typing import Any

from app.config import settings


def build_recommendations(
    geometry: dict[str, Any],
    contrast: dict[str, Any],
    color: dict[str, Any],
    glasses_ratio: float | None,
) -> list[dict[str, Any]]:
    rv = settings.rules_version
    fs = geometry.get("face_shape", "unknown")
    cb = contrast.get("contrast_bucket", "medium")
    season = color.get("seasonal_guess", "unknown")
    season_12 = color.get("seasonal_twelve", "unknown")
    undertone = color.get("undertone_hint", "neutral")
    recs: list[dict[str, Any]] = []

    recs.extend(_glasses_rules(fs, glasses_ratio, rv))
    recs.extend(_hair_rules(fs, rv))
    recs.extend(_makeup_rules(cb, fs, rv))
    recs.extend(_color_clothing_rules(season, season_12, undertone, cb, rv))
    recs.extend(_jewelry_only(undertone, rv))
    recs.append(
        {
            "category": "general",
            "title": "Как интерпретировать результат",
            "detail": (
                "Один снимок не заменяет живой колор-анализ: меняйте освещение или пришлите "
                "ещё фото при дневном свете для более стабильных советов по цвету."
            ),
            "rule_id": "general_disclaimer_multi_photo",
            "rule_version": rv,
            "based_on": {},
        }
    )
    return recs


def _glasses_rules(
    face_shape: str,
    glasses_ratio: float | None,
    rv: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if glasses_ratio is not None and glasses_ratio > 0.02:
        out.append(
            {
                "category": "glasses",
                "title": "Очки на фото",
                "detail": (
                    "На снимке видны очки — для подбора оправы лучше приложить фото без очков "
                    "или с контактными линзами."
                ),
                "rule_id": "glasses_detected_warn",
                "rule_version": rv,
                "based_on": {"glasses_pixel_ratio": glasses_ratio},
            }
        )

    shape_advice = {
        "round": (
            "Угловатые или прямоугольные оправы и высота линзы добавят структуры; избегайте "
            "маленьких круглых очков."
        ),
        "square": (
            "Смягчите линии: овальные или слегка округлые оправы, средняя ширина линзы; "
            "тяжёлые прямоугольники могут усилить угловатость."
        ),
        "oval": (
            "Широкий выбор форм; ориентируйтесь на размер лица и переносицу — пропорциональная "
            "ширина оправы обычно выигрывает."
        ),
        "heart": (
            "Оправы с акцентом внизу (лёгкий cat-eye или ширина в зоне щёк), избегайте перегруза "
            "на лбу."
        ),
        "triangle": (
            "Верх полегче, снизу шире — хорошо смотрятся оправы с вертикальным акцентом вверху "
            "или умеренная ширина линзы без тяжёлого низа."
        ),
        "oblong": (
            "Широкие или высокие линзы, двойной мост — визуально укорачивают лицо; узкие "
            "вертикальные формы могут добавить длины."
        ),
        "diamond": (
            "Овальные и без обода по щекам; избегайте очень узких переносиц — балансируйте ширину скул."
        ),
        "unknown": (
            "Форму лица сложно однозначно классифицировать — ориентируйтесь на баланс ширины линзы "
            "и переносицы на практике при примерке."
        ),
    }
    detail = shape_advice.get(face_shape, shape_advice["unknown"])
    out.append(
        {
            "category": "glasses",
            "title": "Подбор оправы по форме лица",
            "detail": detail,
            "rule_id": f"glasses_face_shape_{face_shape}",
            "rule_version": rv,
            "based_on": {"face_shape": face_shape},
        }
    )
    return out


def _hair_rules(face_shape: str, rv: str) -> list[dict[str, Any]]:
    tips = {
        "round": "Объём у макушки или асимметричный пробор; удлините силуэт без лишней ширины у щёк.",
        "square": "Мягкие слои у лица, текстура и лёгкий объём — сгладить углы челюсти.",
        "heart": "Объём в зоне подбородка / у подбородка, шишки на макушке утяжеляют верх.",
        "triangle": "Объём у макушки и висков балансирует широкую челюсть; избегать тяжёлой длины только у подбородка.",
        "oblong": "Чёлка или горизонтальный объём по бокам; избегать только длинной гладкой вертикали.",
        "diamond": "Боковой объём у скул, мягкие длины до подбородка.",
        "oval": "Универсальная база — держите баланс длины и объёма под задачу образа.",
        "unknown": "Экспериментируйте с пробором и текстурой, пока не найдёте баланс скул и лба.",
    }
    return [
        {
            "category": "hair",
            "title": "Причёска и форма лица",
            "detail": tips.get(face_shape, tips["unknown"]),
            "rule_id": f"hair_face_shape_{face_shape}",
            "rule_version": rv,
            "based_on": {"face_shape": face_shape},
        }
    ]


def _makeup_rules(contrast_bucket: str, face_shape: str, rv: str) -> list[dict[str, Any]]:
    out = []
    if contrast_bucket == "high":
        out.append(
            {
                "category": "makeup",
                "title": "Контраст образа",
                "detail": (
                    "Высокий контраст лица часто хорошо сочетается с чёткими акцентами "
                    "(стрелка, насыщенная губа), но не перегружайте всё сразу."
                ),
                "rule_id": "makeup_contrast_high",
                "rule_version": rv,
                "based_on": {"contrast_bucket": contrast_bucket},
            }
        )
    elif contrast_bucket == "low":
        out.append(
            {
                "category": "makeup",
                "title": "Контраст образа",
                "detail": (
                    "Низкий контраст — мягкие переходы, глянцевые текстуры, наращивание "
                    "контраста через одежду и аксессуары."
                ),
                "rule_id": "makeup_contrast_low",
                "rule_version": rv,
                "based_on": {"contrast_bucket": contrast_bucket},
            }
        )
    if face_shape in ("round", "square", "triangle"):
        out.append(
            {
                "category": "makeup",
                "title": "Контуринг",
                "detail": (
                    "Лёгкий контур по скуловой линии и сторонам лба может добавить структуры "
                    "без резких полос."
                ),
                "rule_id": f"makeup_contouring_{face_shape}",
                "rule_version": rv,
                "based_on": {"face_shape": face_shape},
            }
        )
    return out


def _color_clothing_rules(
    season: str,
    season_12: str,
    undertone: str,
    contrast_bucket: str,
    rv: str,
) -> list[dict[str, Any]]:
    palettes_12 = {
        "light_spring": "Светлая тёплая весна: персик, светлый коралл, нежный аквамарин, "
        "мягкий перламутровый беж; избегать грязных серо-бежевых.",
        "true_spring": "Чистая тёплая весна: яркий коралл, яблочно-зелёный, чистое золото в акцентах.",
        "bright_spring": "Яркая тёплая весна (близко к контрастным): чистые тёплые яркие, бирюза, "
        "изумруд; аккуратно с чёрным.",
        "light_summer": "Светлое холодное лето: пепельная роза, жемчужный, светлый серо-голубой.",
        "true_summer": "Холодное лето: пудровые розово-сиреневые, графитово-синий, приглушённый малахит.",
        "soft_summer": "Мягкое холодное лето: дымчато-розовый, лавандово-серый, приглушённые сложные смеси.",
        "soft_autumn": "Мягкая осень: тёплый тауп, приглушённая горчица, глубокий оливковый.",
        "true_autumn": "Чистая осень: терракота, тёплый шоколад, горчичный, кармин как акцент.",
        "deep_autumn": "Глубокая осень: шоколад, тёмная оливка, изумрудно-зелёный; избегать пастели как базы.",
        "deep_winter": "Глубокая зима: чёрный и белый с холодным красным, изумруд, полночный синий.",
        "true_winter": "Чистая холодная зима: белоснежный, ледяной синий, холодный красный, фуксия.",
        "bright_winter": "Яркая холодная зима: контрастные чистые холодные, яркая фуксия, королевский синий.",
    }
    palettes_4 = {
        "spring": "Тёплые чистые акценты: коралл, изумруд, золотистый беж; избегать только холодной "
        "пепельной гаммы как базы.",
        "summer": "Приглушённые холодные пастели, розово-пудровое, лаванда, серо-голубой.",
        "autumn": "Глубокие тёплые оттенки: терракота, оливковый, горчичный, тёплый коричневый.",
        "winter": "Чистые холодные акценты: белоснежный, чёрный, холодный красный, яркий фуксия.",
        "unknown": "Ориентируйтесь на тест ткани при дневном свете — автооценка сезона приблизительная.",
    }
    detail = palettes_12.get(season_12) or palettes_4.get(season, palettes_4["unknown"])
    out = [
        {
            "category": "clothing_colors",
            "title": "Ориентир по палитре (не диагноз)",
            "detail": detail,
            "rule_id": f"palette_season_{season_12 if season_12 != 'unknown' else season}",
            "rule_version": rv,
            "based_on": {
                "seasonal_twelve": season_12,
                "seasonal_guess": season,
                "undertone_hint": undertone,
            },
        },
        {
            "category": "clothing_colors",
            "title": "Контраст принтов",
            "detail": (
                "При высоком контрасте лица часто лучше смотрятся чёткие цветовые пары; при низком — "
                "мягкие аналоговые сочетания."
                if contrast_bucket == "high"
                else "При низком контрасте лица фону и принтам часто идут приглушённые и сложные оттенки."
            ),
            "rule_id": f"clothing_print_contrast_{contrast_bucket}",
            "rule_version": rv,
            "based_on": {"contrast_bucket": contrast_bucket},
        },
    ]
    return out


def _jewelry_only(undertone: str, rv: str) -> list[dict[str, Any]]:
    metal = (
        "Золотистые металлы чаще гармонируют с тёплым подтоном."
        if undertone == "warm"
        else (
            "Серебро и платина часто лучше сочетаются с холодным подтоном."
            if undertone == "cool"
            else "Смешивайте металлы умеренно — нейтральный подтон допускает оба варианта."
        )
    )
    return [
        {
            "category": "jewelry",
            "title": "Металлы и акценты",
            "detail": metal,
            "rule_id": f"jewelry_metal_{undertone}",
            "rule_version": rv,
            "based_on": {"undertone_hint": undertone},
        }
    ]
