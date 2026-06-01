# Обзор решений: seasonal / personal color analysis

Дата: 2026-05-21  
Контекст: R&D для `face-ai` — классификация цветотипа, undertone, рекомендации по стилю и макияжу.  
Подробные пайплайны «проблема → решение» — в [`best-soltions.md`](../best-soltions.md).

---

## Сводная таблица

| Работа | Тип | Подход | Таксономия | Лучший результат | Ссылка |
|--------|-----|--------|------------|------------------|--------|
| **Deep Armocromia** | Research + датасет | DL: masked RGB → FaRL/ResNeXt | 4 + 12 sub-types | 55.4% (4-class), 31.8% (12-class) | [GitHub](https://github.com/lorenzo-stacchio/Deep-Armocromia) |
| **AUA Capstone** | Academic (ImageJ) | HSV + Munsell axes + lookup | 16 types | Season accuracy не измерялась | PDF на Desktop |
| **Colorinsight** | Student prototype | FaRL + RetinaFace + ResNet18 | 4 seasons | ~60% (4-class) | [GitHub](https://github.com/PSY222/Colorinsight) |
| **Park IMCOM'18** | Academic prototype | Dlib + Lab rules + virtual makeup | 4 seasons + warm/cool | VM satisfaction ≥4/5 (foundation/blush) | [DOI](https://doi.org/10.1145/3164541.3164612) |
| **Colors Matter** | Research (CV pipeline) | Clustering + CIEDE2000 palette match | 8 skin + hair/iris + Warm/Cool | 80% skin 8-class | [arXiv](https://arxiv.org/abs/2505.14931) |

---

## 1. Deep Armocromia

**Авторы:** Stacchio et al.  
**Название:** *Deep Armocromia: A Novel Dataset for Face Seasonal Color Analysis and Classification*  
**Ссылка:** [github.com/lorenzo-stacchio/Deep-Armocromia](https://github.com/lorenzo-stacchio/Deep-Armocromia) · PDF: `Armocromy_ECCV1.pdf`

**Кратко:** Первый публичный expert-labeled датасет (~4920 фото) для Armocromia по Flow Theory (Migliaccio). Facer-парсинг → masked RGB → fine-tune FaRL16/64 или ResNeXt50 → классификация 4 или 12 сезонов.

**Ключевые решения:**
- Expert protocol разметки (2-stage training под Migliaccio)
- Pivot-set знаменитостей + CelebA для масштаба
- Perceptual hash dedup; split без пересечения лиц
- Top-k метрики (top-2 ~81% для 4-class)

**Главный урок для face-ai:** masked color zones + face-specific pretrain; benchmark на открытом датасете; top-k в confidence для соседних сезонов.

---

## 2. AUA Capstone — Automated Seasonal Color Classification

**Авторы:** Khachatryan, Asryan, Sargsyan (American University of Armenia, Spring 2025)  
**Название:** *Automated Seasonal Color Classification and Makeup Recommendation*  
**Ссылка:** PDF `Automated_Seasonal_Color_Classification_and_Makeup_Recommendation-...pdf` на Desktop

**Кратко:** Полностью rule-based пайплайн в ImageJ: HSV skin detection → отдельная сегментация глаз/губ/волос → 4 оси Munsell (hue, chroma, value, contrast) → lookup 16 типов → генерация swatch-палитры → match продуктов из Kaggle Cosmetic Dataset.

**Ключевые решения:**
- 4 explainable оси со шкалой 1–5
- Value = 0.7×skin + 0.15×eyes + 0.15×hair
- Contrast ratio `(L_min+0.05)/(L_max+0.05)`
- Skin palette без red bias (hue 13–24); +20% brightness при плохом свете
- Lip 3-tier brightness clusters; hair = mask subtraction

**Главный урок для face-ai:** explicit Munsell breakdown для UX; product match по nearest swatch; contrast makeup shades.

---

## 3. Colorinsight

**Авторы:** PSY222 (корейский student project)  
**Название:** 퍼스널컬러 진단모델 (Personal Color Diagnosis Model)  
**Ссылка:** [github.com/PSY222/Colorinsight](https://github.com/PSY222/Colorinsight) · [Notion experiments](https://tar-tilapia-c6d.notion.site/403c8d583e3a4f6bb9f76ea6efd991d5)

**Кратко:** FastAPI-сервис: RetinaFace + FaRL parsing → skin mask на чёрном фоне → ResNet18 → 4 сезона. Документированный ablation: RGB L2 (~25%) → tabular ML (~25%) → CNN (~60%).

**Ключевые решения:**
- FaRL устойчивее альтернатив на поворотах
- Skin-mask preview как UX-hook
- Legacy path через губы (L2 vote) — fallback
- ~750 Korean celebs, augmentation из-за малого объёма

**Главный урок для face-ai:** parsing quality критична; iterative ablation; skin-mask preview для доверия пользователя.

---

## 4. Park et al. — Personal Color + Virtual Makeup

**Авторы:** J. Park, H. Kim, S. Ji, E. Hwang  
**Название:** *An Automatic Virtual Makeup Scheme Based on Personal Color Analysis*  
**Ссылка:** [DOI 10.1145/3164541.3164612](https://doi.org/10.1145/3164541.3164612) · PDF: `3164541.3164612.pdf` на Desktop

**Кратко:** End-to-end без DL: Dlib 68 landmarks → извлечение iris (Hough + K-means), hair (Canny), skin (cheek/jaw ROI) → Lab rules (undertone a vs b; season по L-contrast, порог 13) → expert makeup DB + анкета → виртуальный макияж (6 продуктов) с Lab-blend и Gaussian-blurred masks.

**Ключевые решения:**
- Cheek/jaw skin ROI — меньше солнечного bias
- Lab blend с параметром α для foundation/blush/lip
- Gaussian blur на makeup masks — natural gradation
- Expert DB: `(season, occasion, demographics)` → Lab-цвета
- User study: 4 оси (color, region size, location, intensity)

**Главный урок для face-ai:** единственная работа с полным VM pipeline; Lab blend + blurred masks; contrast→season heuristic; questionnaire для стиля.

**Слабые места (authors):** hair Canny seg; flat eyeline/eyebrow templates; undertone a>b слишком грубо.

---

## 5. Colors Matter — AI-Driven Exploration of Human Feature Colors

**Авторы:** R. Alyoubi, T. Alharbi, A. Alghamdi, Y. Alshehri, E. Alghamdi  
**Название:** *Colors Matter: AI-Driven Exploration of Human Feature Colors*  
**Ссылка:** [arXiv:2505.14931](https://arxiv.org/abs/2505.14931) · PDF: `2505.14931v1.pdf` на Desktop · код заявлен в GitHub (ссылка в статье)

**Кратко:** Мультимодальный CV-пайплайн: face photo → skin tone (8 classes), hair color, iris color; wrist photo → Warm/Cool undertone по венам. Rule-based: X-Means/K-means → CIEDE2000 match к reference swatches. Не seasonal analysis.

**Ключевые решения:**
- **Gaussian blur на skin ROI до clustering** — accuracy 42% → 80%
- **HSV** для skin cluster, **LAB** для hair/iris/undertone
- **CIEDE2000** palette matching beat SVM+ResNet18 (80% vs 76%)
- Facer + Timm (skin); RetinaFace + LaPa (hair); Dlib 68 (iris)
- Wrist veins + Delta E для undertone (Warm 80%, Cool 70%)
- CASCo/PERLA **отвергнуты** — плохая сегментация

**Главный урок для face-ai:** blur→cluster→Delta E; разные color spaces под разные признаки; wrist как второй модальность; discrete skin class как дополнение к 12-season.

**Слабые места:** light hair (20–50%); hazel/grey iris; ad-hoc reference values; eval hair/iris ~10 img/class.

---

## Что брать / не брать (синтез)

### Брать

| Идея | Источник |
|------|----------|
| Masked zones → classifier | Deep Armocromia, Colorinsight |
| Top-k neighbor seasons | Deep Armocromia |
| 4 Munsell axes + explainable scores | AUA |
| Skin-mask preview | Colorinsight |
| Lab blend + blurred makeup masks | Park |
| Cheek/jaw skin ROI | Park |
| Blur + k-means + CIEDE2000 | Colors Matter |
| HSV skin / LAB hair-iris split | Colors Matter |
| Wrist vein undertone (optional upload) | Colors Matter |

### Не брать

| Антипаттерн | Источник |
|-------------|----------|
| HSV skin detection без parsing | AUA |
| RGB L2 как primary classifier | Colorinsight |
| Undertone = a > b | Park |
| Hair Canny / iris Hough+K-means | Park |
| Two-stage hierarchical skin (64%) | Colors Matter |
| CASCo / cosine similarity undertone | Colors Matter |
| 8-class skin вместо seasonal taxonomy | Colors Matter |

---

## Связанные артефакты

- [`best-soltions.md`](../best-soltions.md) — детальные пайплайны, таблицы «проблема → решение», roadmap
- [`face-parsing-models-comparison-2026-05.md`](./face-parsing-models-comparison-2026-05.md) — сравнение парсеров (FaRL, SegFace, BiSeNet и др.)
