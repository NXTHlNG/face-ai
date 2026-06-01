# History

## 2026-05-21 — Поиск skills для планирования face/style приложения

**Запрос:** найти skill для проектирования приложения анализа внешности, формы лица, рекомендаций по стилю/макияжу/причёскам, оценки образов, примерки причёсок и очков.

**Результат поиска (`npx skills find`):**
- Единого skill под всю доменную область **не найдено**.
- У проекта уже есть core API (`/analyze`): форма лица, геометрия, цвет/контраст, эвристические рекомендации (FastAPI + OpenCV + MediaPipe/dlib).

**Рекомендованный стек skills:**
1. `yofine/skills@blueprinter` — технические blueprint-диаграммы архитектуры
2. `aj-geddes/useful-ai-prompts@architecture-diagrams` — системные диаграммы
3. `mindrally/skills@computer-vision-opencv` — CV-пайплайн (landmarks, parsing)
4. `eachlabs/skills@eachlabs-fashion-ai` + `eachlabs/skills@virtual-try-on` — примерка и fashion AI
5. `alirezarezvani/claude-skills@code-to-prd` — PRD из существующего кода

## 2026-05-21 — Уточнение: мобильное приложение на React Native

**Контекст:** клиент — React Native (Expo), backend — существующий FastAPI `face-ai`.

**Дополненный стек skills для RN:**
1. `jeffallan/claude-skills@react-native-expert` (2.5K) — основной RN skill
2. `mindrally/skills@expo-react-native-typescript` (887) — Expo + TypeScript
3. `ruvnet/ruflo@agent-spec-mobile-react-native` (481) — mobile spec/architecture
4. `aj-geddes/useful-ai-prompts@react-native-app` (362) — navigation, API, state
5. `alinaqi/claude-bootstrap@ui-mobile` (760) — mobile UI/UX patterns

## 2026-05-21 — Исследование коммерческих AI color analysis сервисов (2024–2026)

**Запрос:** собрать 10–15 структурированных evidence items по Dressika, HueCheck, Season Palette, SeasonAI, Palette, BeautyLove, personalcolor.app и аналогам.

**Источники (primary):** официальные сайты и App Store listings.

**Ключевые находки:**
- **GPT-4V явно:** SeasonAI (`season-ai.com`) — единственный из выборки с явным указанием GPT-4 vision.
- **Computer vision / multimodal без бренда модели:** HueCheck (white paper + homepage: «advanced computer vision»), Dressika, MySeason, BeautyLove, Palette — AI image analysis без названия конкретной LLM.
- **12-season vs 4-season:** большинство продают 12 sub-seasons; бесплатные tier часто дают только 4 parent seasons (HueCheck free, Season Palette web quiz).
- **Accuracy claims:** Dressika до 97% (validation color consultants), HueCheck 96% match accuracy (HueCheck It™), HueCheck «comparable to professional in-person».
- **Pricing:** от free tier до $79.99/год (HueCheck premium), pay-as-you-go credits (SeasonAI €1.99+, BeautyLove 10 credits/run), one-time $5.99–$9.99 (MySeason, HueCheck web).
- **Vendor-stated limitations:** SeasonAI — no refunds after credit use; BeautyLove — «Automated color reads can be wrong»; Dressika — results may fall between two palettes; HueCheck white paper — AI не заменяет нюансы human colorist.

## 2026-05-21 — Литературный обзор: автоматическое определение undertone / skin tone (2023–2026)

**Запрос:** 8–12 evidence items (JSON) по академическим работам: ITA, Monk Skin Tone, CIELAB, ViT, illumination correction.

**Источники (проверены через arXiv / ICCV Open Access / npj Digital Medicine / Annals of Dermatology):**
- Kalb et al. arXiv:2308.09640 (2023) — расхождение ITA-методов
- Thong et al. ICCV 2023 — многомерный hue angle (red–yellow) + L* tone
- Schumann et al. arXiv:2305.09073 (2024) — MST annotation, ограничения ITA при in-the-wild
- Ran et al. ICCV 2025 (HUST) — illumination-invariant albedo, VQGAN
- Bencević et al. arXiv:2504.04494 (2025) — CIELAB/ITA на синтетических dermoscopy
- npj Digital Medicine s41746-025-01770-4 (2025) — автоматический ITA из CIELAB + DensePose/OpenFace
- npj Digital Medicine s41746-025-02245-2 (2025) — MST vs FST, undertone (Pantone), слабая корреляция image-ITA
- Matias et al. arXiv:2603.02475 (2026) — SkinToneNet (ViT) + STW dataset (MST)
- Ha et al. Ann Dermatol 2026 — Transformer patch-level illumination removal

**Вывод:** undertone в CV чаще моделируют через hue angle h* в CIELAB (Thong), а не отдельной «warm/cool» меткой; ITA чувствителен к освещению; MST предпочтительнее FST для автоматической классификации; ViT превосходит classic CV на in-the-wild MST.

## 2026-05-21 — Deep Research: программное определение цветотипа по фото (2025–2026)

**Режим:** deep-research / deep mode  
**Тема:** актуальные решения программного определения цветотипа (seasonal color analysis) по фотографии.

**Артефакты:** `C:\Users\NXTHING\Documents\Color_Type_Analysis_Research_20260521\`
- `research_report_20260521_color_type_analysis.md` — основной отчёт (~4 800 слов, 8 findings)
- `research_report_20260521_color_type_analysis.html` — McKinsey-style HTML
- `sources.jsonl` (29), `evidence.jsonl` (20), `run_manifest.json`

**Валидация:** `validate_report.py` — все 9 проверок PASS.

**Ключевые выводы для face-ai:**
1. Рынок 2026: 12-season default; 3 tech-класса — CV+rules (HueCheck), CNN/ViT (~55% на 4 сезона, Deep Armocromia), GPT-4V (SeasonAI).
2. Колорimetрия (ITA, Monk, hue angle) != сезонный label; academic SOTA для tone 80–92%, для season ~55%.
3. Bottleneck — освещение/AWB; приоритет R&D: HUST-style albedo или time-aware AWB (ICCV 2025).
4. Архитектура face-ai (`color_contrast.py`) aligned с AUA 2025 / Colors Matter — parsing + LAB + fused undertone + probabilistic 12-season.
5. Рекомендации: multi-photo API, illuminant pre-step, optional vein/draping, прозрачный confidence vs vendor claims 97%.

**Outline adaptation (Phase 4.5):** после triangulation добавлены отдельные findings по illumination gap и season vs colorimetry (Kalb 2023, HUST 2025, Pinsker 2025).

## 2026-05-21 — Разбор статьи Deep Armocromia (Armocromy_ECCV1.pdf)

**Запрос:** что конкретно сделали в исследовании и какие выводы.

**Источник:** `C:/Users/NXTHING/Desktop/Armocromy_ECCV1.pdf` — Stacchio et al., «Deep Armocromia: A Novel Dataset for Face Seasonal Color Analysis and Classification» (University of Macerata, финансирование Moda Metrics s.r.l.).

**Суть работы:**
1. Первый публичный датасет **Deep Armocromia** (~4920 фото) с разметкой по **Armocromia Flow Theory**: 4 сезона + 12 sub-types (Warm/Light/Bright/Cool/Soft/Deep).
2. Сбор: ~1900 «pivotal» фото знаменитостей + ~3000 из CelebA, аннотация обученными студентами под экспертом (Rossella Migliaccio).
3. Препроцессинг: dedup (perceptual hash), face parsing (Facer) — маски кожи/волос/глаз.
4. Бенчмарки: 4-class season и 12-class sub-type; fine-tuning frozen backbone (FaRL16/64, ResNeXt50) + 2 FC слоя.

**Результаты:** season accuracy **55.4%** (FaRL64), top-2 **80.8%**; 12 sub-types **31.8%** (FaRL16), top-3 **66.3%**. Путаница Autumn↔Winter, Spring↔Summer, соседние sub-types.

**Выводы авторов:** задача сложная; DL может учить implicit color features, но coarse лучше fine; план — больше данных, hierarchical/ordinal learning, сравнение с classical CV.

**Датасет:** https://github.com/lorenzo-stacchio/Deep-Armocromia

## 2026-05-21 — best-soltions.md: пайплайн Deep Armocromia

**Запрос:** список шагов пайплайна/проблем и как они решены в работе.

**Артефакт:** `best-soltions.md` — таблица 18 этапов (проблема → решение), результаты 4/12 классов, нерешённые gaps, маппинг на face-ai.

## 2026-05-21 — Саммари capstone AUA: Automated Seasonal Color Classification

**Источник:** `C:/Users/NXTHING/Desktop/Automated_Seasonal_Color_Classification_and_Makeup_Recommendation-...pdf` — Khachatryan, Asryan, Sargsyan (AUA, Spring 2025).

**Что делали:** автоматизированный seasonal color analysis из фото лица → 16 типов (12 основных + 4 переходных) → палитры + рекомендации макияжа из Kaggle-датасета. Инструмент: ImageJ + Java-плагины. Датасет: 200 лиц (FEI Face Database, бразильцы, разное освещение).

**Методы (кратко):**
- Кожа: HSV, пороги Hue 3–24 и S×V 40–80 (для тёмного света S×V 30–96); эксперименты с рекурсивной подстройкой диапазонов (среднее/σ) — отклонены.
- Палитра кожи: 5 hue-групп, среднее по hue 13–24 + +20% brightness.
- Глаза/губы: бинаризация после маски кожи, fill holes, морфология; Method 2 — кластеризация регионов по площади/центроиду; губы RGB (R+B)/2>G, 3 кластера по яркости.
- Волосы: разность масок (hue-only vs hue+SV), средний цвет в HSB.
- Сезон: Munsell-подобные 4 фактора (hue undertone, chroma, value, contrast), нормализация 1–5, таблица Fig.3 → 16 типов; палитры ~28 800 комбинаций hue/chroma/value.

**Результаты:** визуальные пайплайны на 200 изображениях; метрика точности палитры кожи — отклонение среднего hue палитры vs среднего hue извлечённых пикселей; количественной accuracy по сезонам не приведено.

**Выводы:** data-driven альтернатива субъективному color analysis; планы — virtual try-on (AR, MediaPipe/OpenCV), ML на собранных данных, расширение датасета за пределы FEI.

**Артефакт:** колонка «Зачем» в блоке «Что перенести» — по смыслу исходных работ, без привязки к текущему коду; удалён раздел «Уже есть в face-ai».

## 2026-05-21 — best-soltions.md: Colorinsight + общие блоки сравнения

**Запрос:** дополнить `best-soltions.md` на основе PSY222/Colorinsight; расширить таблицу сравнения; перенести сравнение и «Что перенести в face-ai» в конец как общие блоки с указанием источника.

**Артефакт:** добавлен раздел **# 3. Colorinsight** (17 шагов пайплайна, результаты, gaps). Внизу — **Сравнение работ** (Deep Armocromia / AUA / Colorinsight) и **Что можно перенести в face-ai** (20 идей «брать», 10 «не брать», roadmap). Удалены дублирующие локальные блоки из разделов 1–2.

## 2026-05-21 — best-soltions.md: дополнение AUA Capstone

**Запрос:** дополнить `best-soltions.md` шагами пайплайна/проблем/решений из capstone AUA.

**Артефакт:** в `best-soltions.md` добавлен раздел **# 2. AUA Capstone** — схема пайплайна, таблица 24 этапов, результаты/ограничения, нерешённые gaps, сравнение с Deep Armocromia, маппинг «брать / не брать» для face-ai. Документ переименован в общий заголовок по двум работам.

## 2026-05-21 — Разбор репозитория PSY222/Colorinsight

**Запрос:** что это за проект и как работает.

**Суть:** корейский student-проект «Colorinsight» — веб-сервис определения **personal color** (4 сезона: spring/summer/autumn/winter) по фото лица. Backend — FastAPI (`facer/main.py`), frontend — Spring Boot на `:3000` (в репо не выложен).

**Пайплайн:**
1. **FaRL face parsing** (`facer` + `retinaface/mobilenet` + `farl/lapa/448`) — сегментация лица.
2. **Основной путь `/image`:** маска кожи (class 1) → ResNet18 (`best_model_resnet_ALL.pth`, ~60% accuracy) → 4 класса сезона.
3. **Альтернативный `/lip`:** RGB с губ → L2 distance до эталонных палитр → majority vote (старый подход, ~20–30% accuracy).

**Ограничения:** датасет ~750 фото корейских знаменитостей; нет requirements.txt; жёсткая связка с localhost:3000; модель не универсальна по этnicity/освещению.

## 2026-05-21 — Саммари Park et al. IMCOM'18 (3164541.3164612)

**Источник:** `C:/Users/NXTHING/Desktop/3164541.3164612.pdf` — J. Park, H. Kim, S. Ji, E. Hwang, «An Automatic Virtual Makeup Scheme Based on Personal Color Analysis» (IMCOM'18, DOI: 10.1145/3164541.3164612).

**Что делали:** двухэтапная система — (1) автоматический personal color analysis по selfie (радужка, волосы, кожа) → warm/cool + 4 сезона; (2) виртуальный макияж (foundation, blush, lip, eyeshadow, eyeline, eyebrow) по предопределённой expert-базе цветов и стилей с учётом анкеты (возраст, пол, цель).

**Что использовали:** Dlib (68 landmarks), 1D interpolation, Circle Hough Transform (зрачок), K-means (iris), Canny + morphological closing (волосы), RGB→Lab (undertone: a vs b; сезон: контраст L + порог 13), skin detection Eq.(1), Gaussian blur (blush/shadow), template mapping (eyeline/eyebrow). Стек: Matlab 2017a, Python 3.5, OpenCV 2.4, iPhone 7.

**Что получили:** датасет 100 фото (20 человек × 5). Точность извлечения регионов высокая (Dlib); волосы хуже глаз/челюсти. Удовлетворённость макияжем (1–5): foundation/blush/shadow ≥4; губы и eyeline/eyebrow ниже (форма губ, несовпадение кривизны шаблонов).

**Выводы:** personal color + rule-based CV даёт рабочий virtual makeup без ручного подбора цветов; слабые места — hair segmentation и template-based eyeline/eyebrow; релевантно face-ai как early baseline (Dlib + Lab + seasonal rules + makeup masks).

## 2026-05-21 — best-soltions.md: дополнение Park IMCOM'18

**Запрос:** дополнить `best-soltions.md` на основе Park et al. (3164541.3164612.pdf).

**Артефакт:** раздел **# 4. Park et al.** — схема пайплайна (PCA + virtual makeup), таблица 20 этапов, результаты user study, gaps. Обновлены **Сравнение работ** (5-й столбец), **Что перенести** (+8 идей «брать», +5 «не брать»), roadmap try-on (Lab blend, makeup DB, blurred masks).

## 2026-05-21 — Саммари arXiv:2505.14931 «Colors Matter»

**Запрос:** саммари PDF `2505.14931v1.pdf` — что делали, что получили, что использовали, выводы.

**Источник:** R. Alyoubi et al., «Colors Matter: AI-Driven Exploration of Human Feature Colors» (arXiv:2505.14931v1, May 2025).

**Суть:** мультимодальный CV-пайплайн для классификации цвета кожи (8 классов), волос, радужки и undertone (Warm/Cool) по фото лица + запястья. Лучший результат по коже — X-Means + CIEDE2000 + HSV + Gaussian blur → 80% accuracy. Релевантно face-ai как reference по color spaces, Delta E и segmentation stack (Facer, Dlib, RetinaFace).

## 2026-05-21 — Дополнение best-soltions.md (Colors Matter)

**Действие:** добавлен раздел **#5 Colors Matter** в `best-soltions.md` — полная схема пайплайна, таблица «проблема → решение» (28 шагов), результаты, нерешённые проблемы. Обновлены: общая таблица сравнения (5 работ), блок «Брать» (+9 идей), «Не брать» (+7 антипаттернов), roadmap.

## 2026-05-21 — Сравнение 6 моделей face parsing (май 2026)

**Запрос:** сравнительная таблица лучших моделей парсинга лица (кожа, губы, глаза, волосы).

**Источники:** OpenCodePapers (CelebAMask-HQ leaderboard), SegFace AAAI 2025 (arXiv:2412.08647), FaRL CVPR 2022, FaceXFormer ICCV 2025, yakhyo/face-parsing, Facer/pyfacer.

**Вывод:** для максимальной точности — **SegFace** (448–512) или **FaRL/Facer**; для real-time/ONNX в проде — **BiSeNet ResNet34** (уже в face-ai); мультизадача — **FaceXFormer**.

**Артефакт:** `artefacts/face-parsing-models-comparison-2026-05.md` — таблица 6 моделей, описание кандидатов, выводы и рекомендации для face-ai.

## 2026-05-21 — Артефакт color-analysis.md

**Запрос:** сохранить обозреваемые решения (Colors Matter, Park и др.) в `artefacts/color-analysis.md` с кратким описанием и ссылками.

**Артефакт:** `artefacts/color-analysis.md` — обзор 5 работ, сводная таблица, ключевые решения, синтез «брать / не брать», ссылки на `best-soltions.md` и face-parsing comparison.

## 2026-05-21 — Реализация плана Seasonal Color Synthesis (v2.0)

**Задача:** синтез rule-based seasonal color analysis по плану (без unit-тестов).

**Архитектура:**
- `app/pipeline/` — orchestrator, mask_postprocess, ensemble, mask_preview, intake
- `app/backends/parsing/` — **bisenet_resnet34** (ONNX), **farl_b** (pyfacer), **segface** (PyTorch, нужны weights), **landmark_fallback** + registry
- `app/backends/color/` — skin mean_lab / xmeans_hsv_deltae, CIEDE2000, extract
- `app/backends/season/` — Munsell 16-type lookup, Gaussian 12-season, swatch vote
- `app/backends/wrist/` — undertone по венам запястья
- `app/data/` — season_lookup_16.json, reference_swatches.json, palettes_16.json

**API v0.2:**
- `POST /analyze` — face + optional `wrist_file`, `meta_json` (BackendOverrides)
- `GET /health` — статус parsing backends
- Ответ: `metrics.seasonal` (4+12+16, top-k, Munsell), `metrics.mask_preview` (base64)

**Config:** `FACE_AI_PARSING_BACKEND`, `FACE_AI_SEASON_CLASSIFIER=ensemble`, `FACE_AI_SKIN_COLOR_BACKEND`, и др. в `.env.example`

**Зависимости:** core — без изменений; ML parsing — `requirements-parsing-ml.txt` (torch, pyfacer)

**Проверка:** synthetic image → season=spring, parsing=bisenet_resnet34.

## 2026-05-21 — Fix: AttributeError в enhance_parsing

**Проблема:** `parse_face()` возвращает `(ParsingResult, backend, notes)`, а `orchestrator.py` передавал весь tuple в `enhance_parsing()` → `AttributeError: 'tuple' object has no attribute 'skin_mask'`.

**Исправление:** распаковка tuple в `app/pipeline/orchestrator.py`; `parsing_notes` добавляются в `confidence.notes` при fallback.

## 2026-05-21 — Fix: неверный label map для farl_b

**Проблема:** `lapa_logits_to_canonical()` использовал индексы полного датасета LaPa (hair=13, glasses=14, eye=3, lip=5–7). Модель `farl/lapa/448` в pyfacer выдаёт **11 классов**: `background, face, rb, lb, re, le, nose, ulip, imouth, llip, hair`. В debug `20260521_201340_80e42d2f`: hair=0 px, eye_region ≈ lb (бровь), lips ≈ le+nose+ulip.

**Исправление:** `app/backends/parsing/label_maps.py` — skin=1, brow=2–3, eye=4–5, lip=7–9, hair=10; glasses не предусмотрены в lapa/448.

## 2026-05-21 — Fix: farl_b argmax + lip fallback

**Проблема:** после фикса индексов зоны строились через prob threshold 0.5 — мелкие классы (губы, глаза) почти пропадали; «хорошие» губы в debug были артеfact старого маппинга (nose+eye). Overlay не рисовал губы/глаза.

**Исправление:**
- `farl_lapa_label_map_to_canonical()` — маски через argmax (как bisenet/segface)
- `enhance_parsing()` — fallback губ по MediaPipe landmarks, если parsing < 0.015% кадра
- `save_parsing_overlay()` — добавлены слои lip/eye
- `app/services/mask_geometry.py` — общая `lip_mask_from_landmarks()`

## 2026-05-21 — Fix: farl_b — неверная модель и label map (корневая причина)

**Корневая проблема (не та, что в двух предыдущих фиксах):**
1. Backend `farl_b` по документации FaRL/Facer — **FaRL-B на CelebAMask-HQ** → `farl/celebm/448` (89.56 mF1), а не `farl/lapa/448`.
2. `farl_b.py` **всегда** вызывал lapa-маппер, даже при другой модели в конфиге.
3. `celebamask_to_canonical()` (yakhyo layout) **не совместим** с pyfacer `celebm/448` (skin=2, hair=14, eyeg=15, eye=8–9 — см. `facer/face_parsing/farl.py`).
4. При нескольких лицах RetinaFace брался неверный индекс → маски с другого лица.

**Официальные label_names (pyfacer):**
- `lapa/448`: background, face, rb, lb, re, le, nose, ulip, imouth, llip, hair
- `celebm/448`: background, neck, face, cloth, rr, lr, rb, lb, re, le, nose, imouth, llip, ulip, hair, eyeg, hat, earr, neck_l

**Исправление:**
- Default `FACE_AI_FARL_MODEL_NAME=farl/celebm/448`
- `farl_label_map_to_canonical()` — маппинг по official label_names для lapa/celebm
- `farl_b.py` — `label_names` из `faces['seg']`; выбор лица: rect содержит центр landmarks → max RetinaFace score
- Предыдущий фикс lapa-индексов был корректен **только для lapa/448**, но не для backend `farl_b`

**Проверка** (`debug_output/20260521_203453_058ebfbe`, celebm/448): skin ~283k, hair ~32k, lip ~15k, eye ~2k (lapa/448 на том же фото: lip 14 px).

## 2026-05-21 — SegFace: рабочая интеграция (kartik-3004/segface)

**Проблема:** `app/backends/parsing/segface.py` был заглушкой (MobileNetV3 + `segface.pth`), несовместимой с официальными весами SegFace; весов в `models/` не было.

**Решение:**
- Клонирован upstream: `vendor/segface` (https://github.com/kartik-3004/segface)
- Backend переписан: `network.get_model()` + checkpoint `state_dict_backbone` из HuggingFace
- Скрипт: `scripts/download_segface_weights.py` → `models/segface/<model>_celeba_<size>/model_299.pt`
- Default: `mobilenet` @ 512 (быстрый вариант из paper)
- Label map: `SEGFACE_CELEB_LABEL_NAMES` / `SEGFACE_CELEB_ZONE_SPEC` (19 классов CelebAMask-HQ)
- Патч `vendor/segface/network/models/segface_celeb.py`: `weights=None` при init (без скачивания ImageNet; веса из checkpoint)
- Конфиг: `FACE_AI_SEGFACE_MODEL`, `FACE_AI_SEGFACE_BACKBONE` (default `segface_celeb`)

**Запуск:** `pip install -r requirements-parsing-ml.txt` → `python scripts/download_segface_weights.py` → `FACE_AI_PARSING_BACKEND=segface`

**Проверка:** `parsing_health()` → `segface: true`; inference на sample image OK.

## 2026-05-21 — SegFace: 404 при скачивании swin_base

**Симптом:** `python scripts/download_segface_weights.py --model swin_base` → `404` на  
`.../swin_base_celeba_512/model_299.pt` (иногда предшествует SSL `UNEXPECTED_EOF_WHILE_READING`).

**Причина:** на HuggingFace папки укорочены (`swinb_celeba_512`, `swinv2b_celeba_512`, …), а скрипт собирал путь из внутреннего имени `--model swin_base`.

**Исправление:**
- `app/segface_weights.py` — маппинг `swin_base` → `swinb`, `swinv2_base` → `swinv2b`, `convnext_base` → `convnext`, …
- `scripts/download_segface_weights.py` — скачивание по HF-пути, установка в `models/segface/swin_base_celeba_512/model_299.pt`
- `segface.py` — fallback: искать веса и в HF-папке (`swinb_celeba_512`), и в локальной (`swin_base_celeba_512`)

**Запуск:** `.venv\Scripts\python.exe scripts/download_segface_weights.py --model swin_base`  
При обрыве SSL — повторить команду или скачать вручную с [HuggingFace SegFace](https://huggingface.co/kartiknarayan/SegFace/tree/main/swinb_celeba_512).

## 2026-05-21 — SegFace: неполная маска кожи в debug

**Симптом:** `04_mask_skin.png` обрезана (нет лба/верхних щёк), при этом остальные зоны SegFace выглядят корректно.

**Причины:**
1. `SEGFACE_CELEB_ZONE_SPEC` брал только `skin` + `nose`; в CelebAMask-HQ щёки/подбородок часто в классе `neck` (index 1).
2. `enhance_parsing()` пересекал маску с `cheek_jaw_skin_mask()` — convex hull только по точкам челюсти (0–16), без лба → обрезка верхней части лица.

**Исправление:**
- Зона кожи: `skin`, `nose`, `neck`, `l_ear`, `r_ear`
- Для `segface` / `farl_b` не применять `cheek_jaw` (модель уже отделяет фон/волосы)
- Debug: `04_mask_skin_parsing.png` — маска до постобработки; `04_mask_skin.png` — после

## 2026-05-21 — BiSeNet: «сломана» финальная маска кожи

**Симптом:** `04_mask_skin_parsing.png` нормальная, `04_mask_skin.png` — остался узкий фрагмент (треугольник).

**Причина:** `cheek_jaw_skin_mask()` и повторное вычитание brow/lip в `enhance_parsing()` применялись и к **bisenet** (и segface/farl до фикса). Для yakhyo класс `skin`=1 уже без глаз/бров/губ; hull по челюсти срезал лоб, а маски бров/губ съедали щёки.

**Исправление:**
- Геометрическая обрезка (`cheek_jaw`, subtract brow/lip) — только для `landmark_fallback`
- `yakhyo_to_canonical`: кожа = классы `1` (skin) + `10` (nose) + `14` (neck)

## 2026-05-21 — Аудит best-soltions.md vs face-ai

**Запрос:** какие наработки из `best-soltions.md` уже применены, какие — нет.

**Итог:** ~60% блока «Брать» внедрено; краткосрочный roadmap ~70%; среднесрочный ~40%; долгосрочный ~10%. Основные пробелы: DL-классификатор сезона, бенчмарк Deep Armocromia, virtual makeup/product match, AUA-эвристики (lip clusters, hue trim, chroma eyes+lips).

## 2026-05-21 — AUA fine-heuristics (губы, hue-trim, chroma)

**Запрос:** внедрить три эвристики из AUA Capstone.

**Сделано:**
- `app/backends/color/aua_heuristics.py` — hue-trim кожи (HSV 13–24), lip 3-tier brightness + фильтр R/B, `aggregate_chroma(skin, eyes, lips)`
- `skin.py`, `color_contrast.py` — hue-trim для skin LAB по умолчанию
- `lip_color_backend=brightness_clusters` по умолчанию (`config.py`)
- `munsell_lookup.py` — ось chroma из skin+eyes+lips (AUA)
- `extract.py` — передаёт `skin_ab`, `iris_ab`, `lip_ab` в Munsell
- `tests/test_aua_heuristics.py`

## 2026-05-21 — Бенчмарк seasonal analysis на Deep Armocromia RGB

**Запрос:** код для теста анализа цветотипа (не модели) на `dataset/`.

**Сделано:**
- `app/eval/deep_armocromia.py` — разбор датасета `dataset/RGB/RGB/{train,test}/<season>/<subtype>/`, маппинг итальянских меток → 4/12 сезонов face-ai
- `scripts/eval_season_dataset.py` — прогон полного пайплайна `analyze_bgr()`; метрики top-1/top-2 (4 сезона), top-1/top-2/top-3 (12), confusion matrix, mistakes JSON
- `tests/test_season_benchmark.py` — smoke-тест (skip если датасета нет)

**Запуск:** `.venv\Scripts\python.exe scripts/eval_season_dataset.py --split test` (912 img); быстрый прогон: `--limit 50`. Результаты: `artefacts/eval/deep_armocromia_latest.json`.

**Smoke (5 img):** 4-season top-1 40%, top-2 80%; parsing_used 100%.

## 2026-05-21 — Park IMCOM'18 classifier (`park_imcom18`)

**Запрос:** имплементация формул Park et al. с переключением.

**Сделано:**
- `app/backends/season/park_imcom18.py` — undertone skin a* vs b*; 4 сезона по mean |ΔL| (skin/hair/iris) vs порог 13 (CIELAB L).
- `FACE_AI_SEASON_CLASSIFIER=park_imcom18` (+ `FACE_AI_PARK_CONTRAST_THRESHOLD`, `FACE_AI_PARK_USE_CIELAB_L_SCALE`).
- API override: `backends.season_classifier: "park_imcom18"`.
- `tests/test_park_imcom18.py` — unit-тесты формул.

## 2026-05-21 — Park в ensemble

**Запрос:** включить `park_imcom18` в ensemble (не только standalone).

**Изменения:**
- `fuse_seasonal(..., park=...)` — веса: Munsell 0.70, Park 0.15, swatch 0.10, wrist 0.05.
- `extract.py` — при `season_classifier=ensemble` всегда считается Park и передаётся в fusion.
- `tests/test_ensemble_park.py` — smoke fusion с Park.
- Standalone `park_imcom18` сохранён для A/B и бенчмарка.

## 2026-05-21 — Сравнение face-ai vs Park IMCOM'18 (код определения цветотипа)

**Запрос:** сопоставить реализацию face-ai с Park et al. (3164541.3164612.pdf) по определению цветотипа; выделить отличия, влияющие на результат.

**Ключевые расхождения с impact на label:**
- Undertone: статья — только кожа, правило `a* > b*` → cool; face-ai — fused skin+hair+iris с порогами b*/a* и neutral.
- Сезон: статья — 4 сезона, порог контраста L≈13 между тремя зонами; face-ai — VCI 0–100, 12/16 типов, ensemble (Munsell+swatch+wrist; Gaussian временно отключён).

## 2026-05-21 — Временное отключение gaussian_twelve

**Запрос:** убрать Gaussian-метод из пайплайна на время.

**Изменения:**
- `app/backends/color/extract.py` — не вызывается `classify_gaussian_twelve`; ветка `gaussian_twelve` удалена из `season_classifier`.
- `app/pipeline/ensemble.py` — ensemble только Munsell (0.85) + swatch (0.10) + wrist (0.05).
- `app/backends/season/season_maps.py` — общие маппинги 16→12→4 (раньше жили в `gaussian_twelve.py`).
- `app/config.py`, `app/schemas/analysis.py` — `gaussian_twelve` убран из допустимых значений.
- `app/backends/season/gaussian_twelve.py` — файл сохранён, не подключён к extract (для будущего возврата).
- Регионы: статья — CHT+K-means (iris), Canny (hair), jaw/cheek hull (skin); face-ai — BiSeNet/FaRL parsing, hue-trim skin, percentile iris ring.
- Доп. сигналы в face-ai: брови, губы, chroma, depth, очки — отсутствуют в статье.

## 2026-05-21 — AUA Capstone шкалы Munsell (OpenCV LAB → 1–5)

**Запрос:** перекалибровать `score_to_bin` под пороги capstone (chroma/value 0–1, contrast 0–21) с явным маппингом из OpenCV LAB.

**Изменения:**
- `app/backends/season/aua_scales.py` — конвертеры `lab_chroma_to_aua`, `opencv_L_to_aua_value`, `lab_luminance_contrast_aua` и `normalize_*_aua` с порогами из Java capstone (0.2/0.4/0.6/0.8 и 5/9/13/17).
- `app/backends/season/munsell_lookup.py` — `compute_munsell_axes` использует AUA-шкалы вместо линейного `score_to_bin` (8–28, 45–78, 1.05–1.55).
- Контраст для Munsell: `(L_max+0.05)/(L_min+0.05)` вместо обратного отношения `(L_min+0.05)/(L_max+0.05)`.
- `tests/test_munsell_aua_scales.py` — unit-тесты порогов и типичных L/chroma.

## 2026-05-21 — Палитра цветов анализа в API (Park-style)

**Запрос:** отдавать в `/analyze` палитры цветов, как в статье Park et al. — цвета, реально используемые при классификации.

**Изменения:**
- `app/backends/color/analysis_palette.py` — сбор RGB/hex swatches по зонам лица (skin hue-trim, hair, brow, iris, lip) + 3 reference RGB якоря для определённого 4-сезона (`reference_swatches.json`, swatch vote).
- `metrics.analysis_palette` в ответе API: `{ face: [...], season_reference: [...] }`.
- Схема: `PaletteSwatch`, `AnalysisPalette` в `app/schemas/analysis.py`; обновлён `analysis_contract.json`.
- `tests/test_analysis_palette.py` — unit-тесты конвертации RGB/LAB и season anchors.

## 2026-05-21 — Простой UI для POST /analyse

**Запрос:** простая страница для вызова POST `/analyse`.

**Изменения:**
- `static/index.html` — форма: фото лица, опционально запястье и `meta_json`, вывод JSON-ответа.
- `app/main.py` — `GET /` отдаёт страницу; алиас `POST /analyse` → тот же handler, что и `/analyze`.
- UI: swatch-карточки палитры лица и season reference, сводка сезона/подтона, маски parsing, рекомендации; JSON в `<details>`.
- API: поле `debug` в ответе (`run_id` + список артефактов), `GET /debug/{run_id}/{filename}` для отдачи файлов из `debug_output/`.
- UI: превью загруженных фото + галерея всех debug-файлов (картинки и текст/json).
- UI: палитры сразу под входными фото.

## 2026-05-21 — Описание алгоритма цветотипа (по коду)

**Запрос:** полный путь алгоритма анализа цветотипа, формулы и факторы влияния (только код).

**Результат:** задокументирован пайплайн `analyze_bgr` → parsing → LAB-признаки → Munsell/Park/Swatch/Ensemble; промежуточный Gaussian-12 в `color_contrast.py` перезаписывается в `extract.py`; `gaussian_twelve.py` не подключён.

## 2026-05-21 — Улучшение семплинга палитры лица

**Запрос:** исправить палитру для фото со светлыми волосами и прищуренными глазами (ошибочно тёмные глаза → завышенный контраст → осень вместо весны).

**Изменения:**
- `app/backends/color/aua_heuristics.py` — `skin_trim_pixels` (отсечение бликов по L), `hair_lab_luminance_trim` (55–88 pct яркости, без overlap с кожей), `brow_lab_from_mask` (median, без overlap с волосами).
- `app/services/color_contrast.py` — EAR-детект прищура; quality gate для iris (min L, не темнее волос); более узкое iris-кольцо при высокой доле тёмных пикселей; hair/brow через новые хелперы; убран fallback `_iris_approx_ab`.
- `app/backends/color/analysis_palette.py` — палитра использует те же hair/brow/iris правила.
- `app/pipeline/mask_postprocess.py` — cheek/jaw hull только для non-DL parsing; brow/lip всегда вычитаются из skin mask.

## 2026-05-21 — Фикс тёмной кожи в палитре (регрессия cheek/jaw)

**Проблема:** после первого фикса кожа `#9D5E59` (L* 46) — цвет бровей/теней; сезон «истинная осень» вместо весны.

**Причина:** cheek/jaw hull на DL parsing оставлял нижнюю зону с тенями; hue-trim без shadow-cut брал красноватые freckle/shadow пиксели.

**Изменения:**
- Откат cheek/jaw для DL parsers (`bisenet`/`farl`/`segface`).
- `skin_trim_pixels` — mid-L band (18–88 pct): без бликов и глубоких теней.
- Brow/lip всегда вычитаются из skin mask (не только geometry mode).

## 2026-05-21 — Undertone: fair cool skin + hair anchor

**Проблема:** Light Summer (платина, голубые глаза) → `light_spring` из‑за warm undertone=5 при a*≈b*.

**Изменения:**
- `app/backends/color/undertone.py` — `skin_undertone_from_ab` (Park a*>b* + fair pink + borderline cool), `infer_face_undertone` (hair ash → cool, iris makeup guard), `iris_undertone_untrusted`.
- `park_imcom18.py` — Park undertone через общий модуль.
- `color_contrast.py` — новый undertone fusion; iris не в chroma при makeup guard; low_contrast → lean cool.
- `extract.py` — Park/swatch используют hue-trim skin ab/rgb (не raw mean маски).

## 2026-05-21 — Iris sampling v2 (sclera / skin bleed)

**Проблема:** светлые голубо-серые глаза семплировались как `#CD9594` ≈ кожа `#C6928F`.

**Изменения:**
- Inner disk 74% iris hull — меньше краёв/склеры.
- Light-eye luminance bands (28–72 pct) для med gray ≥ 118.
- `_iris_exclude_sclera_skin_pixels` — розовая склера + пиксели близко к skin LAB.
- `_median_lab_iris_distinct` — median по 55% пикселей, наиболее удалённых от кожи.
- Quality gate: reject iris если Δab < 8 и ΔL < 14 vs skin.
- `iris_matches_skin` в undertone guard + debug overlay с skin_lab.

## 2026-05-21 — Откат iris sampling к исходному алгоритму

**Причина:** эксперименты (squint/EAR, quality gate, sclera/skin filter) ломали определение цвета глаз.

**Изменения:** восстановлен оригинальный pipeline — luminance ring 20–82, median LAB, `_resolve_iris_lab` без фильтров, `_iris_approx_ab` fallback. Undertone/skin/hair правки сохранены.

## 2026-05-22 — Консультация: иерархический DL-классификатор на Deep Armocromia

**Запрос:** как организовать дообучение (4 сезона → 3 подтипа), какую модель, оценка времени на RTX 5070 12 GB.

**Контекст датасета:** `dataset/RGB/RGB` — train 4008 / test 912; 4×3=12 классов; разметка уже в `app/eval/deep_armocromia.py`. Бенчмарк статьи: 4-class ~55%, 12-class ~32% (FaRL frozen + 2 FC).

**Рекомендация:** каскад shared backbone + head_4 + 4× head_subtype(3); backbone FaRL-16/64 или EfficientNet-B2; препроцессинг — face crop + skin mask (SegFace, как в пайплайне). MVP обучение ~2–4 ч, полный цикл с unfreeze ~6–12 ч на 12 GB.

**Уточнение (цель — max accuracy, не baseline статьи):** paper 55/32% — нижняя планка, не target. Стратегия «как можно лучше»: multi-region input (skin/hair/iris), full fine-tune FaRL-64, 5-fold CV + TTA + fusion с rules (munsell/park) и flat-12 head. Реалистичный потолок на этом датасете ~65–72% (4-class) / ~42–50% (12-class); выше — только с доп. данными или human-in-the-loop.

## 2026-05-22 — Скрипт препроцессинга Deep Armocromia для DL

**Запрос:** скрипт препроцессинга фото (одна masked face crop на sample).

**Артефакты:**
- `app/eval/season_preprocess.py` — landmarks → parsing → masked face crop 224×224
- `scripts/cache_season_dataset.py` — batch-кэш в `artefacts/cache/deep_armocromia/{split}/{id}/rgb.npy` + `meta.json` + `manifest.json`
- `tests/test_season_preprocess.py` — unit-тесты helper'ов

**Запуск:** `.venv\Scripts\python.exe scripts/cache_season_dataset.py --split all --preview` (~4920 img, SegFace ~2–4 ч).

**Обновление:** режим `face` включает губы (signature `face_lips_*`); DL-кэш всегда без skin blur (`apply_skin_blur=False`, signature `*_noblur_*`).

## 2026-05-22 — Оценка качества DL-сэмпла (11726.jpg → preview)

**Запрос:** нормальный ли препроцессированный сэмпл для обучения модели цветотипа.

**Пример:** `test/autunno/deep/11726.jpg` → `artefacts/cache/deep_armocromia/test/5c3fb3955c52e3b9/preview.jpg` (224×224, mask `face_lips`, SegFace, bg=128).

**Вывод:** технически валидный (face + parsing ok), но **не эталонный** — bbox 532×784 сжимается в квадрат без сохранения aspect ratio (лицо «расплющено» по ширине ~1.5×), блики от вспышки на лбу/носу, celebrity-фото с возможной цветокоррекцией. Для DL приемлем как часть датасета Deep Armocromia (тот же пайплайн для всех), но для hard-negative / quality filter — кандидат на пониженный вес или исключение.

## 2026-05-23 — Запуск face-ai-mobile (Expo)

**Запрос:** как запустить мобильное приложение `face-ai-mobile`.

**Стек:** Expo SDK 54, React Native 0.81, TypeScript. Backend — FastAPI `face-ai` на `POST /analyze`.

**Шаги:**
1. Backend: из `face-ai` — `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
2. В `face-ai-mobile/services/analyzeConfig.ts` указать LAN IP ПК (`ANALYZE_UPLOAD_URL`), при необходимости добавить IP в `app.json` → `NSExceptionDomains` (iOS)
3. Mobile: `npm install` → `npm start` (Expo Go на телефоне в той же Wi‑Fi сети, или `npm run android` / `npm run ios` / `npm run web`)

**Выполнено (2026-05-23):** IP обновлён на `192.168.3.20` в `analyzeConfig.ts` и `app.json`; `npm install` ok; backend и Expo запущены локально.

## 2026-05-23 — PRD: Hierarchical Deep Armocromia Season Classifier

**Запрос:** `/to-prd` — синтез PRD из контекста проекта (skill `to-prd`).

**Scope:** следующий крупный gap по roadmap — DL-классификатор сезона на датасете Deep Armocromia с интеграцией в ensemble `/analyze`. Препроцессинг и eval rules уже есть; training code и runtime backend — нет.

**Артефакты:**
- `season-dl/artefacts/prd/deep-armocromia-dl-classifier.md` — полный PRD (Problem, Solution, 40 user stories, Implementation/Testing decisions, Out of Scope)
- `season-dl/scripts/publish_prd_issue.ps1` — публикация issue в GitHub с label `ready-for-agent`

**Deep modules (PRD):** SeasonDataset, HierarchicalSeasonModel, SeasonTrainer, DeepArmocromiaClassifier, PreprocessLetterbox, SeasonModelEval.

**Targets (test 912):** 4-season top-1 ≥58%, 12-season top-1 ≥38%, fused ensemble > max(rules, DL).

**Публикация в issue tracker:** GitHub CLI установлен (`gh 2.92.0`), но `gh auth login` не выполнен — issue не создан автоматически. Команда после авторизации:
```powershell
cd face-ai
.\season-dl\scripts\publish_prd_issue.ps1
```

## 2026-05-23 — Выделение DL scope в `season-dl/`

**Запрос:** перенести всё, что касается DL, в отдельную директорию — другой scope.

**Перенесено в `season-dl/`:**
- `season_dl/preprocess.py` (бывш. `app/eval/season_preprocess.py`)
- `scripts/cache_dataset.py` (бывш. `scripts/cache_season_dataset.py`)
- `tests/test_preprocess.py` (бывш. `tests/test_season_preprocess.py`)
- `artefacts/cache/deep_armocromia/` — кэш crops
- `artefacts/prd/deep-armocromia-dl-classifier.md` — PRD
- `scripts/publish_prd_issue.ps1`

**Осталось в корне face-ai (rule-based scope):**
- `app/eval/deep_armocromia.py` — loader датасета (общий для rules benchmark и DL cache)
- `scripts/eval_season_dataset.py` — benchmark rule-based пайплайна
- `tests/test_season_benchmark.py`

**Запуск cache:** `python season-dl/scripts/cache_dataset.py --split all --preview`

## 2026-05-23 — PRD всего проекта face-ai

**Запрос:** `/to-prd` — существующий PRD в `season-dl/` описывал вещи уровня всего проекта; нужен project-wide PRD.

**Артефакты:**
- `artefacts/prd/face-ai-product.md` — PRD платформы: Problem/Solution, 65 user stories, 18 deep modules, API/mobile/eval/DL roadmap, testing/out of scope
- `scripts/publish_prd_issue.ps1` — публикация project PRD в GitHub Issues (`ready-for-agent`)
- `season-dl/artefacts/prd/deep-armocromia-dl-classifier.md` — уточнён scope: только DL subproject, ссылка на project PRD

**Deep modules (project PRD):** AnalysisOrchestrator, ParsingRegistry, ColorFeatureExtractor, SeasonClassifierEnsemble, GeometryAnalyzer, RecommendationEngine, AnalysisContract, DeepArmocromiaEval, MobileAnalysisClient + DL hooks.

**Публикация:** `.\scripts\publish_prd_issue.ps1` (требует `gh auth login`).

## 2026-05-23 — Brainstorming: дополнительные фичи продукта

**Запрос:** идеи фич сверх AR try-on, product match из БД, оценка образа по колористике; что ещё добавить и как реализовать быстро при ограниченном времени.

**Контекст:** PRD v1 держит try-on и product match в out of scope; ~70% краткосрочного roadmap уже в коде (mask preview, ΔE, palette, lip clusters, wrist API). Мобильный клиент не показывает top-k, palette, mask preview; `/contour/final` отсутствует на backend.

**Следующий шаг:** уточнить горизонт времени и целевую аудиторию (демо vs production) перед приоритизацией и design spec.

**Уточнение (2026-05-23):** цель — **жёсткий wow-эффект на демо** (не production). Приоритет: визуальный impact > точность > масштабируемость.

**Grill-with-docs (2026-05-23):** зафиксирован roadmap priority — (1) Photo try-on, (2) Outfit scanner, (3) Live AR try-on. Создан `CONTEXT.md` с доменными терминами.

**Photo try-on v1:** zone A+B+C (губы/румяna, тени/брови, волосы) — один слайдер, ship-all-measure-cut.

**Try-on architecture:** Dual pipeline (CV + Generative, toggle Classic/AI); путь к zone-split hybrid.

**Analysis architecture:** Mode B (CV pixels always; LLM primary for semantics; rules fallback if LLM off/down).

**Product catalog:** curated `products.json` для demo, ΔE match.

**Generative backend:** Model API adapter (URL-configurable); external/self-hosted — один interface.

**Outfit scanner:** Flow D (отдельный scan + optional inline на первом фото).

**Demo navigation:** Linear wizard A.

**Try-on model (2026-05-23):** три category (makeup, glasses, hairstyle) на Photo try-on; Live AR = future, same TryOnEngine on camera.

## 2026-05-23 — Design spec: Try-On Platform

**Grill-with-docs завершён; design approved.**

**Spec:** `docs/superpowers/specs/2026-05-23-try-on-platform-design.md`

**Содержание:** TryOnEngine (photo→live), LLM primary + rules fallback, dual pipeline CV/Generative, products.json, outfit scanner, demo wizard, API endpoints, implementation order.

**Следующий шаг:** review spec → implementation plan.

## 2026-05-23 — Implementation plan

**Plan:** `docs/superpowers/plans/2026-05-23-try-on-platform-plan.md`

6 vertical slices (P0 foundation → P1 photo try-on + LLM + mobile → P2 outfit → P3 live stub). Demo acceptance criteria defined.

## 2026-05-23 — P0 implementation started

**Done:**
- `app/config.py` — LLM, generative, try-on, products paths; `llm_available` / `generative_available`
- `app/schemas/products.py`, `try_on.py`, `outfit.py`
- `app/data/products.json` (30 SKU), `makeup_db.json` (12 seasons)
- `app/services/product_catalog.py`, `product_matcher.py`
- `GET /products/match`, `/health` products + LLM flags
- `static/overlays/` placeholder PNGs (10)
- `tests/test_product_matcher.py`

**Next:** P1 slice 1 — LLM adapter + `/analyze` merge.

## 2026-05-23 — Debug UI update (static/index.html)

Wizard-style debug UI: /health pills, product match cards, try-on section (Classic/AI + slider), outfit scanner; graceful 404 for unreleased endpoints.

## 2026-05-23 — Fix: API hangs / requests timeout

**Cause:** `/health` called `parsing_health()` (torch/SegFace); `/analyze` ran sync in async handler → blocked event loop. Multiple uvicorn on :8000; default bind 127.0.0.1 blocked mobile (192.168.x).

**Fix:** fast `/health`; `/health/full` for parsing; `run_in_threadpool` for analyze; `scripts/run_dev.ps1` with `--host 0.0.0.0`.

## 2026-05-23 — P1 slice 1: LLM adapter + orchestrator merge

**Done:**
- `app/backends/llm/adapter.py` — OpenAI-compatible multimodal client, retry 1×, JPEG base64
- `app/backends/llm/schema_merge.py` — JSON parse/validate, merge seasonal labels + recommendations
- `app/backends/llm/prompts/analysis_system.txt` — Armocromia system prompt (RU recommendations)
- `app/pipeline/orchestrator.py` — после CV/rules вызывает LLM если `llm_available`; silent fallback
- `app/data/try_on_prompts/generative_makeup.txt`, `generative_hairstyle.txt` — шаблоны для generative (slice 3)
- `tests/test_llm_merge.py` — 5 passed (`.venv`)

**Merge rules:** `metrics.seasonal.*` от LLM; geometry/masks/palette — CV; `recommendations[]` всегда из `build_recommendations` (rules + `palettes_16.json`), после LLM пересборка с сезоном LLM; `seasonal_method=llm` при успехе.

**Fix (2026-05-23):** рекомендации больше не берутся из LLM JSON — только rule-based, как при fallback.

**Logging (2026-05-23):** INFO в консоль uvicorn — raw JSON от LLM, parsed seasonal, CV→LLM merge в orchestrator. Fix: явный `StreamHandler` на `app.backends.llm` / `app.pipeline.orchestrator` (setLevel без handler не выводил в uvicorn).

**Next:** P1 slice 2 — CV Photo try-on (`MakeupRenderer`, `GlassesRenderer`, `HairstyleRenderer`, `TryOnEngine`, `POST /try-on/photo`).

## 2026-06-01 — Generative Photo Try-On (makeup / glasses / hairstyle)

**Done:**
- `docs/generative-api-contract.md` — OpenAI `/v1/images/edits` + `custom_json` inpaint
- `app/config.py` — `generative_transport`, `generative_model`, `generative_timeout_s`, `generative_strength`
- `app/backends/try_on/generative_api.py` — `GenerativeModelAPI` (multipart + JSON), mask RGBA для OpenAI
- `app/backends/try_on/mask_builder.py`, `prompt_builder.py`, `generative_glasses.txt`
- `app/backends/try_on/engine.py` — `TryOnEngine`, порядок hairstyle → makeup → glasses
- `app/pipeline/face_prepare.py` — landmarks + parsing для try-on
- `app/services/try_on_service.py`, `POST /try-on/photo` в `app/main.py` (threadpool)
- `scripts/mock_generative_server.py` — mock `/v1/images/edits` и `/inpaint`
- `tests/test_generative_api.py`, `test_mask_builder.py`, `test_try_on_engine.py`, `test_try_on_endpoint.py`
- `static/index.html` — default AI mode, Classic скрыт при generative on

**CV branch:** `cv: null` до реализации `MakeupRenderer` / overlay renderers.

**Env:** `FACE_AI_GENERATIVE_API_URL` + `FACE_AI_GENERATIVE_TRANSPORT=openai_images_edit|custom_json`

## 2026-06-01 — Fix: generative 401 (no API key)

**Cause:** `FACE_AI_GENERATIVE_API_KEY` не задан, при этом LLM ключ есть (`vsellm` и т.п.).

**Fix:** `settings.resolved_generative_api_key` — fallback на `FACE_AI_LLM_API_KEY`; явное предупреждение в лог, если оба пусты.

## 2026-06-01 — Try-on: переключатель use_mask

**Done:** `meta_json.use_mask` + `FACE_AI_GENERATIVE_USE_MASK` (default true). Без маски — только `image` + `prompt` в `images/edits`; промпт дополняется зонами. UI: «С маской / Без маски». В ответе `categories.*.masked`.

## 2026-06-01 — Try-on: preservation constraints

**Prompts:** `generative_preservation.txt` — aspect ratio, identity, head pose, background; общий negative.  
**API:** `size=auto` (`FACE_AI_GENERATIVE_IMAGE_SIZE`); выход ресайзится к размеру входа, если модель вернула другие размеры.

## 2026-06-01 — find-skills: промпты для image edit (Gemini 3 Pro)

**Запрос:** `google/gemini-3-pro-image-preview` перерисовывает всё изображение вместо локального редактирования.

**CLI:** `npx skills find` — после `npm cache clean --force` работает; поиск: `gemini image edit`, `image edit prompt`, `nano banana`.

**Рекомендованные skills (по релевантности):**
- `google-gemini/gemini-skills@gemini-api-dev` — официальный API, image edit vs generate
- `resciencelab/opc-skills@nanobanana` — Gemini 3 Pro image editing
- `jezweb/claude-skills@image-gen` — addition/removal, Gemini native editing
- `agentspace-so/runcomfy-agent-skills@image-edit` — mask-driven region replacement, identity preservation
- `agentspace-so/runcomfy-agent-skills@nano-banana-edit` — edit-режим Nano Banana

## 2026-06-01 — skills CLI: ERR_MODULE_NOT_FOUND (simple-git.mjs)

**Симптом:** `npx skills add ...` → `Cannot find module .../dist/_chunks/libs/simple-git.mjs` (битый кэш npx, старая/неполная распаковка).

**Fix:** `npm cache clean --force`, удалить `%LOCALAPPDATA%\npm-cache\_npx\ac0ed6aa23b37c1e` (если осталась), затем `npx --yes skills@latest add google-gemini/gemini-skills@gemini-api-dev -y` (или `npm install -g skills@latest` и `skills add ...`). Установлено в `.agents/skills/gemini-api-dev`.

**Контекст проекта:** уже есть `generative_preservation.txt`, маска OpenAI RGBA, `FACE_AI_GENERATIVE_USE_MASK`; проблема может быть в прокси/OpenAI-совместимости, а не только в тексте промпта.

## 2026-06-01 — Fix: Gemini makeup preservation (local mask composite)

**Запрос:** `google/gemini-3-pro-image-preview` перерисовывает всё фото вместо локального макияжа; человек, поза и кадр не должны меняться.

**Причина:** OpenAI `/images/edits` через прокси (vsellm) + `USE_MASK=false` — модель получает только prompt без локального ограничения зон; Gemini semantic edit всё равно может менять весь кадр.

**Fix:**
- `composite_masked()` в `generative_api.py` — после ответа модели смешиваем результат с оригиналом по parsing-маске (губы/щёки/веки/брови); вне маски — 100% исходные пиксели.
- `FACE_AI_GENERATIVE_COMPOSITE_MASK=true` (default); для makeup маска применяется локально даже при `USE_MASK=false`.
- Промпты Gemini-style: «Using the provided portrait… change ONLY… keep pose/identity unchanged» (`generative_makeup.txt`, `generative_preservation.txt`).
- Новый transport `gemini_native` — прямой вызов `google-genai` `generateContent` для `gemini-3-pro-image-preview` (semantic edit вместо OpenAI edits).
- `.env`: `USE_MASK=true`, `COMPOSITE_MASK=true`.

**Файлы:** `generative_api.py`, `renderers/category.py`, `prompt_builder.py`, `config.py`, `generative_makeup.txt`, `generative_preservation.txt`, `docs/generative-api-contract.md`, `requirements.txt` (+`google-genai`).

## 2026-06-01 — Fix: composite artifact on Gemini makeup (polygon face)

**Симптом:** «полигон» на лице, дубли глаз/губ — оригинал по краям, внутри маски чужое перегенерированное лицо.

**Что реально отправлялось:**
- Parsing-маска макияжа = union зон `lips`, `brows`, `blush` (щёки), `shadow` (веко = eye_region минус зрачок), с feather σ=2.5.
- В API (vsellm `/images/edits`): PNG RGBA, **прозрачные** пиксели = зона редактирования (OpenAI-формат).
- Локально: `composite_masked()` вставлял **полный ответ модели** внутрь этой маски; Gemini перерисовал всё лицо → шов и дубли.

**Fix:**
- `resolve_mask_policy()` — для Gemini image + `openai_images_edit`: **не** слать маску в API и **не** делать local composite.
- Default `FACE_AI_GENERATIVE_COMPOSITE_MASK=false`.
- Debug: `debug_output/*/tryon_makeup_mask.png`, `_mask_overlay.jpg`, `_model_raw.jpg`, `_mask_meta.txt`.
- `.env`: `USE_MASK=false`, `COMPOSITE_MASK=false`.

## 2026-06-01 — Fix: Gemini before/after diptych + ignore input reference

**Симптом:** модель отдаёт side-by-side «до/после», другая одежда/фон, цвета не из `makeup_db`.

**Причина:** Gemini image для beauty часто рисует comparison collage; промпт «try-on» без запрета панелей; прокси не держит пиксельный референс как inpaint.

**Fix:**
- Промпты: «attached photograph in place», «exactly ONE full-frame», запрет before/after/split-screen.
- `unwrap_diptych_if_present()` — если ширина ~2× входа, берём правую панель (after).
- `gemini_native`: `response_modalities=["IMAGE"]`, `ImageConfig(aspect_ratio=…)` по входу.
- Палитра: «Use only these target colors (subtle)».

## 2026-06-01 — Остановка процессов на порту 8000

Завершены процессы, занимавшие `127.0.0.1:8000`: PID **16144** (LISTENING, сервер) и PID **3264** (клиентские соединения). Порт освобождён.

## 2026-06-01 — Fix: generative HTTP 400 Unknown parameter response_format

**Симптом:** vsellm / `gpt-image-2` → `Unknown parameter: 'response_format'`.

**Fix:** `resolve_openai_response_format()` — `auto` шлёт `b64_json` только для `dall-e-*`, для `gpt-image-*` поле не передаётся; retry без поля при 400; парсинг ответа по `url` если нет `b64_json`. Env: `FACE_AI_GENERATIVE_RESPONSE_FORMAT=auto|b64_json|omit`.
