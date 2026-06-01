# Пайплайны seasonal color analysis — проблемы и решения

Сводка по работам: что делали, какие проблемы встречали, чем решали.  
Работы: Deep Armocromia · AUA Capstone · Colorinsight · Park IMCOM'18 · Colors Matter (arXiv:2505.14931).  
Общее сравнение и маппинг в `face-ai` — в конце документа.

---

# 1. Deep Armocromia (DL + датасет)

Источник: Stacchio et al., *Deep Armocromia: A Novel Dataset for Face Seasonal Color Analysis and Classification* (`Armocromy_ECCV1.pdf`).

---

## Общая схема пайплайна

```
Сбор фото → Обучение аннотаторов → Разметка (4 сезона + 12 sub-types)
    → Очистка (dedup) → Face parsing (Facer) → Маски кожи/волос/глаз
    → Train/Test split (80/20) → Fine-tuning DL (FaRL / ResNeXt50)
    → Классификация: 4 сезона ИЛИ 12 sub-types → Метрики + confusion matrix
```

---

## Шаги пайплайна: проблема → решение

| # | Этап | Проблема | Как решено в работе |
|---|------|----------|---------------------|
| 1 | **Постановка задачи** | В CV нет открытых датасетов для Armocromia / seasonal color analysis по лицу; единственная близкая работа (Su et al., 2024) — ~1000 web-фото без публикации, без sub-types и без строгого протокола | Введён первый публичный датасет **Deep Armocromia** (~4920 фото) с разметкой по **Flow Theory** (Migliaccio): 4 сезона + 12 sub-types. Датасет: [github.com/lorenzo-stacchio/Deep-Armocromia](https://github.com/lorenzo-stacchio/Deep-Armocromia) |
| 2 | **Доменная модель** | «Сезон» — не бинарная метка, а комбинация undertone, value, contrast, intensity; классические 4 сезона слишком грубы | Принята **Armocromia Flow Theory**: 12 sub-types (Warm, Light, Bright, Cool, Soft, Deep) внутри 4 parent seasons; «скольжение» между соседними сезонами заложено в таксономию |
| 3 | **Ground truth / разметка** | Нельзя заказать разметку на crowdsourcing — нужен экспертный протокол Armocromia | Два этапа обучения студентов под экспертом (Rossella Migliaccio): (a) теория undertone/value/contrast/intensity; (b) протокол сравнения неразмеченного лица с эталонными кластерами |
| 4 | **Сбор pivot-набора** | Нужны надёжные эталоны для transfer-разметки | **Pivotal Armocromia Set**: ~1900 фото ~250 знаменитостей (актеры, модели и т.д.) с уже известной экспертной классификацией; фото из соцсетей |
| 5 | **Масштабирование разметки** | Мало данных только на знаменитостях; нужен объём для DL | Случайная выборка **3000 фото из CelebA**; аннотация по pivot-примерам → итого ~4920 размеченных фото |
| 6 | **Дубликаты** | Повторы искажают train/test и метрики | **Perceptual hash** (Average Hash, 16 bit) — удаление дубликатов |
| 7 | **Изоляция цветовых зон** | Armocromia смотрит только кожу, волосы, глаза — фон и одежда мешают | **Facer** (DL face parsing): маски hair, skin, nose, brows, eyes, mouth → объединение в единую маску «features of interest» |
| 8 | **Качество сегментации** | Автопарсинг ошибается (bbox, маски) | Ручная проверка всех crop/mask; исправление ошибок в **GIMP** |
| 9 | **CelebA vs pivot** | У CelebA уже есть face crops; у pivot — нет | Для pivot — парсинг через Facer; для CelebA — используют готовые face-related данные + та же логика масок |
| 10 | **Train / test split** | Утечка одного человека в обе выборки завышает accuracy | Split **80/20** (~4000 / ~900) с гарантией **нет пересечения лиц** между train и test |
| 11 | **Два бенчмарка** | Coarse (4) и fine (12) — разная сложность; одна метрика не покрывает обе | Отдельные задачи: **Season Classification (4)** и **Season Sub-Type Classification (12)**; метрики: accuracy, precision, recall, F1, top-k |
| 12 | **Выбор признаков / backbone** | Неясно, нужны face-specific или general ImageNet-признаки | Сравнение **FaRL16/64** (ViT, LAION-Face) vs **ResNeXt50** (ImageNet); frozen backbone + trainable head |
| 13 | **Классификатор** | Переобучение на ~5k фото | Два FC-слоя: первый — dim/2 + ReLU + **dropout 0.5**; второй — num_classes (4 или 12) |
| 14 | **Аугментации** | Мало данных, риск overfit | Random crop, flip 50%, color jitter (brightness/contrast/saturation), random sharpness |
| 15 | **Обучение** | Стабильное сравнение моделей | 50 epochs, AdamW (lr=1e-3), Cosine Annealing warm restarts, batch 64, input 224×224 |
| 16 | **Masked input для DL** | Классификатор не должен «смотреть» на фон | **RGB-masked images** (только пиксели внутри face-parsing маски) подаются в backbone |
| 17 | **Интерпретация ошибок** | Непонятно, где модель ломается | **Confusion matrix** по лучшим моделям; анализ пар Autumn↔Winter, Spring↔Summer, Deep/Cool sub-types |
| 18 | **Оценка «почти правильных» ответов** | Один argmax скрывает близость соседних сезонов (Flow Theory) | **Top-2** для 4 классов (~81%), **Top-3** для 12 (~66%) — отдельные метрики |

---

## Результаты по этапам (что получилось)

| Задача | Лучшая модель | Accuracy | Top-k | Главный вывод |
|--------|---------------|----------|-------|---------------|
| 4 сезона | FaRL64 | **55.4%** | Top-2 **80.8%** | Face-specific pretrain лучше ImageNet; Winter предсказывается лучше; Autumn↔Winter — главная путаница |
| 12 sub-types | FaRL16 | **31.8%** | Top-3 **66.3%** | Fine-grained сложнее; FaRL64 хуже на 12 классах — «широкие» признаки мешают тонкой гранулярности |

---

## Нерешённые проблемы (явно в Conclusion)

| Проблема | Статус в работе | Запланированное решение (future work) |
|----------|-----------------|--------------------------------------|
| Низкая accuracy на 12 sub-types (~32%) | Не решено | Больше данных; hierarchical / ordinal learning |
| Путаница соседних сезонов | Частично объяснено (Flow Theory, overlapping features) | Явная иерархия season → sub-type |
| DL vs classical CV | Не сравнивали | Систематическое сравнение с LAB / rule-based пайплайном |
| Освещение / AWB | Не учитывалось | Не описано в статье (gap для follow-up) |
| CelebA bias | Признан косвенно (random sample «чтобы избежать bias») | Расширение pivot-set и in-the-wild selfie |

---

# 2. AUA Capstone — rule-based seasonal color + makeup (ImageJ)

Источник: Khachatryan, Asryan, Sargsyan, *Automated Seasonal Color Classification and Makeup Recommendation* (AUA, Spring 2025).  
PDF: `Automated_Seasonal_Color_Classification_and_Makeup_Recommendation-...pdf` на Desktop.

**Стек:** ImageJ + Java-плагины/макросы (без ML). **Датасет:** 200 лиц FEI (разное освещение). **Выход:** 16 типов (12 + 4 transitional), палитры, рекомендации из Kaggle Cosmetic Dataset.

---

## Общая схема пайплайна

```
Фото лица
  → [Ch.1] HSV skin detection (Hue + S×V) → эксперименты порогов / рекурсия (отклонена)
  → Skin palette (5 hue-групп, avg hue 13–24, +20% brightness)
  → [Ch.2] Маска кожи → binary → fill holes → морфология → вычитание
  → Eyes/lips: Method1 (Y-split) → Method2 (кластеры по area + centroid) ✓
  → Hair: crop над бровями; mask(hue-only) − mask(hue+SV) → avg HSB
  → [Ch.3] 4 фактора Munsell (hue, chroma, value, contrast) → шкала 1–5
  → Lookup-таблица → 16 seasonal types → генерация swatch-палитры (hue/chroma/value grid)
  → Product match: Δ к среднему цвету кожи / губ+глаз+сезон / contrast shades
```

---

## Шаги пайплайна: проблема → решение

| # | Этап | Проблема | Как решено в работе |
|---|------|----------|---------------------|
| 1 | **Постановка** | Ручной seasonal color analysis субъективен, дорог, не воспроизводим | Автоматический data-driven пайплайн: фото → 16 типов → палитра + макияж; цель — virtual try-on |
| 2 | **Инструмент** | Нужен быстрый прототип без DL-инфраструктуры | **ImageJ** + custom Java plugins и recordable macros |
| 3 | **Датасет** | Нет разметки сезонов; нужны разные условия света | **FEI Face Database**, 200 фото; в тексте — план ML на этих же результатах |
| 4 | **Детекция кожи** | RGB коррелирован; кожа зависит от освещения и этноса | **HSV**: Hue в диапазоне кожи + **S×V×256** в диапазоне; литература 0–30 hue, SV 30–150 как старт |
| 5 | **Подбор порогов кожи** | Широкие пороги захватывают плечи, тени, брови, губы | Итерации: SV 45–105 / 35–105; комбо Hue 3–21 + SV 35–80 (слишком узко — почти только губы); финал **Hue 3–24, S×V 40–80** |
| 6 | **Тёмное освещение** | Те же пороги «съедают» кожу или тянут тени | Отдельный режим: **Hue 3–24, S×V 30–96** (~100) |
| 7 | **Персональные пороги** | Универсальные диапазоны не на всех 200 фото | **Recursive Method 1:** сужение hue вокруг μ±15 до сходимости — **отклонено** (волосы, брови, губы; потеря кожи у глаз) |
| 8 | **Персональные пороги v2** | Mean слишком груб | **Recursive Method 2:** hue = μ ± kσ, k∈{1,1.5,2} по σ; stop если range < 7 — лучше, но всё равно теряются щёки/край лица → **фиксированные пороги** |
| 9 | **Палитра кожи** | Одно среднее по всем skin-пикселям → неверный undertone | 4 метода группировки hue×SV; **Method 4:** 5 hue-групп (3–24), в среднее только **hue 13–24** (меньше веса красным пикселям) |
| 10 | **Свет кожи в кадре** | Средний цвет темнее реальной кожи (освещение FEI) | **+20% brightness** на цветах палитры перед финальным average |
| 11 | **Глаза и губы — сегментация** | Нужно отделить от кожи и фона без DL parsing | После skin mask: grayscale → threshold **254** → binary → **fill holes** → invert → subtract → dilate → mask на оригинал |
| 12 | **Глаза/губы — Method 1** | Простое «всё не-кожа» даёт шум | Разделение по **средней Y** несkin-пикселей: верх = глаза, низ = губы — **много false positives** |
| 13 | **Глаза/губы — Method 2** | Method 1 ненадёжен | **Connected components:** 2 крупнейших верхних региона = глаза; нижний = губы (area + centroid) |
| 14 | **Цвет глаз** | Зрачок/склера портят среднее | Палитра 4 цвета: **low saturation** + **low/mid brightness**; ч/б отбрасывают, среднее по 2 оставшим |
| 15 | **Цвет губ** | Тени дают коричневый bias; (R+B)/2>G один средний тёмный | Фильтр **brightness > 50** (luminance 0.299R+0.587G+0.114B); сортировка → **3 кластера** по яркости → avg в HSB (hue через vector math) |
| 16 | **Волосы** | Волосы близки к коже по hue, отличаются S/V | Crop от **bbox глаз (y+20)**; mask A: hue 3–24 + SV 40–80 (кожа); mask B: только hue 3–24 (кожа+волосы); **B − A** = hair; avg HSB (hue через cos/sin) |
| 17 | **Сезон — модель** | Нужна связь с color theory, не только эвристики | **Munsell-подобные 4 оси:** undertone (hue кожи), chroma, value, contrast ratio **(L_min+0.05)/(L_max+0.05)** |
| 18 | **Сезон — агрегация** | Как свести зоны лица в один профиль | Hue = hue кожи; Chroma = mean(skin, eyes, lips); Value = **0.7×skin + 0.15×eyes + 0.15×hair**; Contrast = L по skin/hair/eyes |
| 19 | **Сезон — шкала** | Несопоставимые диапазоны факторов | Каждый фактор → **нормализация 1–5** (5 равных bins; undertone: red/pink = cold, yellow = warm) |
| 20 | **Сезон — класс** | 12 sub-types + пограничные | **Lookup-таблица Fig.3** → **16 типов** (12 + True Bright/Light/Soft/Deep) |
| 21 | **Палитра сезона** | Текстовые советы недостаточны | Обратное отображение scores → диапазоны hue (полное колесо 0–360°), chroma, value → сетка (~28 800 комбинаций; UI — шаг hue 1/30, chroma/value 1/20) |
| 22 | **Продукты** | Нет связи сезона с SKU | **Kaggle Cosmetic Brand Products Dataset**; foundation/concealer → ближайший к **avg skin**; lipstick/blush/eyeshadow → губы+глаза+сезонная палитра; плюс **contrast shades** |
| 23 | **Метрика качества** | Нет accuracy сезона vs эксперт | Только **hue deviation:** avg hue skin pixels vs avg hue палитры; сезонная accuracy **не приведена** |
| 24 | **Валидация датасета** | FEI в основном один контекст (бразильцы) | В Conclusion: нужен **другой датасет** + будущий **supervised ML** на собранных палитрах |

---

## Результаты по этапам (что получилось)

| Этап | Результат | Ограничение |
|------|-----------|-------------|
| Skin HSV | Рабочие маски на подмножестве 200 фото; визуальные сравнения методов | Зависимость от освещения; отдельные пороги для dark lighting |
| Skin palette | Method 4 стабильнее Method 1–3 на их примерах | Метрика только hue deviation, не ΔE / не сезон |
| Eyes/lips | Method 2 лучше Method 1 | Без landmarks; падает на нестандартном повороте/позе |
| Hair | Mask subtraction работает при видимой линии роста | Плохо при закрытых волосах, чёлке, однотонном фоне |
| Season 16 | End-to-end пример (e.g. Warm Spring + lipstick) | Нет численной accuracy vs human stylist |
| Products | Логика nearest-match описана, демо на уровне концепта | Нет A/B, нет user study |

---

## Нерешённые проблемы (явно в Conclusion / Future Work)

| Проблема | Статус в работе | Запланированное решение |
|----------|-----------------|-------------------------|
| Субъективность заменена не полностью | Пороги подобраны на 200 фото вручную | ML на размеченных палитрах |
| Рекурсивная адаптация порогов | **Не работает** | Не использовать; фиксированные или parsing+Lab |
| Этническое / lighting diversity | FEI bias | Другие датасеты, AWB-коррекция |
| Virtual try-on | Не реализован | AR: OpenCV/Dlib, MediaPipe/FaceMesh, ARKit/ARCore, alpha blend |
| Сравнение с DL | Нет | Их же future work + бенчмарк vs Deep Armocromia |

---

# 3. Colorinsight (student web service, FaRL + ResNet18)

Источник: [github.com/PSY222/Colorinsight](https://github.com/PSY222/Colorinsight) — корейский student-проект «퍼스널컬러 진단모델».  
Backend: FastAPI (`facer/main.py`). Frontend: Spring Boot на `localhost:3000` (в репо **не выложен**).  
Датасет: ~750 фото корейских знаменитостей (Google Image crawling, Selenium) — **не опубликован** (privacy).

---

## Общая схема пайплайна

```
Selfie (base64) → FastAPI
  → RetinaFace (retinaface/mobilenet) — детекция лица
  → FaRL face parsing (farl/lapa/448) — сегментация зон
  → [основной /image] skin mask (class 1, ≥0.5) → RGB crop «кожа на чёрном»
      → ResNet18 (best_model_resnet_ALL.pth) → 4 сезона
      → callback Spring Boot /output
  → [legacy /lip] lip mask (classes 7+9) → RGB sample (40 px) → L2 до эталонных палитр
      → majority vote → callback /output2
```

---

## Шаги пайплайна: проблема → решение

| # | Этап | Проблема | Как решено в работе |
|---|------|----------|---------------------|
| 1 | **Постановка** | Платные personal color консультации субъективны и дороги; растёт спрос на personalized beauty | Веб-сервис Colorinsight: фото → автоматический сезон → рекомендации (UI вне репо) |
| 2 | **Датасет** | Нет открытых labeled данных для Korean personal color | **~750 фото** корейских celebs, crawling через **Selenium**; **data augmentation** из-за малого объёма |
| 3 | **Privacy** | Публикация фото celebs проблематична | Датасет **не загружен** в репозиторий |
| 4 | **Сегментация лица** | Классические/слабые парсеры ошибаются на поворотах, нестандартной форме | Сравнили несколько моделей → выбрали **FaRL** (face parsing) + **RetinaFace** — лучше на сложных позах |
| 5 | **Изоляция кожи** | Фон, волосы, макияж искажают цвет кожи | Из seg logits: **class 1 = skin**, threshold ≥ 0.5 → `save_skin_mask()` → пиксели кожи на чёрном фоне (`temp.jpg`) |
| 6 | **Подход 1 — RGB + L2** | Как классифицировать сезон без DL? | Случайная выборка RGB с **кожи** → L2 distance до 3 эталонных RGB на сезон → vote — **20–30% accuracy**, **autumn почти не определяется** |
| 7 | **Подход 2 — tabular ML** | Может, достаточно RGB-фич? | Structured dataset (R,G,B columns) + классический ML classifier — тоже **20–30%**, тот же провал на autumn |
| 8 | **Подход 3 — CNN на маске** | Нужны spatial patterns undertone/chroma | Dataset **skin-mask images** из celeb-фото; сравнили **MobileNet, ResNet, EfficientNet** → **ResNet18 + Adam** лучший → **~60% accuracy** |
| 9 | **Два inference path** | Старый lip-based метод ещё в коде | **`/image`** — ResNet на skin mask (prod); **`/lip`** — RGB с **губ** (classes 7+9), filter R≥97 B≤227, 40 random px, L2 vote (legacy) |
| 10 | **Lip filter** | Помада, тени, блики на губах | Hard filter: **R ≥ 97, B ≤ 227** перед sampling |
| 11 | **Reference swatches** | Нужны эталоны 4 сезонов | Фиксированные RGB triplets в `calc_dis()` (3 цвета × 4 сезона из research paper) |
| 12 | **Serving** | Нужен API для мобильного/web клиента | **FastAPI** POST `/image`, `/lip`; CORS для `:3000`; base64 in/out |
| 13 | **Интеграция с UI** | Backend не standalone | После inference — **HTTP callback** на Spring Boot (`/output`, `/output2`) с base64-encoded result |
| 14 | **Модель в репо** | Colab training, deploy weights | `best_model_resnet_ALL.pth` (~45 MB) loaded in `skin_model.py` |
| 15 | **Inference transforms** | Preprocessing для ResNet | Resize 224, ToTensor, Normalize(0.5); **RandomHorizontalFlip/VerticalFlip при inference** (недетерминизм — вероятный баг) |
| 16 | **Метрики / ablation** | Как доказать прогресс 20%→60%? | Подробные experiment reports в [Notion](https://tar-tilapia-c6d.notion.site/403c8d583e3a4f6bb9f76ea6efd991d5?v=f9b650bea3e144918ec577eb464ddcd5) (Korean) |
| 17 | **Ограничения accuracy** | 60% недостаточно для prod | В README: нужны **больше epoch, больше данных, обычные люди разных races/углов** |

---

## Результаты по этапам (что получилось)

| Подход | Метрика | Главный вывод |
|--------|---------|---------------|
| L2 RGB (кожа или губы) | **~20–30%** | Простые color distances не работают; autumn — systematic failure |
| Tabular ML на RGB | **~20–30%** | Табличные фичи без spatial context недостаточны |
| ResNet18 на skin-mask image | **~60%** | Mask → CNN — рабочий baseline; лучше ImageNet-классификаторов в их сравнении |
| FaRL parsing | Качественно (без числа) | Устойчивее альтернатив на поворотах лица |

---

## Нерешённые проблемы (явно в README / Possible Improvements)

| Проблема | Статус в работе | Запланированное решение |
|----------|-----------------|-------------------------|
| Accuracy ~60% | Недостаточно | Больше данных, больше epochs |
| Korean celeb bias | Признано | Selfies обычных людей, diverse races |
| Только 4 сезона | By design | Нет 12 sub-types |
| Нет requirements.txt / deploy docs | Не решено | — |
| Spring Boot hard dependency | Архитектурный lock-in | — |
| Random augmentations at inference | Баг | Убрать flips в eval mode |
| Нет seasonal GT metrics в репо | Только narrative ~60% | — |

---

# 4. Park et al. — rule-based personal color + virtual makeup (Dlib)

Источник: J. Park, H. Kim, S. Ji, E. Hwang, *An Automatic Virtual Makeup Scheme Based on Personal Color Analysis* (IMCOM'18, DOI: [10.1145/3164541.3164612](https://doi.org/10.1145/3164541.3164612)).  
PDF: `3164541.3164612.pdf` на Desktop.

**Стек:** Dlib (68 landmarks), OpenCV 2.4, Python 3.5, Matlab 2017a. **Датасет:** 100 selfie (20 человек × 5), iPhone 7, естественный свет. **Выход:** warm/cool + 4 сезона → виртуальный макияж (foundation, blush, lip, eyeshadow, eyeline, eyebrow) по expert makeup DB + анкета.

---

## Общая схема пайплайна

```
Selfie
  → Dlib 68 landmarks (DFL)
  → [PCA] Iris: eye 12 pts → 1D interp → Hough circle (pupil) → K-means (iris color)
  → [PCA] Hair: region above brows → Canny(R,G,B) → morph closing → hair mask
  → [PCA] Skin: cheek/jaw via jaw interp (меньше солнца)
  → RGB→Lab: undertone = compare a vs b (skin); season = L-contrast iris/hair/skin (threshold 13)
  → Анкета (age, gender, makeup goal) + lookup expert makeup DB
  → [VM] Foundation: skin Eq.(1) + face crop; Lab blend Eq.(2–3)
  → [VM] Blush / shadow: landmark masks + Gaussian blur (gradation)
  → [VM] Lip: Dlib lip segments + interp
  → [VM] Eyeline / eyebrow: template resize + map to eye/brow geometry
  → User study: 4 criteria × 5 makeup items (1–5)
```

---

## Шаги пайплайна: проблема → решение

| # | Этап | Проблема | Как решено в работе |
|---|------|----------|---------------------|
| 1 | **Постановка** | Personal color дорог и субъективен; virtual makeup apps требуют ручного подбора цветов | End-to-end: selfie → auto personal color → auto virtual makeup без выбора пользователем |
| 2 | **Landmarks** | Нужны стабильные зоны лица без тяжёлого DL parsing | **Dlib** pre-trained face marker — **68 точек** (DFL); reuse для PCA и makeup masks |
| 3 | **Iris — контур глаза** | 12 точек Dlib не дают плотную маску радужки | **1D interpolation** верхней/нижней дуги глаза → замкнутая eye region |
| 4 | **Iris — зрачок** | Нужно отделить радужку от склеры/ресниц | **Circle Hough Transform** на eye mask → pupil circle |
| 5 | **Iris — цвет** | В pupil area есть тёмные артефакты | **K-means** на пикселях с относительно **высокими RGB** вне pupil = iris color |
| 6 | **Hair — ROI** | Где искать «натуральный» цвет волос | Область **выше highest eyebrow landmark** (parietal candidate) |
| 7 | **Hair — сегментация** | Grayscale/binary не отделяет лоб от волос | **Canny** отдельно по R, G, B → sum edges → **morphological closing** → hair mask |
| 8 | **Skin — ROI** | Среднее по всему лицу тянет тени и солнце | Щёки / **jaw line**: interp по jaw coords DFL, пара точек соединяется в polygon |
| 9 | **Undertone** | Нужен объективный warm/cool | **RGB→Lab**; по skin: если **a > b** → cool, иначе warm (упрощённое правило) |
| 10 | **Season (4)** | Внутри tone нужны spring/summer/autumn/winter | **L-contrast** между iris, hair, skin; порог **13** отделяет bright (spring/winter) vs muted (autumn/summer); spring/autumn = warm, summer/winter = cool |
| 11 | **Контекст пользователя** | Один сезон → разный макияж (daily vs chic) | **Questionnaire**: age, gender, purpose of makeup → фильтр записей в DB |
| 12 | **Makeup DB** | Нет единого стандарта цветов макияжа | **Expert-defined database** (colorist): цвета + методы по personal color + situation; shapes для shadow/eyeline |
| 13 | **Foundation — skin mask** | Eq.(1) Peer et al. захватывает фон/одежду | Crop по контуру лица DFL; вырез eyes/lips через interp; upper = hair line, lower = jaw |
| 14 | **Foundation — natural blend** | Прямая замена RGB «убивает» текстуру кожи | Конверт в **Lab**; смешивание **Eq.(2–3)** с параметром **α** (strength); clamp min/max L,a,b |
| 15 | **Blush** | Пятно без градиента выглядит неестественно | 4 cheek points → closed curve (interp) → **symmetric** left/right → **Gaussian blur** на mask |
| 16 | **Lip** | Dlib дает coarse lip polygon | Сегменты oral angle / upper / lower lip → interp → closed lip mask |
| 17 | **Eyeshadow** | Нет parsing века | **Relative geometry** eye–eyebrow (e.g. 1/3 point); 3–4 anchor points → shadow mask → Gaussian blur |
| 18 | **Eyeline / eyebrow** | Произвольные формы по стилю | **Predefined templates**; resize по длине глаза; map к верхнему краю; brow — fill old brow skin color then template |
| 19 | **Оценка PCA** | Насколько надёжны iris/hair/skin? | 100 фото; accuracy извлечения регионов (Fig. 11): **высокая** на Dlib; **hair ниже** eye/jaw |
| 20 | **Оценка VM** | Качество рендера субъективно | Survey **20 subjects**, 5 items × 4 criteria (color, region size, location, intensity), scale 1–5 |

---

## Результаты по этапам (что получилось)

| Этап | Результат | Ограничение |
|------|-----------|-------------|
| Landmark + iris/jaw | Высокая accuracy извлечения (Fig. 11) | Зависимость от качества Dlib |
| Hair extraction | Работает на типичных причёсках | **Низкая accuracy** при unusual style или hair ≈ skin color |
| Personal color (4 season) | End-to-end без эксперта | Примитивный undertone (a vs b); нет AWB; только 4 сезона |
| Foundation / blush / shadow | User satisfaction **≥ 4/5** (Fig. 13) | — |
| Lip / eyeline / eyebrow | Satisfaction **ниже** 4 | Fixed lip shape vs real contour; template curvature ≠ user brow/eye line |
| Product loop | Полный preview makeup на selfie | Нет метрики accuracy сезона vs human colorist |

---

## Нерешённые проблемы (явно в Conclusion / Discussion)

| Проблема | Статус в работе | Запланированное / implicit gap |
|----------|-----------------|--------------------------------|
| Hair segmentation | **Слабое звено** (Fig. 11) | Не предложено DL parsing; Canny fragile |
| Eyeline / eyebrow templates | **Low user scores** | Curvature mismatch; нужен spline/warp, не flat template |
| Lip contour | **Low score on region size** | Landmarks ≠ индивидуальный контур губ |
| Illumination / AWB | Не учитывалось | Только natural light iPhone 7 |
| Season GT | Нет численной accuracy vs colorist | Только feature extraction accuracy |
| 4 seasons only | By design | Нет 12 sub-types |
| Expert DB maintenance | Static lookup | Масштабирование палитр вручную |

---

# 5. Colors Matter — feature color classification (CV + Delta E)

Источник: R. Alyoubi, T. Alharbi, A. Alghamdi, Y. Alshehri, E. Alghamdi, *Colors Matter: AI-Driven Exploration of Human Feature Colors* (arXiv:2505.14931v1, May 2025).  
PDF: `2505.14931v1.pdf` на Desktop. Код и датасеты заявлены в GitHub (ссылка в статье).

**Стек:** Facer, Timm, RetinaFace, LaPa, Dlib (68 landmarks), OpenCV, X-Means/K-means, CIEDE2000. **Датасет:** 720 + 400 фото (8 skin classes); ~10 фото/класс для hair/iris/undertone (smoke only). **Выход:** 8-class skin tone, hair color, iris color, Warm/Cool undertone (по венам запястья).

---

## Общая схема пайплайна

```
[Face photo]                          [Wrist photo]
  → Facer: face detection + parsing
  → Timm: skin segmentation (filter non-skin)
  → X-Means on HSV skin pixels (+ optional Gaussian blur)
  → Dominant cluster → CIEDE2000 match → 8-class skin tone

  → RetinaFace + LaPa → hair mask
  → K-means (k=3) in LAB → dominant + avg color
  → CIEDE2000 → hair color category

  → Dlib 68 landmarks → circular iris mask (exclude pupil/sclera)
  → avg RGB → LAB → CIEDE2000 → iris color category

[Wrist]
  → LAB conversion → skin/vein masks (thresholds)
  → morphological closing → mean LAB of veins
  → CIEDE2000 vs warm [70,20,40] / cool [60,-20,-30] → Warm | Cool

→ Results to server (beauty tech / personalization)
```

---

## Шаги пайплайна: проблема → решение

| # | Этап | Проблема | Как решено в работе |
|---|------|----------|---------------------|
| 1 | **Постановка** | Тон кожи, волос, глаз и undertone — subtle, зависят от освещения и этноса; простой RGB/HSV detection недостаточен | Мультимодальный пайплайн: **face + wrist**; discrete classification по кастомным палитрам через perceptual color distance |
| 2 | **Таксономия кожи** | Fitzpatrick (6) — bias к white skin; Monk (10) — мало оттенков; PERLA — не для diverse skin tone classification | Собственная **8-class skin tone scale** (Fig. 4), подобранная trial-and-error под classifier boundaries; не lab-calibrated |
| 3 | **Датасет skin tone** | Нужны diverse tones + lighting; мало labeled данных | **Dataset 1:** 720 img (FFHQ 312 + Face Research Lab London 88 + SFHQ synthetic 320); **Dataset 2:** 400 FFHQ (50/class) — для lighting/shadow stress-test |
| 4 | **Датасет hair/iris/undertone** | Нет structured GT для fine-grained hair/eye/vein | ~**10 img/class** из web — только **smoke test**, не training/eval protocol |
| 5 | **Face detection** | Фон, поза, освещение мешают изоляции лица | **Facer** (DL) — face detection + parsing; устойчивее CASCo и classical HSV |
| 6 | **Skin segmentation** | Eyes, hair, teeth, shadows попадают в skin sample | **Timm** pre-trained facial parsing → filter non-skin; после CASCo-failure перешли на Facer+Timm |
| 7 | **CASCo baseline** | Готовый инструмент skin tone (CASCo + PERLA) | Протестировали → **отказались**: segmentation errors (shadows, hair, background); PERLA не покрывает diverse range |
| 8 | **Dominant skin color** | Mean RGB/HSV по маске тянет блики, тени, поры | **X-Means** на HSV пикселях skin ROI; k-means++ init; **largest cluster center** = dominant tone |
| 9 | **Шум на skin ROI** | Fine details и noise смещают dominant color | **Gaussian blur** на skin region **до** clustering — ключевой прирост: accuracy **0.42 → 0.73–0.80** |
| 10 | **Color space для skin** | RGB Euclidean не perceptually uniform | Лучший combo: cluster в **HSV**, match через **CIEDE2000** (конвертация в LAB для ΔE) |
| 11 | **Skin classification** | Нужен объективный nearest-class без DL | **argmin CIEDE2000**(dominant color, 8 reference swatches) — проще и точнее SVM+ResNet18 (76% vs 80%) |
| 12 | **Gamma correction** | Lighting inconsistencies across datasets | **Gamma correction** + blur kernel ∝ image size — нормализация перед clustering |
| 13 | **Two-stage skin** | 8 классов близки по brightness — путаница на границах | Main class (4 groups) → subclass (2 each) через CIEDE2000 — **64%**, хуже одноступенчатого |
| 14 | **SVM alternative** | Может DL-features лучше rules? | **ResNet-18 features + SVM** (GridSearchCV, augment to 3544 samples): test **75–76%** — ниже Delta E–HSV+blur |
| 15 | **Hair segmentation** | Hair ≈ skin по hue; background noise | **RetinaFace** detection + **LaPa** parsing → hair object mask |
| 16 | **Hair color extraction** | Mixed shades, highlights | **K-means k=3** in **LAB** → dominant cluster; также **average** color; финальный score = **avg distance** dominant + average vs palette |
| 17 | **Hair — слабые классы** | Blonde/brown/dark blonde overlap | Discrete palette match; тёмные (black, dark brown) **80–90%**, светлые (blonde, brown) **20–50%** — underrepresentation + lightness range |
| 18 | **Iris — v1** | Pupil size varies; pupil color artifacts | Distance iris–pupil + masks; RGB→Lab; 3×3 categories — overlap medium/light |
| 19 | **Iris — v2** | Noise, multi-color hazel | **Gaussian blur** on iris + self-attention — лучше, но hazel gradients всё ещё hard |
| 20 | **Iris — v3 (final)** | Нужна стабильная geometry | **Dlib 68 landmarks** → eye region → **circular mask** (center+radius), exclude pupil/sclera; avg RGB→Lab→**CIEDE2000**; оба глаза → average |
| 21 | **Iris results** | Grey vs blue/green confusion | Dark blue/hazel/black **100%**; grey **70%** (overlap light blue/green) |
| 22 | **Undertone — face-only gap** | Overtone (melanin) ≠ undertone (hue beneath); makeup/tan/lighting mask undertone on face | Отдельный модуль: **wrist photo** + **vein color** |
| 23 | **Undertone — v1** | Cosine similarity на vein LAB — ambiguous green/blue | Warm/cool refs + **cosine similarity** — misclass near boundary |
| 24 | **Undertone — v2 (final)** | Нужна perceptual precision | LAB skin/vein masks → morph **closing** → mean vein LAB → **CIEDE2000** vs warm `[70,20,40]` / cool `[60,-20,-30]` |
| 25 | **Undertone refs** | Нет стандарта LAB для warm/cool veins | **Trial-and-error** reference values — не lab-calibrated |
| 26 | **Undertone results** | Binary Warm/Cool only | Warm **80%**, Cool **70%** на ~10 test images/class |
| 27 | **Evaluation skin** | Fair comparison blur vs no blur | Primary + Secondary datasets; metrics: accuracy, precision, recall, F1 |
| 28 | **Evaluation hair/iris/undertone** | Мало GT | Manual truth table, **10 images/class** — exploratory, не rigorous benchmark |

---

## Результаты по этапам (что получилось)

| Задача / метод | Метрика | Главный вывод |
|----------------|---------|---------------|
| Skin — X-Means + CIEDE2000 + **HSV** + **blur** | Acc **80%** (secondary), **73%** (primary); F1 **0.80** | **Blur + HSV + Delta E** — лучший rule-based рецепт; без blur ~40–42% |
| Skin — X-Means + CIEDE2000 + RGB + blur | Acc **65–66%** | RGB хуже HSV для skin overtone |
| Skin — X-Means + Euclidean + RGB + blur | Acc **53–56%** | Euclidean проигрывает CIEDE2000 |
| Skin — Two-stage (main→sub class) | Acc **64%** | Иерархия не помогла vs flat 8-class |
| Skin — SVM + ResNet-18 | Test acc **75–76%** | DL-features не beat простой palette matching |
| Hair color (10 img/class) | Dark brown **90%**, black **80%**, brown **20%**, blonde **50%** | Тёмные стабильнее; светлые/смешанные — главная слабость |
| Iris color (10 img/class) | Dark blue/hazel/black **100%**; grey **70%** | Уникальные оттенки легко; grey/hazel gradients hard |
| Undertone wrist (10 img/class) | Warm **80%**, Cool **70%** | CIEDE2000 лучше cosine; второй модальности (wrist) достаточно для demo |

---

## Нерешённые проблемы (явно в Discussion / limitations)

| Проблема | Статус в работе | Запланированное / implicit gap |
|----------|-----------------|--------------------------------|
| Custom palettes не lab-calibrated | Признано (trial-and-error) | Formal pigment study / expert calibration |
| Hair/iris/undertone eval | ~10 img/class only | Rigorous labeled dataset + cross-validation |
| Light hair colors (blonde, brown) | **20–50%** accuracy | Больше данных; mixed-color modeling |
| Hazel / grey iris | Overlap medium/light | Multi-sample iris ring; gradient-aware clustering |
| Neutral undertone | Не моделируется | Только Warm vs Cool binary |
| Illumination / AWB | Gamma + blur only | Нет systematic AWB или albedo correction |
| Seasonal color analysis | **Вне scope** | 8 skin classes ≠ 12-season armocromia |
| CASCo / PERLA | Отвергнуты | — |
| Two-stage hierarchy | Хуже flat | Не рекомендуется |

---

# Сравнение работ (общая таблица)

| Критерий | Deep Armocromia | AUA Capstone | Colorinsight | Park IMCOM'18 | Colors Matter |
|----------|-----------------|--------------|--------------|---------------|---------------|
| **Источник** | Stacchio et al., ECCV workshop; [Deep-Armocromia](https://github.com/lorenzo-stacchio/Deep-Armocromia) | Khachatryan et al., AUA 2025 (PDF) | [PSY222/Colorinsight](https://github.com/PSY222/Colorinsight) | Park et al., IMCOM'18 (DOI 3164541.3164612) | Alyoubi et al., [arXiv:2505.14931](https://arxiv.org/abs/2505.14931) |
| **Тип** | Research + публичный датасет | Academic capstone (ImageJ) | Student prod prototype (FastAPI) | Academic prototype (Matlab/Python) | Research prototype (CV pipeline + GitHub) |
| **Подход** | DL end-to-end: masked RGB → class | Classical CV: HSV + morphology + lookup table | Hybrid: FaRL parsing + ResNet18 **или** RGB L2 fallback | Classical CV: Dlib landmarks + Lab rules + **virtual makeup render** | **Rule-based CV:** clustering + **CIEDE2000** palette match; optional SVM baseline |
| **Сегментация** | Facer (DL parsing), ручная правка в GIMP | HSV skin mask + binary morphology | FaRL (`farl/lapa/448`) + RetinaFace | **Dlib 68** + interp masks; hair = Canny; iris = Hough+K-means | **Facer + Timm** (skin); **RetinaFace+LaPa** (hair); **Dlib 68** (iris); LAB masks (wrist veins) |
| **Цветовое пространство** | RGB (masked image в CNN) | HSV / HSB + Munsell-like 4 axes | RGB (L2) + RGB masked image (CNN) | **CIELAB** (undertone a/b; season L-contrast) | **HSV** (skin cluster) + **LAB** (hair/iris/undertone) + **CIEDE2000** matching |
| **Зоны лица** | Skin + hair + eyes (единая маска) | Skin, eyes, lips, hair (отдельно) | Skin (main) / lips (legacy path) | **Iris + hair + cheek/jaw skin** (для PCA); full face skin (foundation) | **Skin, hair, iris** (face); **wrist veins** (undertone) — **multi-modal** |
| **Таксономия** | **4 + 12 sub-types** (Flow Theory) | **16 types** (12 + 4 transitional) | **4 seasons** only | **4 seasons** + warm/cool; questionnaire for makeup style | **8 skin classes** + discrete hair/iris palettes + **Warm/Cool** undertone (**не seasonal**) |
| **Датасет** | ~4920, expert-labeled, CelebA + pivot celebs | 200 FEI, **без сезонной GT** | ~750 Korean celebs, **не опубликован** | **100** selfie (20×5), natural light | **1120** skin (FFHQ+London+SFHQ); hair/iris/undertone ~10/class smoke |
| **Разметка** | 2-stage expert training (Migliaccio protocol) | Rule output only | Manual season labels for celebs | Expert makeup DB; **нет season accuracy vs colorist** | 8 skin classes; hair/iris/undertone — minimal manual GT |
| **DL backbone** | FaRL16/64, ResNeXt50 | Нет (future work) | ResNet18 (MobileNet/EffNet compared) | **Нет** (Dlib только как landmark detector) | ResNet-18 **только как feature extractor для SVM** (76%); prod path — rules |
| **Accuracy (season / skin)** | **55.4%** (4-class, FaRL64); **31.8%** (12-class) | Не измерялась | **~60%** (4-class, ResNet, skin mask) | **Не измерялась** (только region extraction accuracy) | **80%** skin 8-class (Delta E–HSV+blur); hair/iris/undertone 70–100% (weak eval) |
| **Top-k** | Top-2 **80.8%** (4); Top-3 **66.3%** (12) | — | Не reported | — | — |
| **Explainability** | Confusion matrix, neighbor seasons | 4 Munsell axes → scores 1–5 | Нет (black-box CNN) | Lab rules + expert DB lookup | **Delta E scores** к каждому swatch; discrete class labels |
| **Продуктовый слой** | Нет | Kaggle product match + contrast shades | Web UI (Spring Boot, не в репо) | **Virtual makeup render** (6 products) + user satisfaction survey | Beauty tech / personalization (концепт; без VM) |
| **Virtual try-on** | Нет | Planned (AR) | Нет (classification only) | **Реализован** (Lab blend + blurred masks + templates) | Нет |
| **Освещение / AWB** | Не учитывалось | Отдельные HSV пороги для dark light; +20% brightness | Не учитывалось | Natural light only; cheek/jaw skin ROI | **Gaussian blur + gamma correction**; без AWB |
| **Bias / diversity** | CelebA random sample | FEI (Brazil-focused) | Korean celebrities only | 20 subjects, iPhone 7 | FFHQ + London + SFHQ synthetic; wrist undertone refs ad-hoc |
| **Главная слабость** | 12-class accuracy ~32% | Нет season metrics; HSV fragile | ~60%, Korean bias | **Hair seg**; **template eyeline/brow**; primitive undertone | **Light hair**; **hazel/grey iris**; undertone refs не calibrated; **не seasonal** |
| **Главный урок** | Masked DL + expert data + top-k; 12-class hard | Explicit 4-axis model + product layer | Iterative ablation (RGB→ML→CNN); parsing quality критична | **Единственный** с полным VM pipeline; Lab blend + blurred masks; contrast→season heuristic | **Blur→cluster→CIEDE2000** beats SVM; **HSV for skin / LAB for rest**; **wrist veins** for undertone |

---

# Что можно перенести в face-ai

Сводка идей из всех работ: **что** взять, **откуда**, **зачем** (по смыслу исходной работы — независимо от текущего кода).

---

## Брать

| # | Идея | Откуда | Зачем |
|---|------|--------|-------|
| 1 | **Masked color zones → classifier** | Deep Armocromia, Colorinsight | Классификатор не «видит» фон/одежду; masked RGB дал прирост 20–30% → ~55–60% vs raw RGB heuristics |
| 2 | **Два уровня: 4-season headline + 12-season detail** | Deep Armocromia, AUA | Простой UX (4 сезона) + точность fine-grained (12–16 sub-types); parent season как агрегация sub-type |
| 3 | **Top-k / neighbor seasons в confidence** | Deep Armocromia | При ~55% accuracy top-2 даёт ~81% — соседние сезоны часто семантически близки (Flow Theory) |
| 4 | **Benchmark на Deep Armocromia** | Deep Armocromia | Единственный открытый expert-labeled set (~4920) — объективное сравнение rules vs DL |
| 5 | **4 оси Munsell + шкала 1–5 (explainable)** | AUA | Сезон как undertone / chroma / value / contrast — понятный пользователю breakdown, не black-box label |
| 6 | **Value = 0.7×skin + 0.15×eyes + 0.15×hair** | AUA | «Светлота типажа» — не только кожа: глаза и волосы влияют на perceived depth |
| 7 | **Contrast ratio (L_min+0.05)/(L_max+0.05)** | AUA | Относительный контраст между зонами лица — ключевой признак winter vs summer в их lookup |
| 8 | **Chroma по skin + eyes + lips** | AUA | Chroma одной кожи недостаточен; глаза и губы уточняют soft vs bright sub-types |
| 9 | **Skin palette без red bias (hue 13–24 / trimmed mean)** | AUA | Красные пиксели (губы, блики, тени) смещают undertone; trimmed hue range стабилизирует палитру |
| 10 | **Brightness compensation при плохом свете** | AUA | FEI-освещение занижало value; +20% brightness на swatch уменьшил hue deviation |
| 11 | **Lip 3-tier brightness clusters** | AUA, Colorinsight | AUA: средний RGB губ тёмный из-за теней — кластеры по яркости дают natural lip color; Colorinsight: R/B filter отсекает помаду |
| 12 | **Lookup 16 типов (12 + transitional)** | AUA | Явная таблица для пограничных случаев между sub-types (True Bright/Light/Soft/Deep) |
| 13 | **Product match по ΔE / nearest swatch** | AUA | Связь abstract season → конкретный SKU (foundation к skin, lipstick к lips+palette) |
| 14 | **Contrast makeup shades** | AUA | Рекомендации не только по hue season, но и по уровню контраста образа (high/low contrast makeup) |
| 15 | **Итеративный ablation: RGB → tabular → CNN** | Colorinsight | Документированный путь: L2 ~25% → tabular ML ~25% → CNN ~60%; не гадать, а мерить каждый шаг |
| 16 | **Multi-path inference с fallback** | Colorinsight | Разные зоны (кожа vs губы) — разные сигналы; fallback когда primary path ненадёжен (makeup, parsing fail) |
| 17 | **Reference swatches как sanity-check** | Colorinsight, AUA | Colorinsight: фиксированные RGB anchors + vote; AUA: inverse lookup из scores в swatch grid — визуальная проверка label |
| 18 | **Skin-mask preview для пользователя** | Colorinsight | Доверие к сервису: пользователь видит, какие пиксели пошли в анализ (demo video — главный UX-hook) |
| 19 | **FaRL как face parser** | Colorinsight, Deep Armocromia | FaRL устойчивее альтернатив на поворотах и нестандартной форме; обе работы строят на нём/Facer pipeline |
| 20 | **Сравнение backbone’ов перед выбором** | Colorinsight, Deep Armocromia | Colorinsight: ResNet > MobileNet/EffNet; Deep Armocromia: FaRL64 > ResNeXt50 на 4-class, FaRL16 лучше на 12-class — выбор зависит от granularity |
| 21 | **Virtual makeup render (Lab blend + α)** | Park IMCOM'18 | Eq.(2–3): смешивание foundation/blush/lip в **Lab** с clamp — естественнее RGB multiply; параметр α = intensity |
| 22 | **Gaussian-blurred makeup masks** | Park IMCOM'18 | Blush/eyeshadow через **размытую** mask — gradation без «наклейки»; must-have для try-on |
| 23 | **Cheek/jaw skin ROI для undertone** | Park IMCOM'18 | Щёки/линия челюсти меньше под солнцем — стабильнее skin color, чем mean по всему лицу |
| 24 | **L-contrast между зонами → season brightness** | Park IMCOM'18 | Контраст **L** iris/hair/skin (порог 13) отделяет bright vs muted season — родственно AUA contrast ratio и face-ai `value_contrast_index` |
| 25 | **Expert makeup DB + questionnaire** | Park IMCOM'18 | `(season, occasion, demographics)` → конкретные Lab-цвета и shapes; не только текстовые советы |
| 26 | **User satisfaction rubric (4 axes)** | Park IMCOM'18 | Оценка try-on: color / region size / location / intensity — выявляет слабые продукты (lip, eyeline) |
| 27 | **Landmark-derived makeup zones** | Park IMCOM'18 | Blush/shadow/lip masks из interp на Dlib — лёгкий AR без DL parsing (fallback path) |
| 28 | **Contrast → makeup intensity** | Park IMCOM'18 + AUA | High facial contrast → сильнее lip/line; low → мягче blush (связать `contrast_bucket` с α в try-on) |
| 29 | **Gaussian blur на color ROI перед извлечением** | Colors Matter | Blur на skin mask **до** clustering: accuracy skin **42% → 80%**; убирает поры/блики, оставляет overtone |
| 30 | **Dominant color через X-Means/K-means, не mean** | Colors Matter, Park | Largest cluster center стабильнее mean/median по маске (тени, края маски, highlights) |
| 31 | **CIEDE2000 palette matching** | Colors Matter, AUA | Perceptually uniform nearest swatch; beat SVM+ResNet18 (80% vs 76%); explainable `delta_e_scores` |
| 32 | **HSV для skin, LAB для hair/iris/undertone** | Colors Matter | Разные color spaces под разные признаки — не один LAB everywhere |
| 33 | **Wrist vein undertone (второй upload)** | Colors Matter | Undertone с лица путается (makeup, tan); вены на запястье → Warm/Cool 70–80% |
| 34 | **Hair: k=3 LAB + avg(dominant, mean) distance** | Colors Matter | Два представителя цвета волос → robustнее одного mean при highlights |
| 35 | **Iris: circular mask по Dlib landmarks** | Colors Matter | Проще percentile ring; exclude pupil/sclera; оба глаза → average |
| 36 | **Discrete skin_tone_class как доп. к seasonal** | Colors Matter | 8-class overtone label параллельно 12-season — UX «ваш оттенок foundation» без замены season logic |
| 37 | **CASCo/PERLA как anti-baseline** | Colors Matter | Authors rejected — segmentation quality > готовый skin tone tool |

---

## Не брать

| # | Антипаттерн | Откуда | Почему |
|---|-------------|--------|--------|
| 1 | HSV skin detection без parsing | AUA | Ломается на фоне, плечах, тенях; рекурсивная подстройка порогов явно отклонена авторами |
| 2 | Y-axis split для глаз/губ | AUA | «Много false positives» — Method 1 хуже Method 2 (connected components) |
| 3 | Рекурсивные hue/SV ranges | AUA | Теряется кожа у глаз, захватываются волосы/брови — authors rejected |
| 4 | RGB L2 как **основной** классификатор сезона | Colorinsight | ~20–30% accuracy; systematic failure на autumn |
| 5 | Random flip/augment **при inference** | Colorinsight | Недетерминизм предсказания; ошибка в prod-коде |
| 6 | ResNet weights «как есть» без retrain | Colorinsight | ~750 Korean celebs — неизвестная generalization на другие этnicity/условия |
| 7 | Spring Boot callback hard dependency | Colorinsight | Backend не standalone; результат уходит только через hardcoded localhost callback |
| 8 | ImageJ/Java runtime | AUA | Прототипный стек; не масштабируется как web/mobile API |
| 9 | Генерация 28 800 swatches | AUA | Избыточно для UI; достаточно curated subset на сезон |
| 10 | Только 4 сезона без sub-types | Colorinsight, Park | Теряется nuance (soft/bright/deep/light); serious works идут на 12+ |
| 11 | Undertone = **a > b → cool** | Park IMCOM'18 | Слишком грубо; лучше fused undertone (skin+hair+iris) как в face-ai |
| 12 | Iris: **Hough circle + K-means** | Park IMCOM'18 | Хуже percentile iris ring / parsing eye region |
| 13 | Hair: **Canny по R/G/B** | Park IMCOM'18 | Низкая accuracy (Fig. 11); использовать face parsing hair mask |
| 14 | Eyeline/eyebrow **flat templates** | Park IMCOM'18 | Авторы сами: low satisfaction; нужен spline/warp по brow landmarks |
| 15 | Fixed lip polygon from landmarks | Park IMCOM'18 | Low score «region size»; parsing lip mask или dense lip mesh |
| 16 | **Two-stage hierarchical skin classifier** | Colors Matter | 64% vs 80% flat — иерархия main→sub class не помогла |
| 17 | **SVM+ResNet18 как primary path** | Colors Matter | 76% vs 80% rule-based Delta E — сложнее, не точнее |
| 18 | **Trial-and-error palette refs без calibration** | Colors Matter | Warm/cool vein LAB и skin swatches ad-hoc — нельзя копировать числа 1:1 |
| 19 | **8-class skin scale вместо seasonal taxonomy** | Colors Matter | Overtone ≠ season; использовать как **доп. label**, не замену 12-season |
| 20 | **CASCo library для segmentation** | Colors Matter | Authors explicitly rejected — shadow/hair/background failures |
| 21 | **Cosine similarity для undertone** | Colors Matter | Хуже CIEDE2000; ambiguous green/blue boundary |
| 22 | **~10 img/class eval как production metric** | Colors Matter | Hair/iris/undertone numbers — exploratory only |

---

## Приоритетный roadmap (синтез всех работ)

1. **Краткосрочно:** skin-mask preview; LAB reference anchors; lip brightness filter (AUA + Colorinsight); Munsell 1–5 scores (AUA); cheek/jaw skin sub-mask (Park); **blur + k-means dominant color + CIEDE2000 matcher** (Colors Matter).
2. **Среднесрочно:** optional DL head на masked skin + ensemble с heuristics (Deep Armocromia + Colorinsight); benchmark на Deep Armocromia; top-k в confidence (Deep Armocromia); **makeup_db.json + Lab-blended try-on** с blurred blush/shadow (Park); **optional wrist photo → vein undertone fusion** (Colors Matter).
3. **Долгосрочно:** product match по ΔE (AUA + Colors Matter); questionnaire → style (Park); user study 4 axes (Park); собственный labeled set beyond celeb bias (gap во всех работах); **calibrated reference swatches** (не trial-and-error как Colors Matter).
