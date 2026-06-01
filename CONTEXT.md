# face-ai

Self-hosted платформа анализа внешности по селфи: цветотип (Armocromia), геометрия лица, рекомендации и визуальные try-on-фичи.

## Language

**Seasonal color analysis**:
Определение цветотипа (4/12/16 сезонов) по изолированным цветовым зонам лица — кожа, волосы, глаза, губы.
_Avoid_: color analysis, color typing

**Try-on category**:
Один из трёх типов визуальной примерки: **makeup**, **glasses**, **hairstyle**. Общая модель как в Snapchat/TikTok filters — маска/overlay, tracked к лицу.
_Avoid_: zone (внутреннее имя sub-region), effect, filter

**Photo try-on**:
Примерка на статичном фото по трём **try-on category** (makeup, glasses, hairstyle) на одном слайдере до/после. Базовая реализация; **Live AR try-on** — то же API рендереров на video frames.
_Avoid_: virtual makeup (слишком узко)

**Live AR try-on**:
Photo try-on на потоке камеры: те же три category, face mesh tracking, real-time compositing. Фаза 3 — **развитие** Photo try-on, не отдельный продукт.
_Avoid_: AR (без уточнения)

**Makeup try-on**:
Sub-regions: губы, румяna, тени, брови — CV Lab-blend и/или generative inpaint по parsing masks.
_Avoid_: beauty filter

**Virtual glasses try-on**:
Overlay оправы по face landmarks / mesh; SKU с `overlay_asset` в catalog. Не путать с **glasses recommendation** (текст по форме лица) и **glasses detection** (очки уже на фото).
_Avoid_: frames advice

**Hairstyle try-on**:
Overlay/template или inpaint по hair mask + head mesh — смена стиля и/или цвета. Не путать с извлечением цвета волос для seasonal analysis.
_Avoid_: hair color analysis

**Outfit scanner**:
Оценка совместимости цветов одежды с **уже известным** цветотипом. Primary: второй upload; optional: inline hint на первом фото.
_Avoid_: outfit analysis, look rating

**Product match**:
Подбор SKU из catalog по ΔE к палитре или try-on category.
_Avoid_: recommendations (rule-based текст)

**Product catalog**:
`products.json`: SKU с LAB, `season_tags`, `category` (`lipstick`, `foundation`, `frames`, `hairstyle`, …), опционально `overlay_asset` для glasses/hairstyle templates.
_Avoid_: inventory

**CV renderer** / **Generative renderer**:
Два backend'а try-on. CV — Lab-blend по masks. Generative — Model API (HTTP, URL-configurable).

**Dual pipeline**:
Photo try-on → CV + Generative preview; UI toggle Classic / AI; путь к zone-split hybrid.

**Analysis presentation**:
UI без маркировки AI vs rule-based. Provenance — только debug/API.

## Flagged ambiguities

**Try-on vs Live AR**: один **TryOnEngine**, разный `mode: photo | live`. Live AR не дублирует логику.

## Roadmap priority (resolved 2026-05-23)

1. Photo try-on (makeup + glasses + hairstyle)
2. Outfit scanner
3. Live AR try-on (same three categories)

## Photo try-on scope (resolved 2026-05-23)

**Три category на одном слайдере**; ship all → measure → cut weak sub-regions.

| Category | Sub-regions | Primary renderer |
|----------|-------------|------------------|
| makeup | lips, blush, eyeshadow, brows | CV Lab-blend; generative fallback per zone |
| glasses | frame overlay | Landmark warp + catalog `overlay_asset` |
| hairstyle | style + color | Template/mask overlay; generative for style swap |

**Dual pipeline** сохраняется для makeup + hairstyle; glasses — CV overlay (generative опционально позже).

## Analysis architecture (resolved 2026-05-23)

- CV pipeline **всегда** (parsing, masks, LAB).
- LLM: image + system prompt → JSON — **primary** semantics.
- Rule-based — **fallback** if LLM off/down.
- Try-on colors: LLM season → `makeup_db` / `palettes_16` lookup.

## Product catalog (resolved 2026-05-23)

Curated `products.json`; ΔE match; categories включают makeup SKUs и assets для glasses/hairstyle.

## Outfit scanner (resolved 2026-05-23)

Flow D: отдельный scan + optional inline; gate without prior analyze.

## Demo navigation (resolved 2026-05-23)

Linear wizard: Camera → Analyze → Try-on slider (+ products) → Outfit scan.

## Example dialogue

**Dev:** Live AR — отдельная кодовая база?  
**Expert:** Нет. Тот же TryOnEngine, `mode=live`, face mesh вместо static masks.

**Dev:** Очки в try-on и glasses recommendation — одно и то же?  
**Expert:** Нет. Recommendation — текст «вам подойдут cat-eye»; try-on — PNG оправы на лице из catalog.
