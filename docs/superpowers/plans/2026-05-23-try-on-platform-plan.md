# Implementation Plan: Try-On Platform

> **Design spec:** `docs/superpowers/specs/2026-05-23-try-on-platform-design.md`  
> **Glossary:** `CONTEXT.md`  
> **Target:** demo-ready Phase 1 + Phase 2; Phase 3 stub only

## Overview

| Phase | Deliverable | Demo-critical |
|-------|-------------|---------------|
| **P0** | Foundation: catalog, schemas, config | ✅ |
| **P1** | Photo try-on + LLM analyze + mobile wizard | ✅ |
| **P2** | Outfit scanner + inline hint | ✅ |
| **P3** | Live AR stub (`TryOnEngine.render_live_frame`) | ❌ post-demo |

**Estimated calendar:** P0–P1 ≈ 2–3 weeks; P2 ≈ 3–5 days; P3 stub ≈ 1 day.

**Parallel tracks:** Backend (`face-ai`) and Mobile (`face-ai-mobile`) can split after P0 schemas land.

---

## Vertical slice 0 — Foundation (2–3 days)

### 0.1 Config & env

**Files:** `app/config.py`, `.env.example`

Add settings:

```
llm_enabled: bool = True
llm_api_url: str = ""
llm_api_key: str = ""
generative_api_url: str = "none"
generative_api_key: str = ""
tryon_default_categories: str = "makeup,glasses,hairstyle"
outfit_inline_min_ratio: float = 0.15
```

**Done when:** `/health` reports `llm_available`, `generative_available` booleans.

### 0.2 Pydantic schemas

**Files:**

- `app/schemas/try_on.py` — `TryOnPhotoResult`, `TryOnCategory`, category metadata
- `app/schemas/products.py` — `ProductSku`, `ProductMatchResult`
- `app/schemas/outfit.py` — `OutfitScanResult`, `OutfitHint`
- Extend `app/schemas/analysis.py` — optional `metrics.outfit_hint`
- Export `app/schemas/try_on_contract.json` for mobile

**Done when:** models validate spec examples; JSON schema exported.

### 0.3 Product catalog & matcher

**Files:**

- `data/products.json` — seed 30 SKU (10 lipstick, 5 blush, 5 foundation, 5 frames w/ `overlay_asset`, 5 hairstyle w/ `overlay_asset`)
- `data/makeup_db.json` — Lab colors per 12-season × zone (lips, blush, shadow, brows, hair)
- `app/services/product_catalog.py` — load + validate
- `app/services/product_matcher.py` — ΔE rank via `app/backends/color/delta_e.py`
- `app/main.py` — `GET /products/match`
- `tests/test_product_matcher.py`

**Done when:** `GET /products/match?season_twelve=light_summer&category=lipstick&top_k=3` returns ranked SKUs.

### 0.4 Overlay assets (minimal)

**Files:**

- `static/overlays/frames/*.png` — 5 transparent PNGs
- `static/overlays/hairstyles/*.png` — 5 hair templates
- Reference paths in `products.json`

**Done when:** assets loadable via `GET /static/overlays/...` or FileResponse.

---

## Vertical slice 1 — LLM analysis (3–4 days)

### 1.1 LLM adapter port

**Files:**

- `app/backends/llm/adapter.py` — `LLMAnalysisAdapter.analyze(image_bytes, prompt) -> dict`
- `app/backends/llm/prompts/analysis_system.txt` — Armocromia taxonomy + JSON shape
- `app/backends/llm/schema_merge.py` — merge LLM dict into `AnalysisResponse` fields
- `tests/test_llm_merge.py` — mock adapter, invalid JSON → fallback

**Contract:**

- Input: base64 image + system prompt
- Output: `{ seasonal_twelve, seasonal_guess, recommendations[], undertone_hint, ... }`
- Validate with Pydantic; retry 1×; on fail → use CV/rules season only

### 1.2 Orchestrator integration

**Files:**

- `app/pipeline/orchestrator.py` — after `ColorFeatureExtractor`, call LLM if enabled
- `app/services/analysis_service.py` — pass through
- Keep `SeasonClassifierEnsemble` as fallback path unchanged

**Merge rules:**

| Field | Source |
|-------|--------|
| `metrics.seasonal.*` (labels) | LLM primary |
| `metrics.geometry`, masks, palette | CV always |
| `recommendations[]` | LLM if present, else `rules.build_recommendations` |
| `classifier_contributors` | rules output in debug only when fallback used |

**Done when:**

- `FACE_AI_LLM_ENABLED=false` → identical behavior to today
- Mock LLM → season from JSON, no UI-facing AI label

### 1.3 Prompt templates for try-on/generative

**Files:** `data/try_on_prompts/generative_makeup.txt`, `generative_hairstyle.txt`

---

## Vertical slice 2 — CV Photo try-on (4–5 days)

### 2.1 Makeup renderer (CV)

**Files:**

- `app/backends/try_on/lab_blend.py` — Park Eq. Lab blend + α from `contrast_bucket`
- `app/backends/try_on/makeup_renderer.py` — lips, blush, shadow, brows zones
- Reuse: `app/pipeline/mask_postprocess.py`, `app/services/mask_geometry.py`

**Done when:** unit test — fixture BGR + masks → changed pixels only inside lip mask.

### 2.2 Glasses renderer

**Files:**

- `app/backends/try_on/glasses_renderer.py`
- Warp `overlay_asset` to nose bridge + temple landmarks (MediaPipe indices)
- Scale from `glasses_ratio` / face width heuristic

**Done when:** test — frame bbox overlaps expected landmark region.

### 2.3 Hairstyle renderer (CV path)

**Files:**

- `app/backends/try_on/hairstyle_renderer.py`
- Hair recolor via `hair_mask` Lab blend
- Template overlay: warp hairstyle PNG to head top + ear landmarks

**Done when:** composite preserves non-hair pixels.

### 2.4 TryOnEngine compositor

**Files:**

- `app/backends/try_on/engine.py` — `TryOnEngine.render_photo()`
- Compositing order: hairstyle → makeup → glasses
- `app/backends/try_on/types.py` — shared types
- `tests/test_try_on_engine.py` — order + category skip

### 2.5 API endpoint

**Files:**

- `app/main.py` — `POST /try-on/photo`
- `app/services/try_on_service.py` — decode image, run parsing, engine, base64 out
- Accept `meta_json`: `{ season_twelve, categories[], product_skus{}, generative: bool }`

**Done when:** Postman curl returns `{ original_b64, cv: { composite_b64, categories } }` with all 3 categories.

---

## Vertical slice 3 — Generative dual pipeline (2–3 days)

### 3.1 Model API adapter

**Files:**

- `app/backends/try_on/generative_api.py` — HTTP client, mask-guided inpaint
- Configurable URL/key; timeout; graceful `None` on failure

**Request shape (adapter-internal, normalize any provider):**

```json
{ "image_b64", "mask_b64", "prompt", "negative_prompt" }
```

### 3.2 Dual pipeline in engine

**Files:** extend `TryOnEngine.render_photo()`

- Run CV composite first (sync)
- If `generative_api_url != none` and `meta.generative`: parallel async generative pass
- Return both branches

**Done when:** response includes `cv` and `generative` keys; generative omitted when URL=none.

---

## Vertical slice 4 — Mobile demo wizard (4–5 days)

### 4.1 Navigation & types

**Files:**

- `navigation/types.ts` — add `Analyzing`, `PhotoTryOn`, `OutfitScan` routes + params
- `navigation/AppNavigator.tsx` — linear stack order

**Flow:**

```
Home → PhotoSource → CameraCapture → Analyzing → AnalysisResult
  → PhotoTryOn → OutfitScan
```

### 4.2 Analyzing screen

**Files:**

- `screens/AnalyzingScreen.tsx` — animation while upload runs
- Refactor `uploadAnalyze.ts` — navigate to Analyzing before fetch, then Result

### 4.3 Extend AnalysisResult

**Files:**

- `screens/AnalysisResultScreen.tsx` — palette strip from `metrics.analysis_palette`
- CTA button «Примерить» → `PhotoTryOn`
- Optional inline outfit hint badge if `metrics.outfit_hint.visible`

### 4.4 PhotoTryOn screen

**Files:**

- `screens/PhotoTryOnScreen.tsx`
- `services/uploadTryOnPhoto.ts` — `POST /try-on/photo`
- `services/fetchProductMatch.ts` — `GET /products/match`
- `components/BeforeAfterSlider.tsx` — pan gesture slider
- Classic | AI toggle; category chips (re-fetch or toggle pre-rendered layers)

**Done when:** demo path works on Expo Go against LAN backend.

### 4.5 Product cards

**Files:** `components/ProductCard.tsx` — brand, name, match %

---

## Vertical slice 5 — Outfit scanner (3–4 days)

### 5.1 Backend scanner

**Files:**

- `app/services/outfit_scanner.py` — non-face k-means, ΔE vs season palette
- `app/main.py` — `POST /outfit/scan`
- Optional LLM narrative via same adapter (separate prompt)
- `tests/test_outfit_scanner.py`

### 5.2 Inline hint on analyze

**Files:**

- `app/pipeline/orchestrator.py` — if non-face ratio > threshold, attach `metrics.outfit_hint`

### 5.3 Mobile OutfitScan screen

**Files:**

- `screens/OutfitScanScreen.tsx` — second photo capture or skip
- Gate: no `analysisResultStore` → redirect Home
- Score ring UI + suggestions list

**Done when:** wizard end-to-end: selfie → try-on → outfit score.

---

## Vertical slice 6 — Phase 3 stub (1 day)

**Files:**

- `app/backends/try_on/engine.py` — `render_live_frame()` raises `NotImplementedError` with docstring pointing to Phase 3
- `docs/superpowers/specs/2026-05-23-live-ar-slice.md` — optional thin follow-up spec

**Done when:** interface exists; mobile does not call it yet.

---

## Task checklist (copy to tracker)

### Backend

- [ ] P0: config + health flags
- [ ] P0: schemas + JSON contract
- [ ] P0: `products.json` + `makeup_db.json` + matcher + tests
- [ ] P0: overlay PNG assets
- [ ] P1: LLM adapter + orchestrator merge + tests
- [ ] P1: MakeupRenderer CV
- [ ] P1: GlassesRenderer
- [ ] P1: HairstyleRenderer CV
- [ ] P1: TryOnEngine + `/try-on/photo`
- [ ] P1: GenerativeModelAPI + dual pipeline
- [ ] P2: OutfitScanner + `/outfit/scan`
- [ ] P2: inline outfit hint on `/analyze`
- [ ] P3: `render_live_frame` stub

### Mobile

- [ ] P1: navigator + AnalyzingScreen
- [ ] P1: AnalysisResult palette + CTA
- [ ] P1: PhotoTryOnScreen + slider + toggle
- [ ] P1: product cards + API clients
- [ ] P2: OutfitScanScreen + gate

### Demo prep

- [ ] Pick LLM API URL + key in `.env`
- [ ] Pick generative Model API URL (or CV-only fallback)
- [ ] 3 test faces rehearsed on projector Wi‑Fi
- [ ] Fallback rehearsed: LLM off + generative off

---

## Suggested execution order (single dev)

```
Week 1
  Mon-Tue   P0 foundation (catalog, schemas, /products/match)
  Wed-Fri   P1 LLM adapter + orchestrator merge

Week 2
  Mon-Wed   P1 CV try-on renderers + TryOnEngine + /try-on/photo
  Thu-Fri   P1 Generative adapter + dual pipeline

Week 3
  Mon-Wed   P1 mobile wizard (Analyzing → Result → PhotoTryOn)
  Thu-Fri   P2 outfit scanner backend + mobile screen

Buffer     polish, demo rehearsal, cut weak try-on zones
```

---

## Acceptance criteria (demo-ready)

1. Linear wizard completes without crash on 3 fixture photos.
2. Try-on slider shows visible makeup + glasses + hairstyle change (CV branch minimum).
3. Classic / AI toggle works when generative API configured; hidden when not.
4. At least 3 product cards appear under try-on with match %.
5. Outfit scan returns score 0–100 after analyze; blocked before analyze.
6. LLM outage: app still returns season + try-on (rules fallback), indistinguishable in UI.
7. No «AI» / «rule-based» labels anywhere in mobile.

---

## Open decisions (blockers — resolve before slice 1.1)

| # | Decision | Default if no answer |
|---|----------|----------------------|
| 1 | LLM provider URL | OpenAI-compatible `/v1/chat/completions` multimodal |
| 2 | Generative API URL | CV-only demo (`none`) until provider chosen |
| 3 | Initial SKU curation | Agent seeds 30 entries from public shade data |

---

## Files touched (summary)

| Area | New | Modified |
|------|-----|----------|
| Backend core | `app/backends/try_on/*`, `app/backends/llm/*`, `app/services/product_*.py`, `app/services/outfit_scanner.py`, `app/services/try_on_service.py` | `app/main.py`, `app/config.py`, `app/pipeline/orchestrator.py`, `app/schemas/*` |
| Data | `data/products.json`, `data/makeup_db.json`, `data/try_on_prompts/*`, `static/overlays/**` | — |
| Tests | `tests/test_product_matcher.py`, `tests/test_try_on_*.py`, `tests/test_llm_merge.py`, `tests/test_outfit_scanner.py` | — |
| Mobile | `screens/AnalyzingScreen.tsx`, `PhotoTryOnScreen.tsx`, `OutfitScanScreen.tsx`, `components/*`, `services/*` | `AppNavigator.tsx`, `AnalysisResultScreen.tsx`, `navigation/types.ts` |
