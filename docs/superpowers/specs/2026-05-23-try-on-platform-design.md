# Design: Try-On Platform (Photo → Outfit → Live AR)

> **Status:** Approved 2026-05-23 (grill-with-docs session)  
> **Domain glossary:** `CONTEXT.md`  
> **Supersedes for try-on scope:** PRD v1 out-of-scope items (virtual try-on, product match) — explicitly in scope for demo track.

## Problem

Demo needs a **hard wow effect**: unified visual try-on (makeup, glasses, hairstyle) tied to seasonal color analysis, product recommendations, and outfit compatibility — with LLM-driven semantics and generative rendering alongside deterministic CV.

## Goals

1. **Photo try-on** (phase 1): one before/after slider with three try-on categories.
2. **Outfit scanner** (phase 2): score outfit colors against known season.
3. **Live AR try-on** (phase 3): same `TryOnEngine` on camera frames.
4. **LLM primary analysis**: image + system prompt → structured JSON; rules fallback only.
5. **Dual pipeline**: CV + Generative previews with Classic / AI toggle; path to zone-split hybrid.
6. **Product match**: curated `products.json`, ΔE matching, linked to try-on results.
7. **Demo UX**: linear wizard; no AI vs rule-based labels in mobile UI.

## Non-goals (this spec)

- PostgreSQL product admin (post-demo).
- Full benchmark integration for LLM season labels.
- Live AR in demo v1.
- Tab-based navigation (post-demo).
- Illumination correction, multi-photo intake.

---

## Domain model

See `CONTEXT.md`. Key terms:

| Term | Meaning |
|------|---------|
| **Try-on category** | `makeup` \| `glasses` \| `hairstyle` |
| **TryOnEngine** | Shared compositor; `mode: photo \| live` |
| **Dual pipeline** | CV + Generative outputs per photo try-on |
| **Product catalog** | `products.json` with LAB, season_tags, optional overlay_asset |

**Compositing order:** hairstyle → makeup → glasses (front).

---

## Roadmap phases

```
Phase 1  POST /analyze (LLM + CV) + POST /try-on/photo + products + demo wizard
Phase 2  POST /outfit/scan + inline outfit hint on /analyze
Phase 3  TryOnEngine mode=live (face mesh, real-time)
```

---

## Architecture

### High-level flow (demo wizard)

```
[Camera]
  → POST /analyze (face photo)
  → [Analyzing animation]
  → Result screen (season, palette — unified UI, no AI labels)
  → POST /try-on/photo
  → Slider: original | CV | Generative (+ category toggles)
  → GET /products/match (cards under slider)
  → POST /outfit/scan (optional 2nd photo OR inline hint from step 1)
  → Score ring + LLM narrative
```

### Analysis pipeline (Mode B)

```
Image bytes
  → IntakeAndQualityGate
  → LandmarkService
  → ParsingRegistry → masks (always)
  → ColorFeatureExtractor → LAB, palette (always)
  → LLMAnalysisAdapter (primary)
       image + system_prompt → JSON (season, recommendations, …)
  → IF llm unavailable OR FACE_AI_LLM_ENABLED=false:
       SeasonClassifierEnsemble (Munsell/Park/swatch/wrist) — fallback
  → GeometryAnalyzer
  → Merge LLM JSON into AnalysisResponse (schema-validated)
  → Optional: inline outfit hint (non-face mask sufficient)
```

**Color authority for try-on:** LLM season → lookup in `makeup_db.json` / `palettes_16.json`. CV palette used for sanity check and fallback when LLM season invalid.

**UI policy:** Mobile shows unified result. `classifier_contributors`, `seasonal_method` remain in API/debug only.

### TryOnEngine (new)

```python
class TryOnEngine:
    def render_photo(
        self,
        image_bgr,
        parsing_result,
        landmarks,
        season_twelve: str,
        *,
        categories: list[TryOnCategory],  # makeup, glasses, hairstyle
        product_skus: dict[str, str] | None,  # category → sku
        mode_generative: bool = True,
    ) -> TryOnPhotoResult: ...

    def render_live_frame(...):  # Phase 3 stub
        ...
```

**Sub-renderers:**

| Renderer | Category | Photo (P1) | Live (P3) |
|----------|----------|------------|-----------|
| `MakeupRenderer` | makeup | CV Lab-blend (lips, blush, shadow, brows) + Generative inpaint | CV real-time on mesh |
| `GlassesRenderer` | glasses | Landmark warp + catalog `overlay_asset` | Real-time warp |
| `HairstyleRenderer` | hairstyle | Template/mask overlay + Generative style swap | Recolor live; style swap on tap → Model API |

**Dual pipeline response:**

```json
{
  "original_b64": "...",
  "cv": {
    "composite_b64": "...",
    "categories": {
      "makeup": { "zones": ["lips", "blush"], "renderer": "cv" },
      "glasses": { "sku": "frame-001", "renderer": "cv" },
      "hairstyle": { "renderer": "cv", "type": "recolor" }
    }
  },
  "generative": {
    "composite_b64": "...",
    "categories": { "...": { "renderer": "generative" } }
  },
  "active_mode": "cv"
}
```

Evolution to **zone-split hybrid:** orchestrator assigns renderer per zone without breaking this schema (`active_mode: "hybrid"`).

### Generative backend

- **Port:** `GenerativeModelAPI.render(image, mask, prompt) → b64`
- **Config:** `FACE_AI_GENERATIVE_API_URL`, `FACE_AI_GENERATIVE_API_KEY`
- External vs self-hosted: same HTTP interface; only URL differs.
- `FACE_AI_GENERATIVE_API_URL=none` → generative branch omitted; CV-only, no error.

Prompts built from LLM season + category + optional product SKU metadata.

### LLM adapter

- **Port:** `LLMAnalysisAdapter.analyze(image_b64, system_prompt) → dict`
- **Config:** `FACE_AI_LLM_API_URL`, `FACE_AI_LLM_ENABLED`
- Output validated against JSON Schema (subset of `/analyze` or full merge).
- On validation failure: retry once → fallback to rule-based season.

System prompt includes: Armocromia taxonomy, required JSON shape, Russian recommendation tone.

### Product catalog

**File:** `data/products.json`

```json
{
  "sku": "mac-ruby-woo",
  "brand": "MAC",
  "name": "Ruby Woo",
  "category": "lipstick",
  "lab": [50.1, 65.2, 15.3],
  "season_tags": ["true_winter", "bright_winter"],
  "overlay_asset": null
}
```

Categories: `lipstick`, `foundation`, `blush`, `frames`, `hairstyle`, …

**Match:** `ProductMatcher.match(season_twelve, category, target_lab?, top_k=3)` using `delta_e_cie2000`.

Priority: match within LLM 12-season → expand to parent 4-season if empty.

Frames/hairstyle entries include `overlay_asset` (PNG with alpha) for warp renderers.

### Outfit scanner

**Endpoint:** `POST /outfit/scan`

**Input:** multipart `file` + `season_twelve` (from mobile session) or full analyze context.

**Gate:** 400 if no prior analyze in session (mobile stores last `AnalysisResponse`).

**CV path:**

1. Parsing on outfit photo (or reuse face parsing if same session photo).
2. Non-face mask → k-means dominant colors (3–5 clusters).
3. ΔE each cluster vs season palette from `analysis_palette.season_reference`.
4. Aggregate → `compatibility_score` 0–100.

**LLM path (optional):** image + prompt → `{ score, issues[], suggestions[] }` merged with CV score (average or LLM narrative only).

**Inline hint (Flow D):** During `/analyze`, if non-face pixel ratio > threshold, include `metrics.outfit_hint: { score, visible: true }`.

---

## API changes

### Extended `POST /analyze`

- Runs CV pipeline always.
- Calls LLM adapter when enabled; merges into response.
- Falls back to ensemble on LLM failure.
- New optional field: `metrics.outfit_hint` (inline).

No breaking changes to existing required fields.

### New `POST /try-on/photo`

**Input:**

- `file` (image)
- `meta_json`: `{ "season_twelve", "categories": ["makeup","glasses","hairstyle"], "product_skus": {...}, "generative": true }`
- Or: `analyze_run_id` to reuse masks from debug dump (optimization).

**Output:** `TryOnPhotoResult` (see schema above).

Runs parsing if not cached; composes all requested categories.

### New `POST /outfit/scan`

**Input:** `file`, `season_twelve`, optional `session_id`.

**Output:**

```json
{
  "compatibility_score": 78,
  "dominant_colors": [{ "hex", "lab", "season_delta_e" }],
  "issues": ["..."],
  "suggestions": ["..."],
  "product_alternatives": [{ "sku", "name", "match_pct" }]
}
```

### New `GET /products/match`

Query: `season_twelve`, `category`, optional `target_lab`, `top_k`.

Returns ranked SKU list with ΔE and match percentage.

---

## Mobile (face-ai-mobile)

### Phase 1 screens (linear wizard)

1. `CameraCaptureScreen` (existing)
2. `AnalyzingScreen` (new — skeleton animation)
3. `AnalysisResultScreen` (extend — season, palette strip)
4. `PhotoTryOnScreen` (new — slider, Classic/AI toggle, category chips, product cards)
5. `OutfitScanScreen` (new — second capture or skip)

### PhotoTryOnScreen UX

- Horizontal before/after slider (full width).
- Toggle: Classic | AI (maps to cv / generative composite).
- Chips: Makeup | Glasses | Hairstyle (toggle categories on/off → re-request or local composite if pre-rendered).
- Product cards scroll below matched category.

### State

- `analysisResultStore` — last `/analyze` JSON + season.
- AsyncStorage session for outfit gate.

---

## Data files (new)

| File | Purpose |
|------|---------|
| `data/products.json` | SKU catalog (~30–100 demo entries) |
| `data/makeup_db.json` | Lab colors per 12-season × zone |
| `data/try_on_prompts/` | System prompt templates for LLM + generative |
| `static/overlays/frames/` | PNG assets referenced by products.json |
| `static/overlays/hairstyles/` | Template hair overlays |

---

## Configuration

| Env | Default | Description |
|-----|---------|-------------|
| `FACE_AI_LLM_ENABLED` | `true` | Disable → rules fallback |
| `FACE_AI_LLM_API_URL` | — | LLM HTTP endpoint |
| `FACE_AI_GENERATIVE_API_URL` | `none` | Model API; `none` = CV only |
| `FACE_AI_TRYON_CATEGORIES` | `makeup,glasses,hairstyle` | Default categories |
| `FACE_AI_OUTFIT_INLINE_MIN_RATIO` | `0.15` | Min non-face pixels for inline hint |

---

## Error handling

| Failure | Behavior |
|---------|----------|
| LLM down | Silent fallback to rules; same UI |
| Generative down | Hide AI toggle; CV slider only |
| Glasses asset missing | Skip glasses category; note in debug |
| Outfit scan without analyze | 400 + mobile redirect to camera |
| Parsing fail | Try-on blocked; show photo_quality issues |

---

## Testing

### Backend unit tests

1. `MakeupRenderer` — Lab blend on fixture image + mask → output shape unchanged.
2. `GlassesRenderer` — landmark warp places frame within tolerance.
3. `ProductMatcher` — known LAB → expected top SKU.
4. `OutfitScanner` — synthetic non-face colors → expected score band.
5. `LLMAnalysisAdapter` — mock HTTP → schema-valid merge.
6. `TryOnEngine` — compositing order (hairstyle under glasses).

### Integration (manual demo)

- End-to-end wizard on 3 fixture faces.
- LLM disabled → rules fallback invisible to user.
- Generative disabled → Classic-only slider.

### Not in scope

- Live AR frame rate tests (phase 3).
- LLM accuracy benchmark on Deep Armocromia.

---

## Implementation order (suggested)

1. `products.json` + `ProductMatcher` + `GET /products/match`
2. `LLMAnalysisAdapter` + merge into `/analyze`
3. `MakeupRenderer` (CV) + `POST /try-on/photo` (makeup only)
4. `GenerativeModelAPI` + dual pipeline toggle
5. `GlassesRenderer` + `HairstyleRenderer`
6. Mobile wizard screens 2–4
7. `OutfitScanner` + screen 5 + inline hint
8. Live AR stub + phase 3 PRD slice

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| LLM JSON drift | Strict schema validation + fallback |
| Generative latency | Async parallel CV + gen; show CV first |
| Glasses/hairstyle quality | Curated overlay assets; generative for hair style |
| Scope creep | ship-all-measure-cut per category |
| PRD interpretability conflict | Documented; debug-only provenance |

---

## Open items (post-spec)

- Choose concrete LLM provider URL for demo.
- Choose Model API provider for generative inpaint.
- Curate initial 30 SKU + 5 frame overlays + 5 hairstyle templates.
- ADR candidate: unified TryOnEngine (photo/live) — write if team wants permanent record.
