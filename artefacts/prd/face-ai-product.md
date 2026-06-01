# PRD: face-ai — Face Shape & Seasonal Color Analysis Platform

> **Scope:** весь продукт face-ai — FastAPI backend (`/analyze`), rule-based seasonal color pipeline, geometry/recommendations, debug UI, benchmark/eval, мобильный клиент face-ai-mobile, и интеграция DL-классификатора после обучения. Обучение DL вынесено в подпроект `season-dl/` (отдельный PRD: `season-dl/artefacts/prd/deep-armocromia-dl-classifier.md`).

## Problem Statement

Пользователям нужен доступный, воспроизводимый анализ внешности по одному селфи: форма лица, цветотип (Armocromia / seasonal color analysis), подтон кожи, контрастность и практические рекомендации по стилю, макияжу и аксессуарам. Профессиональный color analysis дорог, субъективен и требует очной консультации; коммерческие AI-сервисы (Dressika, HueCheck, SeasonAI и др.) дают 12-season результат, но часто непрозрачны по методологии, зависят от облака и не объединяют геометрию лица с цветотипом в одном self-hosted API.

face-ai уже реализует rule-based пайплайн v0.2: face parsing, LAB-признаки по зонам, Munsell 16-type lookup, ensemble (Munsell + Park IMCOM'18 + swatch vote + optional wrist undertone), геометрию лица и rule-based рекомендации. Expo-приложение face-ai-mobile отправляет фото на `POST /analyze` и показывает результат. Однако на публичном бенчмарке Deep Armocromia rule-based пайплайн отстаёт от опубликованных DL-базлайнов: ~40% 4-season top-1 vs ~55% в статье, ~20% 12-season top-1 vs ~32%. Крупнейший gap — отсутствие обученного DL-классификатора и его интеграции в ensemble. Дополнительные пробелы: мобильное приложение вызывает несуществующий endpoint `/contour/final`; нет illumination correction и multi-photo intake; virtual makeup и product match только в roadmap.

Пользователям мобильного приложения и API-потребителям нужна более высокая точность на соседних сезонах (Autumn↔Winter, Spring↔Summer, adjacent 12 sub-types) при сохранении интерпретируемости rule-based contributors. Продукт должен возвращать прозрачный confidence, top-k соседних сезонов и палитру лица, чтобы пользователь понимал, когда метка неоднозначна.

## Solution

Построить end-to-end платформу **face-ai**: self-hosted FastAPI backend с модульным CV-пайплайном (landmarks → parsing → color/contrast → season classification → geometry → recommendations), стабильным JSON-контрактом для клиентов, debug UI для стилистов и инженеров, eval harness на Deep Armocromia для измеримого прогресса, и Expo mobile client как основной UX. Season classification эволюционирует от rules-only к **fused ensemble** (Munsell + Park + swatch + wrist + DL), где DL обучается в `season-dl/` и подключается как опциональный backend без breaking changes в API. Мобильный клиент дополняется недостающими endpoints и richer отображением top-k / confidence / палитры.

## User Stories

### End users (mobile & web)

1. As a mobile app user, I want to take or pick a selfie and receive my seasonal color type within a few seconds, so that I can trust makeup and wardrobe suggestions.
2. As a mobile app user, I want to see my face shape (oval, round, square, heart, etc.) alongside my colorotype, so that hairstyle and glasses recommendations feel personalized.
3. As a mobile app user, I want to see my top-3 possible 12 sub-types when the model is uncertain, so that I can explore neighbor palettes instead of receiving a wrong definitive label.
4. As a mobile app user, I want a visual palette of my skin, hair, eyes, and lips extracted from my photo, so that I understand what the system "sees" in my coloring.
5. As a mobile app user, I want season reference swatches shown next to my face palette, so that I can compare my colors to my assigned season anchors.
6. As a mobile app user, I want practical recommendations for glasses, hair, makeup, clothing, and jewelry based on my geometry and season, so that the app feels actionable not just diagnostic.
7. As a mobile app user, I want clear feedback when my photo quality is too poor (blur, exposure, face too small), so that I know to retake the photo instead of getting a wrong result.
8. As a mobile app user, I want analysis to work on my phone over Wi‑Fi to a self-hosted backend, so that my face photo stays on infrastructure I control.
9. As a mobile app user, I want a face contour overlay on my photo for visual confirmation of geometry analysis, so that I trust the face-shape result.
10. As a mobile app user, I want undertone (warm/cool/neutral) explained in plain language, so that I understand why a season was assigned.
11. As a user with makeup-heavy photos, I want the system to sample lips and eyes reliably despite cosmetics, so that chroma and contrast are not skewed.
12. As a user with fair cool skin and ash hair, I want undertone fusion to avoid misclassifying me as warm spring, so that light summer/cool palettes are suggested correctly.
13. As a user with non-ideal lighting, I want masked color zones (face regions isolated from background), so that background color does not dominate season predictions.
14. As a user with glasses, I want the system to detect glasses and adjust parsing/confidence notes, so that I understand when frames affect color reads.
15. As a privacy-conscious user, I want no third-party seasonal color API calls, so that my biometric data stays local to my deployment.

### Stylists & debug UI users

16. As a stylist using the debug UI, I want to upload face and optional wrist photos with backend overrides, so that I can test different parsing and classifier configurations.
17. As a stylist, I want to see which classifiers contributed to my season label (Munsell, Park, swatch, wrist, DL), so that I can explain results to clients.
18. As a stylist, I want debug artifacts (masks, seasonal JSON, contour image) saved per run, so that I can inspect failure modes visually.
19. As a stylist, I want Munsell axis scores (undertone, chroma, value, contrast) on a 1–5 scale, so that I can walk clients through the reasoning.
20. As a stylist, I want value contrast index (VCI) and contrast bucket (low/medium/high), so that I can discuss contrast-appropriate makeup and clothing.

### API consumers & integrators

21. As an API consumer, I want a stable `POST /analyze` response schema with JSON Schema at `/schema.json`, so that clients can validate responses programmatically.
22. As an API consumer, I want `metrics.seasonal` with 4-season, 12-season, and 16 Munsell-type labels plus confidences and top-k, so that downstream apps can render rich seasonal UX.
23. As an API consumer, I want `metrics.analysis_palette` with face and season_reference swatches (RGB, hex, LAB), so that I can build palette visualization without re-deriving colors.
24. As an API consumer, I want `metrics.seasonal.classifier_contributors` including per-backend guesses and confidences, so that I can show provenance in UI.
25. As an API consumer, I want optional `wrist_file` upload for vein undertone prior, so that ambiguous face-only undertone can be disambiguated.
26. As an API consumer, I want `meta_json.backends` overrides (parsing backend, skin color method, season classifier), so that I can A/B test configurations without redeploying.
27. As an API consumer, I want no breaking changes when DL is added to ensemble, so that face-ai-mobile continues working without modification.
28. As an API consumer, I want `GET /health` reporting parsing backend availability and DL model load status, so that deployment issues are visible before user-facing failures.
29. As an API consumer, I want `photo_quality.passes_gate` and `issues[]` when analysis is blocked or degraded, so that clients can prompt retakes appropriately.
30. As an API consumer, I want mask preview images (base64) in the response, so that I can show parsing quality to users.

### Backend & pipeline engineers

31. As a backend engineer, I want a modular pipeline orchestrator with clear stage boundaries, so that each step can be tested and swapped independently.
32. As a backend engineer, I want a parsing registry with fallback chain (BiSeNet → FaRL → landmark fallback), so that analysis degrades gracefully when a backend is unavailable.
33. As a backend engineer, I want configurable season classifier via environment (`FACE_AI_SEASON_CLASSIFIER=munsell_lookup|park|ensemble|deep_armocromia_dl`), so that deployments can tune behavior.
34. As a backend engineer, I want ensemble fusion to accept DL probability vectors alongside Munsell/Park/swatch/wrist, so that learned and rule-based signals combine without breaking contracts.
35. As a backend engineer, I want live inference preprocessing to match DL training preprocessing (mask mode, resize, background), so that train/serve skew is minimized.
36. As a backend engineer, I want lazy DL weight loading on first request, so that cold start without GPU does not block API startup.
37. As a backend engineer, I want clear errors when DL weights are missing but backend is requested, so that misconfiguration is diagnosable from `/health`.
38. As a backend engineer, I want a dedicated contour endpoint or equivalent artifact URL for mobile overlay, so that face-ai-mobile's contour feature works.
39. As a backend engineer, I want debug dump service writing numbered artifacts per run, so that support can reproduce issues from production-like runs.
40. As a backend engineer, I want production dependencies separated from training dependencies (PyTorch in optional requirements), so that inference stays lightweight when DL is disabled.

### ML engineers & data scientists

41. As an ML engineer, I want a public benchmark harness on Deep Armocromia (912-image test split), so that every pipeline change is measurable against paper baselines.
42. As an ML engineer, I want eval scripts reporting 4-season top-1/top-2 and 12-season top-1/top-2/top-3 plus confusion matrices, so that I can compare rules-only, DL-only, and fused methods.
43. As an ML engineer, I want mistake JSON listing misclassified samples with paths and predicted vs ground truth, so that I can inspect failure modes visually.
44. As an ML engineer, I want the dataset loader mapping Italian folder labels to face-ai 4/12 season IDs, so that labels stay consistent across rules eval and DL training.
45. As an ML engineer, I want hierarchical DL classification (4 seasons → 3 sub-types each) trained in `season-dl/`, so that the model respects Armocromia Flow Theory structure.
46. As an ML engineer, I want letterbox/pad resize instead of aspect-ratio squash in DL preprocessing, so that face geometry is not distorted before training.
47. As an ML engineer, I want 5-fold cross-validation during DL training, so that generalization is estimated without peeking at the test set.
48. As an ML engineer, I want optional fusion weight tuning on a validation fold, so that ensemble weights are data-driven rather than hand-tuned.
49. As an ML engineer, I want a flat 12-class DL head as an ablation baseline, so that hierarchical training benefit is verifiable.
50. As a data scientist, I want preprocessing cache manifests tracking config signature, so that stale caches are detectable when preprocessing changes.

### Mobile developers

51. As a mobile developer, I want a typed parse layer mapping `/analyze` JSON to UI models (face shape RU labels, season, recommendations), so that API schema evolution is isolated from screens.
52. As a mobile developer, I want configurable backend URL for LAN deployment, so that Expo Go can reach the dev server on the same network.
53. As a mobile developer, I want the backend to return `seasonal_method` indicating when DL contributed, so that the app can show an "AI-enhanced" badge if desired.
54. As a mobile developer, I want optional wrist photo upload in a future app version, so that undertone disambiguation is available to power users.
55. As a mobile developer, I want top-k seasons and confidence rendered in AnalysisResultScreen, so that uncertain results are communicated honestly.

### QA & DevOps

56. As a QA engineer, I want pytest coverage for season backends, ensemble fusion, palette building, and preprocess helpers, so that regressions are caught in CI.
57. As a QA engineer, I want smoke eval on a small Deep Armocromia subset that skips gracefully when dataset is absent, so that CI does not require 5 GB of images.
58. As a DevOps engineer, I want `/health` and version info for rules and parsing backends, so that deployments are observable.
59. As a support engineer, I want Russian disclaimer in API response, so that legal/expectation management is consistent across clients.

### Product & research

60. As a product owner, I want success metrics defined for season accuracy (rules baseline documented; fused target ≥58% 4-season, ≥38% 12-season top-1 on test), so that we know when DL integration is ready.
61. As a product owner, I want the fused ensemble to beat both rules-only and DL-only on the full test set, so that DL investment delivers measurable user value.
62. As a product owner, I want virtual makeup and product matching explicitly out of scope for v1, so that the team stays focused on classification accuracy first.
63. As a product owner, I want illumination correction and multi-photo intake deferred, so that v1 ships on existing single-photo flow.
64. As a researcher, I want benchmark results saved in versioned eval artifacts, so that progress against Deep Armocromia paper numbers is auditable.
65. As a researcher, I want alignment with academic best practices (Munsell axes, Park undertone, Colors Matter ΔE skin clustering) documented in artefacts, so that design choices are traceable to literature.

## Implementation Decisions

### Deep modules (testable in isolation)

1. **AnalysisOrchestrator** — coordinates intake → landmarks → quality gate → parsing → mask postprocess → color/contrast → season → geometry → recommendations → response assembly; single entry `analyze_bgr()`.
2. **IntakeAndQualityGate** — decode image bytes, enforce sharpness/exposure/face coverage thresholds; returns `photo_quality` block and optional early exit.
3. **LandmarkService** — MediaPipe (default) or dlib81; bbox + dense landmarks for parsing fallback and geometry ratios.
4. **ParsingRegistry** — pluggable backends (BiSeNet ONNX, FaRL, SegFace) with ordered fallback; mask postprocess (brow/lip subtract, lip landmark fallback, DL vs geometry cheek rules).
5. **ColorFeatureExtractor** — per-zone LAB (skin hue-trim, hair luminance trim, iris ring, lip brightness clusters); undertone fusion (skin ab, hair ash, iris guards); VCI and contrast bucket; AUA heuristics.
6. **SeasonClassifierEnsemble** — unified interface for Munsell 16-type lookup, Park IMCOM'18, swatch vote, wrist undertone prior, and future DL backend; weighted fusion with configurable weights; outputs 4/12/16 labels, top-k, Munsell scores, contributor dict.
7. **MunsellLookup** — four axes (undertone, chroma, value, contrast) normalized 1–5 → 16 types via lookup table; primary rule-based season signal (ensemble weight 0.70 default).
8. **ParkIMCOM18** — undertone from skin a* vs b*; 4-season from mean |ΔL| vs threshold; standalone or ensemble contributor.
9. **SwatchVote** — skin RGB vs reference swatch anchors in LAB; always-on ensemble contributor.
10. **WristUndertone** — optional second upload; vein HSV + ΔE palette; shifts 4-season prior in ensemble.
11. **DeepArmocromiaClassifier** (post-training) — runtime DL backend mirroring Park/Munsell interface; lazy-loaded weights; trained in `season-dl/` subproject.
12. **GeometryAnalyzer** — face shape classification (oval, round, square, heart, diamond, oblong, triangle variants) from landmark ratios; confidence score.
13. **RecommendationEngine** — rule-based outputs by category (glasses, hair, makeup, clothing, jewelry) from geometry + season + contrast; versioned rules with `rule_id` and `based_on` provenance.
14. **AnalysisPaletteBuilder** — face swatches per region + season reference anchors from `reference_swatches.json`; exposed as `metrics.analysis_palette`.
15. **AnalysisContract** — Pydantic models + exported JSON Schema; non-breaking extensions only for new optional fields.
16. **DebugDumpService** — numbered artifacts per run (meta, masks, seasonal JSON, contour image); served via `GET /debug/{run_id}/{filename}`.
17. **DeepArmocromiaEval** — dataset loader (Italian labels → face-ai IDs), subprocess or in-process eval, metrics JSON + confusion matrices + mistakes list; shared between rules benchmark and DL cache.
18. **MobileAnalysisClient** (face-ai-mobile) — photo capture/pick, multipart upload, response parse, result store, navigation; configurable LAN backend URL.

### Pipeline architecture

```
POST /analyze (face + optional wrist + meta_json)
  → IntakeAndQualityGate
  → LandmarkService
  → ParsingRegistry → MaskPostprocess
  → ColorFeatureExtractor (LAB, undertone, VCI)
  → SeasonClassifierEnsemble (Munsell + Park + swatch + wrist [+ DL])
  → GeometryAnalyzer
  → RecommendationEngine
  → AnalysisPaletteBuilder + MaskPreview
  → AnalysisResponse + optional DebugDump
```

### Season classification (current + target)

- **Default:** `ensemble` — Munsell 0.70, Park 0.15, swatch 0.10, wrist 0.05.
- **Standalone modes:** `munsell_lookup`, `park`, `deep_armocromia_dl` (future).
- **Taxonomy:** 4 parent seasons (spring/summer/autumn/winter) → 12 sub-types (e.g. light_summer, deep_autumn) → 16 Munsell types (AUA-style lookup).
- **Gaussian twelve:** code preserved in `gaussian_twelve.py` but disconnected; do not re-enable without explicit decision.
- **DL integration (target fusion weights, tune on val fold):**

```python
weights = {
    "deep_armocromia_dl": 0.35,
    "munsell": 0.40,
    "park": 0.10,
    "swatch": 0.10,
    "wrist": 0.05,
}
```

- **DL training:** isolated in `season-dl/` — SeasonDataset, HierarchicalSeasonModel, SeasonTrainer, PreprocessLetterbox, SeasonModelEval (see DL sub-PRD).

### Parsing & color extraction

- Default parsing: BiSeNet ResNet34 ONNX; optional FaRL, SegFace for quality experiments.
- Skin: `mean_lab` default or `xmeans_hsv_deltae` (Colors Matter–style); hue-trim 13–24, shadow cut, luminance trim.
- Hair: luminance percentile trim avoiding skin overlap.
- Iris: luminance ring sampling; undertone guards when iris matches skin.
- Lips: 3-tier brightness clusters (AUA capstone pattern).

### API contract (stable + extensions)

- Top-level: `photo_quality`, `metrics`, `confidence`, `recommendations[]`, `debug`, `disclaimer`.
- `metrics.seasonal`: 4/12/16 labels, confidences, top-k, `munsell_scores`, `classifier_contributors`, `delta_e_scores`, `undertone_source`, `seasonal_method`.
- `metrics.analysis_palette`: `{ face: [PaletteSwatch], season_reference: [...] }`.
- `metrics.geometry`: ratios, `face_shape`, confidence.
- `metrics.contrast`: per-region L*, VCI, `contrast_bucket`.
- Extensions for DL: `classifier_contributors["deep_armocromia_dl"]`, `seasonal_method` values — no breaking changes.

### Mobile integration

- Upload: `POST /analyze` multipart `file` from face-ai-mobile.
- Parse layer maps geometry, season, recommendations to RU UI strings.
- **Gap to close:** `POST /contour/final` or expose contour via debug artifact URL pattern mobile already expects.
- Optional future: wrist upload, top-k UI, seasonal confidence bars.

### Repository layout

| Area | Responsibility |
|------|----------------|
| Root `app/` | Production API, pipeline, backends, services, schemas |
| Root `tests/` | Unit/integration tests for rules pipeline |
| Root `scripts/eval_season_dataset.py` | Rule-based Deep Armocromia benchmark |
| Root `static/` | Debug web UI |
| `dataset/RGB/RGB/` | Deep Armocromia images (external) |
| `artefacts/eval/` | Benchmark JSON outputs |
| `season-dl/` | DL preprocess cache, training (future), DL-specific tests and PRD |
| `face-ai-mobile/` | Expo React Native client (separate repo) |

### Success metrics

| Metric | Current (rules, smoke) | Paper DL | Target (fused + DL) |
|--------|------------------------|----------|---------------------|
| 4-season top-1 | ~40% | 55.4% | ≥ 58% |
| 4-season top-2 | — | 80.8% | ≥ 82% |
| 12-season top-1 | ~20% | 31.8% | ≥ 38% |
| 12-season top-3 | — | 66.3% | ≥ 68% |
| Fused top-1 (4) | — | — | > max(rules, DL) |
| Inference latency (GPU DL) | — | — | < 100 ms |
| API end-to-end (typical) | — | — | < 5 s on LAN mobile |

## Testing Decisions

**Principle:** Test external behavior and contracts — API response shape, classifier outputs, fusion results, palette contents — not private implementation details unless guarding critical invariants (label mapping, probability sums).

### Modules to test (root face-ai)

1. **MunsellLookup / AUA scales** — axis bin boundaries, 16-type lookup for fixture LAB profiles.
2. **ParkIMCOM18** — undertone and 4-season formulas against known inputs.
3. **SeasonClassifierEnsemble** — fixed contributor dicts → expected fused top-1; DL weight 0 preserves legacy behavior.
4. **ColorFeatureExtractor / AUA heuristics** — hue-trim, lip clusters, hair trim, undertone fusion edge cases (fair cool skin, ash hair).
5. **AnalysisPaletteBuilder** — RGB/hex/LAB conversion; season reference anchors for each 4-season.
6. **AnalysisContract** — response validates against JSON Schema; required fields present when face detected.
7. **DeepArmocromiaEval smoke** — 2-image eval or subprocess with `--limit 5`; skips if dataset missing.
8. **IntakeAndQualityGate** — blocked analysis returns `metrics: null` and quality issues (behavioral).

### Modules to test (season-dl subproject)

9. **SeasonDataset** — fixture cache entry → correct 4/12 labels; signature mismatch skipped.
10. **HierarchicalSeasonModel** — forward output keys; `probs_twelve` sums to ~1.0.
11. **DeepArmocromiaClassifier (runtime)** — dict shape matches Park backend; confidence ∈ [0,1].
12. **PreprocessLetterbox** — aspect ratio preserved within ε before square resize.

### Prior art in codebase

- `tests/test_munsell_aua_scales.py`, `tests/test_park_imcom18.py`, `tests/test_ensemble_park.py`
- `tests/test_aua_heuristics.py`, `tests/test_analysis_palette.py`
- `tests/test_season_benchmark.py` (root); `season-dl/tests/test_preprocess.py`

### Not unit-tested (integration/manual)

- Full `/analyze` end-to-end with live parsing models (manual via debug UI).
- Full 912-image benchmark (local before release; CI optional).
- DL training convergence (checkpoint val metrics).
- Mobile E2E on device (manual Expo Go).
- GPU memory fit for DL batch sizes.

## Out of Scope

- Virtual makeup try-on and AR overlay (Park long-term roadmap; researched, not built).
- Product shade matching via ΔE to cosmetic catalog (AUA capstone pattern).
- Custom dataset collection, re-labeling, or crowdsourced Armocromia annotation beyond Deep Armocromia.
- Illumination correction / AWB pre-step (flagged in deep research; separate future PRD).
- Multi-photo intake API (single face photo only for v1).
- Re-enabling `gaussian_twelve` classifier without explicit product decision.
- ONNX export / on-device DL inference in mobile (backend-only DL for v1).
- Wrist photo used in DL training (face crop only for DL; wrist remains rules ensemble input).
- Human-in-the-loop relabeling tool.
- Production GPU autoscaling infrastructure.
- Questionnaire-driven style profiling (commercial apps feature; not in v1).
- Third-party cloud seasonal color API integration (self-hosted only).

## Further Notes

- **Domain vocabulary:** Armocromia Flow Theory, 4 parent seasons, 12 sub-types, 16 Munsell types, undertone (warm/cool/neutral), VCI (value contrast index), masked color zones, parsing backends, ensemble contributors, top-k neighbor seasons, Deep Armocromia benchmark.
- **Research artefacts:** `artefacts/best-soltions.md` (5 pipeline comparisons + roadmap), `artefacts/color-analysis.md`, `artefacts/face-parsing-models-comparison-2026-05.md`.
- **DL sub-PRD:** training architecture, cache workflow, and DL-specific user stories live in `season-dl/artefacts/prd/deep-armocromia-dl-classifier.md` — implement after or in parallel with ensemble integration hooks in root API.
- **Current cache state:** partial DL preprocess cache (~smoke samples); full ~4920-image cache required before training.
- **Rule baseline artifact:** `artefacts/eval/deep_armocromia_latest.json` — smoke eval; full benchmark pending.
- **Neighbor confusion:** expect Autumn↔Winter and Spring↔Summer errors; top-k UI and fused ensemble mitigate user-facing wrong labels.
- **Commercial context:** market apps claim 96–97% accuracy with opaque methods; face-ai prioritizes transparency and measurable benchmark progress over vendor-style claims.
- **Mobile repo:** face-ai-mobile is separate git repo; API contract stability is the integration boundary.
- **No formal ADRs:** architectural decisions captured in this PRD, history.md, and research artefacts.
