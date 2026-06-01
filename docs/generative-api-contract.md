# Generative try-on API contract

Face AI calls an external inpaint service via `GenerativeModelAPI` (`app/backends/try_on/generative_api.py`).

## Configuration

| Env | Description |
|-----|-------------|
| `FACE_AI_GENERATIVE_API_URL` | Base URL or full endpoint. Use `none` to disable. |
| `FACE_AI_GENERATIVE_API_KEY` | Bearer token; if empty, uses `FACE_AI_LLM_API_KEY` |
| `FACE_AI_GENERATIVE_TRANSPORT` | `openai_images_edit` (default), `custom_json`, or `gemini_native` |
| `FACE_AI_GENERATIVE_MODEL` | Model id for OpenAI Images API or Gemini (`gemini-3-pro-image-preview`) |
| `FACE_AI_GENERATIVE_TIMEOUT_S` | HTTP timeout (default 90) |
| `FACE_AI_GENERATIVE_STRENGTH` | Used by `custom_json` only (default 0.75) |
| `FACE_AI_GENERATIVE_USE_MASK` | Server default for `use_mask` in try-on meta (default true) |
| `FACE_AI_GENERATIVE_COMPOSITE_MASK` | Blend model output with original inside parsing mask (default true; critical for makeup) |

**Try-on request** (`meta_json`): `"use_mask": true|false` overrides server default. When `false`, OpenAI edit is called without `mask` file; prompts include explicit region hints.

### OpenAI-compatible (default)

```env
FACE_AI_GENERATIVE_API_URL=https://api.openai.com/v1
FACE_AI_GENERATIVE_TRANSPORT=openai_images_edit
FACE_AI_GENERATIVE_MODEL=dall-e-2
```

Resolved endpoint: `POST {base}/images/edits` when URL ends with `/v1`.

**Request:** `multipart/form-data`

| Field | Type | Notes |
|-------|------|-------|
| `image` | PNG file | Source photo (RGB) |
| `mask` | PNG file | RGBA; **transparent** pixels = inpaint region |
| `prompt` | string | Edit instruction |
| `model` | string | From `FACE_AI_GENERATIVE_MODEL` |
| `n` | `1` | |
| `response_format` | `b64_json` | |

**Response:**

```json
{ "data": [{ "b64_json": "<PNG or JPEG bytes as base64>" }] }
```

### Custom JSON (self-hosted)

```env
FACE_AI_GENERATIVE_TRANSPORT=custom_json
FACE_AI_GENERATIVE_API_URL=http://127.0.0.1:8090/inpaint
```

**Request:** `application/json`

```json
{
  "image_b64": "<JPEG>",
  "mask_b64": "<PNG, 255 = inpaint region>",
  "prompt": "...",
  "negative_prompt": "...",
  "strength": 0.75
}
```

**Response:**

```json
{ "image_b64": "<JPEG result>" }
```

### Gemini native (google-genai SDK)

For `gemini-3-pro-image-preview` and similar image models, use semantic editing via `generateContent` instead of OpenAI `/images/edits`:

```env
FACE_AI_GENERATIVE_TRANSPORT=gemini_native
FACE_AI_GENERATIVE_MODEL=gemini-3-pro-image-preview
# FACE_AI_GENERATIVE_API_KEY=<Google AI Studio key>
# FACE_AI_GENERATIVE_API_URL=none
```

Requires `google-genai`. The model receives the source photo + preservation prompt; makeup zones are blended locally with parsing masks when `FACE_AI_GENERATIVE_COMPOSITE_MASK=true`.

## Local mock

```bash
python scripts/mock_generative_server.py
# FACE_AI_GENERATIVE_API_URL=http://127.0.0.1:8090/v1
```
