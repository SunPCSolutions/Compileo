# Embeddings Plugin User Guide (Qdrant + Ollama)

This guide explains the **Qdrant Embedding Plugin** for high-quality RAG indexing from chunked documents (especially textbook-style chapter/subchapter content).

---

## 1. What this plugin provides

After installation, the plugin contributes:

- API routes under `/api/v1/plugins/embeddings/*`
- CLI command group `compileo embeddings ...`
- GUI page **🧠 Embeddings**
- Background executor for async embedding/indexing jobs
- End-to-end Ollama embedding + Qdrant collection/create/upsert flow

---

## 2. Core concepts

### 2.1 `preprocess_policy`

`preprocess_policy` controls how chunk text is prepared before embedding/taxonomy shaping.

- Current recommended/supported value: `markdown_preserve_headings`
- Why it helps QA RAG: preserves section headers and hierarchy, which improves retrieval specificity for subsection-style questions.

### 2.2 `taxonomy_mode`: `chunk` vs `legacy`

#### `chunk` mode (recommended)

- Generates embedding-focused taxonomy per chunk.
- Stores per-chunk semantic trees in tenant `embed` table.
- Materializes a project taxonomy record (`Embedding Taxonomy (chunk mode)`) so Taxonomy UI can discover it.
- Best for topic-local precision and reduced cross-topic leakage.
- Supports chunk-batch execution for throughput tuning via `taxonomy_chunk_batch_size` while preserving **chunk-wide output** (each chunk still gets its own taxonomy tree/leaf IDs).

#### `legacy` mode

- Reuses existing project taxonomy matching/annotation path.
- Useful when a project already has a taxonomy you explicitly want to reuse.

### 2.3 Micro-segmentation for long chunks

Before embedding, long chunk text is split into overlapping token segments.

- `segment_token_size`: tokens per segment
- `segment_overlap_tokens`: overlap between adjacent segments
- `skip_segmentation`: bypass extra splitting and embed each selected chunk as-is

Default QA-RAG values:

- `segment_token_size = 512`
- `segment_overlap_tokens = 80`

These defaults balance semantic context continuity and retrieval granularity.

### 2.4 Compact taxonomy payload contract

When taxonomy metadata is attached, payload uses:

```json
{
  "taxonomy": {
    "tree": [
      {
        "name": "Acute Kidney Injury",
        "children": [
          { "name": "Diagnosis" },
          { "name": "Treatment" }
        ]
      }
    ],
    "leaf_ids": ["11010", "11011"]
  }
}
```

---

## 3. Settings keys (API/GUI/CLI)

| Setting Key | Default | Notes |
|---|---|---|
| `embedding_qdrant_url` | `http://localhost:6333` | Qdrant base URL |
| `embedding_qdrant_collection` | `compileo_chunks` | Collection name |
| `embedding_ollama_url` | `http://localhost:11434` | Ollama base URL |
| `embedding_ollama_model` | `nomic-embed-text` | Embedding model |
| `embedding_distance` | `Cosine` | `Cosine`, `Dot`, `Euclid` |
| `embedding_batch_size` | `128` | Qdrant upsert batch size |
| `embedding_include_taxonomy` | `true` | Attach taxonomy metadata |
| `embedding_taxonomy_mode` | `chunk` | `chunk` or `legacy` |
| `embedding_preprocess_policy` | `markdown_preserve_headings` | Preserve heading structure |
| `embedding_taxonomy_generator` | `gemini` | `gemini`, `grok`, `ollama`, `openai` |
| `embedding_taxonomy_processing_mode` | `fast` | `fast` or `complete` |
| `embedding_taxonomy_anchor_main_topic` | `true` | Anchor taxonomy under chunk main topic |
| `embedding_taxonomy_chunk_batch_size` | `10` | Chunks processed per taxonomy batch and per-batch concurrent workers (e.g., 50 => up to 50 concurrent for full batch); taxonomy remains per chunk |
| `embedding_segment_token_size` | `512` | Segment size for micro-segmentation |
| `embedding_segment_overlap_tokens` | `80` | Segment overlap (must be smaller than size) |
| `embedding_skip_segmentation` | `false` | If `true`, embed each selected chunk as one segment |

---

## 4. GUI workflow

1. Open **🧠 Embeddings**.
2. Select project.
3. Configure Qdrant/Ollama/taxonomy settings.
4. Tune segmentation with sliders:
   - **Segment token size**: `256–1024` (step `32`, default `512`)
   - **Segment overlap tokens**: `40–200` (step `8`, default `80`)
   - **Skip micro-segmentation**: embed each selected chunk directly without additional splitting
5. Use **Preview Payload** to inspect both chunk-level and segment-level metadata.
6. Submit async indexing job.

Recommended baseline for textbook QA RAG:

- `include_taxonomy = true`
- `taxonomy_mode = chunk`
- `taxonomy_anchor_main_topic = true`
- `preprocess_policy = markdown_preserve_headings`
- `segment_token_size = 512`
- `segment_overlap_tokens = 80`

---

## 5. API reference

Base path: `/api/v1/plugins/embeddings`

### 5.1 `GET /settings`

Returns current embedding plugin settings for current user.

### 5.2 `POST /settings`

Persist embedding plugin settings.

Request body fields:

- `qdrant_url: string`
- `collection: string`
- `ollama_url: string`
- `ollama_model: string`
- `distance: string`
- `batch_size: integer`
- `include_taxonomy: boolean`
- `taxonomy_mode: string`
- `preprocess_policy: string`
- `taxonomy_generator: string`
- `taxonomy_processing_mode: string`
- `taxonomy_anchor_main_topic: boolean`
- `taxonomy_chunk_batch_size: integer`
- `segment_token_size: integer`
- `segment_overlap_tokens: integer`
- `skip_segmentation: boolean`

### 5.3 `GET /scope?project_id=<project_id>`

Returns:

- available projects
- project documents + chunk counts (when `project_id` provided)
- taxonomy existence summary
- detected Ollama models

### 5.4 `POST /index/preview`

Safe preview; does not submit a job.

Request body supports:

- `project_id` (required for practical use)
- `document_id`
- `document_ids[]`
- `chunk_ids[]`
- `include_taxonomy`
- `taxonomy_mode`
- `auto_generate_taxonomy`
- `taxonomy_generator`
- `taxonomy_processing_mode`
- `taxonomy_anchor_main_topic`
- `taxonomy_chunk_batch_size`
- `collection`
- `model`
- `segment_token_size`
- `segment_overlap_tokens`
- `skip_segmentation`
- `dry_run`

Preview response includes both chunk and segment diagnostics:

- `preview`
- `preview_segments`
- `preview_count`
- `preview_segment_count`
- `segmented_parent_count`
- `segment_token_size`
- `segment_overlap_tokens`
- `skip_segmentation`

### 5.5 `POST /index`

Submits async embedding/indexing job.

Request body uses same fields as `/index/preview`.

`project_id` is required.

Response fields:

- `status` (`submitted`)
- `job_id`
- `executor_key` (`qdrant_embed_index`)
- `taxonomy_status`
- `annotation_status`

Minimal request example:

```json
{
  "project_id": "09694174-b265-4900-936b-debabf85ef82",
  "include_taxonomy": true,
  "taxonomy_mode": "chunk",
  "taxonomy_generator": "gemini",
  "taxonomy_processing_mode": "fast",
  "taxonomy_anchor_main_topic": true,
  "taxonomy_chunk_batch_size": 10,
  "collection": "compileo_chunks",
  "model": "nomic-embed-text",
  "segment_token_size": 512,
  "segment_overlap_tokens": 80,
  "skip_segmentation": false,
  "dry_run": false
}
```

---

## 6. CLI reference

### 6.1 `compileo embeddings settings`

Options:

- `--owner-id` (required)
- `--qdrant-url`
- `--collection`
- `--ollama-url`
- `--ollama-model`
- `--distance`
- `--batch-size`
- `--include-taxonomy / --no-include-taxonomy`
- `--taxonomy-mode`
- `--taxonomy-generator`
- `--taxonomy-processing-mode`
- `--taxonomy-anchor-main-topic / --no-taxonomy-anchor-main-topic`
- `--taxonomy-chunk-batch-size`
- `--segment-token-size`
- `--segment-overlap-tokens`
- `--skip-segmentation / --no-skip-segmentation`
- `--preprocess-policy`

Example:

```bash
compileo embeddings settings \
  --owner-id <owner_id> \
  --qdrant-url http://localhost:6333 \
  --collection compileo_chunks \
  --ollama-url http://localhost:11434 \
  --ollama-model nomic-embed-text \
  --distance Cosine \
  --batch-size 128 \
  --include-taxonomy \
  --taxonomy-mode chunk \
  --taxonomy-generator gemini \
  --taxonomy-processing-mode fast \
  --taxonomy-anchor-main-topic \
  --taxonomy-chunk-batch-size 10 \
  --segment-token-size 512 \
  --segment-overlap-tokens 80 \
  --no-skip-segmentation \
  --preprocess-policy markdown_preserve_headings
```

### 6.2 `compileo embeddings index`

Options:

- `--owner-id` (required)
- `--project-id` (required)
- `--document-id` (repeatable)
- `--taxonomy-anchor-main-topic / --no-taxonomy-anchor-main-topic`
- `--taxonomy-chunk-batch-size`
- `--segment-token-size`
- `--segment-overlap-tokens`
- `--skip-segmentation / --no-skip-segmentation`
- `--dry-run`

Example:

```bash
compileo embeddings index \
  --owner-id <owner_id> \
  --project-id <project_id> \
  --document-id <doc_id_1> \
  --document-id <doc_id_2> \
  --taxonomy-anchor-main-topic \
  --taxonomy-chunk-batch-size 10 \
  --segment-token-size 512 \
  --segment-overlap-tokens 80 \
  --skip-segmentation \
  --dry-run
```

### 6.3 Segmentation behavior notes

- If chunk token count is below `segment_token_size`, output remains one segment even if overlap is non-zero.
- `segment_overlap_tokens` only affects sliding-window splitting when chunk token count exceeds `segment_token_size`.
- If `skip_segmentation=true`, chunk is always treated as one segment (with segment metadata still populated).
- `taxonomy_chunk_batch_size` controls both batch grouping and in-batch concurrency for chunk taxonomy generation; taxonomy output is still generated and stored per chunk.

---

## 7. Troubleshooting

### "No taxonomy detected in embeddings scope"

- Run at least one `chunk` mode indexing request with `include_taxonomy=true`.
- Confirm project has chunks in selected scope.

### Topic anchoring appears generic

- Keep `taxonomy_anchor_main_topic=true`.
- Use stronger taxonomy generator/model.
- Ensure chunk text is non-empty and preserves headings.

### Segment count unexpectedly high

- Increase `segment_token_size`.
- Reduce `segment_overlap_tokens`.

---

## 8. Related guides

- [Plugin Management Overview](plugins.md)
- [Plugin Management API](pluginsapi.md)
- [Plugin Management CLI](pluginscli.md)
- [Taxonomy API Guide](taxonomyapi.md)
- [Job Handling API Guide](jobhandlingapi.md)
