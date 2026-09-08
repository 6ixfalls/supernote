# Supernote Private Cloud Server

This package provides a self-hosted implementation of the Supernote Cloud server, enhanced with AI-powered insights, a modern Web UI, and robust background processing.

## Core Features

-   **Seamless Sync**: Implements the native Supernote sync protocol.
-   **AI Synthesis**: Automatically transcribes handwriting and identifies key insights using Google Gemini.
-   **Knowledge Exploration**: Cross-notebook semantic search and web-based file browsing.
-   **Private & Local**: Store your notes and metadata on your own infrastructure.

## Getting Started

See the main [README.md](../../README.md) for a quick start guide.

### Prerequisites

-   A Supernote device (Nomad, A5 X, A6 X, etc.)
-   Python 3.13+ or Docker.
-   (Recommended) **Vercel AI Gateway API key** for OCR and Summarization.

### Configuration

The server is configured via `config/config.yaml` or environment variables.

For a comprehensive reference, see the [ServerConfig documentation](https://allenporter.github.io/supernote/supernote/server.html#ServerConfig).

#### AI Configuration

AI features are served by Gemini models routed through the
[Vercel AI SDK](https://ai-python.dev/) and the AI Gateway. To enable them, set
an AI Gateway API key:
```bash
export SUPERNOTE_AI_API_KEY="your-ai-gateway-api-key"
# Alternatively, the SDK-native AI_GATEWAY_API_KEY is honored as a fallback.
```

Additional AI settings:

| Setting | Env Var | Default | Description |
| --- | --- | --- | --- |
| `ai.provider` | `SUPERNOTE_AI_PROVIDER` | `google` | Gateway serving provider (`google` for the Gemini API or `vertex` for Vertex AI). |
| `ai.ocr_model` | `SUPERNOTE_AI_OCR_MODEL` | `gemini-3.6-flash` | Model used for OCR and summarization. |
| `ai.embedding_model` | `SUPERNOTE_AI_EMBEDDING_MODEL` | `gemini-embedding-001` | Model used for embeddings. |
| `ai.flex` | `SUPERNOTE_AI_FLEX` | `false` | Use the Flex service tier for lower-cost, batch-tolerant generation. |
| `ai.zdr` | `SUPERNOTE_AI_ZDR` | `false` | Require zero-data-retention routing for generation and embeddings. |
| `ai.max_concurrency` | `SUPERNOTE_AI_MAX_CONCURRENCY` | `5` | Maximum number of concurrent AI API calls. |
| `ai.prompts_dir` | `SUPERNOTE_AI_PROMPTS_DIR` | unset | Directory containing custom OCR and summarization prompts. |

### Running the Server

Start the server using the unified `supernote` CLI:

```bash
# Start the server on port 8080
supernote serve
```

To override settings via environment:

```bash
export SUPERNOTE_PORT=8080
export SUPERNOTE_HOST=0.0.0.0
supernote serve
```

### Running with Docker

```bash
# Build the image
docker build -t supernote .

# Run the container
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/storage:/storage \
  -e SUPERNOTE_AI_API_KEY="your-ai-gateway-key" \
  --name supernote-server \
  supernote serve
```

### Connecting Your Device

1. Review the [official Private Cloud setup guide](https://support.supernote.com/Whats-New/setting-up-your-own-supernote-private-cloud-beta).
2. Ensure your Supernote device and server are on the same Wi-Fi network.
3. On your Supernote device, go to **Settings** > **Sync** > **Supernote Cloud**.
4. Select **Private Cloud** and enter your server's IP and port (e.g., `192.168.1.100:8080`).
5. Attempt to login using the credentials created via `supernote admin user add`.
6. Configure folders to sync (e.g., `Note`, `Document`, `EXPORT`) in **Settings** > **Drive** > **Private Cloud**.

## Robustness & Maintenance

Supernote Knowledge Hub is built for long-term stability:

- **Database Migrations**: Uses Alembic for seamless schema and data updates (see [Alembic Migration History & Conventions](../alembic/README.md)).
- **Background Polling**: Automatically recovers stalled processing tasks.
- **Integrity Checks**: Periodically verifies file storage consistency.
- **Storage Quotas**: Manage user storage limits effectively.

## Debugging & Tracing

The server logs all incoming requests to `storage/system/server_trace.log`:

```bash
tail -f storage/system/server_trace.log
```

## Development

- **Entry Point**: `supernote/server/app.py`
- **Tests**: `tests/server/`
- **Ephemeral Mode**: Run `supernote serve --ephemeral` for a transient, pre-configured test instance.

For contribution guidelines, see [CONTRIBUTING.md](../docs/CONTRIBUTING.md).
