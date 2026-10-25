# Spring Boot → LLM (Claude)

The smallest useful Java + LLM integration: one Spring Boot `POST` endpoint that forwards
a prompt to Claude through the **official Anthropic Java SDK** and returns the answer as JSON.

The existing Java track (`onnx-java/`, `springboot-ml-api/`) runs a local ONNX vision model.
This one calls a hosted LLM instead, the other half of how AI is usually wired into JVM
backends.

## Files

| File | What it does |
|---|---|
| `LlmApiApplication.java` | Spring Boot entry point. Creates one `AnthropicClient` bean (thread-safe, shared connection pool) |
| `ChatController.java` | `POST /api/chat`: validates the prompt, calls `client.messages().create(...)`, joins the text blocks, returns answer + token usage |
| `application.properties` | Port `8081` (so it can run next to `springboot-ml-api` on 8080) and the model id |

## Run

Requires Java 21+, Maven, and an Anthropic API key.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
mvn clean package
java -jar target/springboot-llm-api-1.0.0.jar
```

```bash
curl -X POST localhost:8081/api/chat \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "Explain ONNX in one sentence."}'
```

```json
{"answer": "...", "model": "claude-opus-5-5", "inputTokens": 25, "outputTokens": 30}
```

| Case | Status |
|---|---|
| Normal answer | `200` |
| Missing or blank `prompt` | `400` |
| Model declined (`stop_reason = refusal`) | `422` |

## What was learned

- **The SDK does the plumbing.** `AnthropicOkHttpClient.fromEnv()` reads the key from the
  environment, so no secret ever goes in `application.properties`. It also handles retries
  on 429/5xx. The whole LLM call is one builder chain.
- **One client per app.** The client owns an HTTP connection pool, so it is a Spring
  `@Bean` with `destroyMethod = "close"`, not something built per request.
- **The response is a list of content blocks, not a string.** Text has to be pulled out of
  the `text` blocks and joined.
- **Check `stop_reason` before reading content.** A safety refusal comes back as HTTP 200
  with `stop_reason = refusal` and no text. Without the check, the endpoint would return an
  empty answer with status 200. The request also turns on server-side `fallbacks: "default"`
  (beta header `server-side-fallback-2026-07-01`), so the API first retries a declined
  request on a fallback model by itself.
- **Model id lives in config** (`llm.model`), so swapping models needs no code change.

## How it was verified

No API key was available while building this, so it was verified against a local stub of
the Messages API (`ANTHROPIC_BASE_URL=http://127.0.0.1:9999`). The stub logged the outgoing
request: `POST /v1/messages` with the right model, system prompt, `max_tokens`, beta header
and `fallbacks` body field. The app parsed the canned reply into
`{"answer":"Paris.","model":"claude-opus-5-5","inputTokens":12,"outputTokens":3}`, and the
`400` and `422` paths returned as expected. **A real call to the Anthropic API has not been
made yet.** Run it once with a real key to confirm.

## Deliberately left out

Streaming, conversation history, structured JSON output (backlog #16), tests, Docker. The
backlog item asks for a minimal endpoint, and each of these is its own topic.
