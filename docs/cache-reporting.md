# Per-request cache reporting

The launcher enables `--enable-prompt-tokens-details` so OpenAI-compatible
clients can read `usage.prompt_tokens_details.cached_tokens` in Chat
Completions responses. For streaming requests, send
`"stream_options": {"include_usage": true}` and read the final usage chunk.
A cold request may report zero; enabling reporting does not guarantee a hit.

This is response metadata only: it does not enable or retune prefix caching,
change cache retention, or change the KV pool. Prometheus cache metrics at
`/metrics` are separate and do not depend on this flag.

Existing servers only pick up this launcher change when the operator next
recreates/restarts them. The accompanying CPU-only regression checks the
launch arguments; it does not start a server or prove live cache reuse.
