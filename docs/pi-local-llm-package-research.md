# Pi + llama.cpp integration research

_Research snapshot: 2026-08-23. Sources: Pi's official docs/package gallery, npm metadata, first-party repositories/source, and local smoke tests against the running server._

## Bottom line

- **Best overall:** use Pi's built-in llama.cpp integration, not a package, if the server can run in router mode. Pi already provides `/login llama.cpp`, `/llama` model download/load/unload, and `/model` selection. It is first-party and has no third-party extension risk. [Official docs](https://pi.dev/docs/latest/llama-cpp)
- **Best package for the currently running single-model server:** **`pi-llamacpp-provider` 0.2.0**. It was the most accurate connector in a live test: it detected the 38,400-token context, capped output at 16,384, detected reasoning and vision, and completed an end-to-end prompt successfully.
- **Best package when using llama.cpp's multi-model router and wanting richer status/load UX:** **`pi-llama-cpp` 0.9.2**. It is newer, more widely used, and offers `/models`, load/unload/switch, progress, multiple servers, and Pi 0.84+ peer compatibility, but its model metadata is less conservative.

## Current-machine observation

The server at `http://127.0.0.1:8080` is reachable and currently exposes one model:

- Model: `ggml-org/Qwen3.8-27B-GGUF:Q8_0`
- Context: 38,400 tokens
- Vision: enabled
- Tool-capable chat template: reported by llama.cpp
- Launcher: `C:\Users\pre\AppData\Local\Microsoft\WindowsApps\llama.exe`, version `b10217-ddd4ec142`
- Active command: `llama.exe serve -hf ggml-org/Qwen3.8-27B-GGUF:Q8_0`
- Server shape: single-model, not Pi's required router-mode workflow

Therefore, Pi's built-in `/llama` lifecycle manager will require restarting the server in router mode. The installed `llama.exe` help confirms support for `--models-dir`, `--no-models-autoload`, `--jinja`, and `--host`, so that migration is available. A connector package is appropriate only if the current single-model launch should remain unchanged.

## Live package tests

Both candidates were first loaded temporarily with `pi -e`. After validation, `pi-llamacpp-provider` was installed globally; `pi-llama-cpp` remains uninstalled.

### `pi-llamacpp-provider`

```text
provider  model                            context  max-out  thinking  images
llamacpp  ggml-org/Qwen3.8-27B-GGUF:Q8_0  38.4K    16.4K    yes       yes
```

Three isolated end-to-end checks passed:

1. A no-tools, thinking-off prompt completed successfully.
2. With thinking set to `low`, Pi received a structured `thinking` block and the correct final response.
3. The model issued a real `read` tool call for a temporary marker file, Pi returned a successful tool result, and the model reproduced the marker exactly on the following turn.

### `pi-llama-cpp`

```text
provider                            model                            context  max-out  thinking  images
llama-server=http://127.0.0.1:8080  ggml-org/Qwen3.8-27B-GGUF:Q8_0  38.4K    38.4K    yes       yes
```

Discovery worked, but it advertised the full context window as the per-turn output limit.

## Focused comparison

| Option | Setup and model UX | Metadata/request correctness | Compatibility and maintenance | Verdict |
|---|---|---|---|---|
| **Pi built-in llama.cpp** | `/login llama.cpp`, `/llama`, `/model`; downloads, loads, and unloads models | First-party integration; documents `--jinja` for compatible tool calling | Ships with current Pi | **Best overall if router mode is acceptable** |
| **`pi-llamacpp-provider` 0.2.0** | Zero-config at default `127.0.0.1:8080`; automatic startup discovery; no model-management TUI | Reads `/v1/models` and `/props`; detects context, vision, chat-template reasoning; uses `system`, `max_tokens`, bounded output, and chat-template thinking controls | Zero runtime dependencies; Pi peer `*`; Node >=22.19; older release than `pi-llama-cpp` | **Best for the current single-model server** |
| **`pi-llama-cpp` 0.9.2** | `/models` status/info/unload; router load/switch; SSE progress; single/legacy/multiple-server support | Correct live context/vision here, but source marks every model reasoning-capable and sets `maxTokens` equal to context | Zero runtime dependencies; peers require Pi >=0.84; recently updated; highest usage among llama.cpp-specific packages | **Best third-party router UX, with metadata caveats** |
| `pi-localllm-provider` 0.5.1 | `/localllm` wizard; multiple servers/backends; manual refresh | Good llama.cpp context/vision detection, but conservatively marks llama.cpp reasoning false and does not manage router models | Zero runtime dependencies; Pi peer `*`; current | Best general multi-runtime wizard, not best llama.cpp-specific connector |
| `@hypabolic/crossbar` 0.8.0 | Polished auto-discovery and multi-backend TUI | Its own matrix has no llama.cpp load/unload or switching unless llama-swap is added | Exact peer pins to Pi/TUI/typebox 0.79.9 conflict with installed Pi 0.84.2 | Do not choose until peer compatibility is updated |
| `pi-llama-switch` 1.0.2 | Starts/restarts one server per hand-written model command | Useful when every model needs different launch flags | Requires substantial JSON and owns process lifecycle | Specialized, not easiest general setup |
| `models.json` | Manual JSON; no third-party code | Most explicit control over context, output, reasoning, and compatibility | Built into Pi and hot-reloaded | Best low-risk fallback for one stable endpoint |

## Windows fit

`pi-llamacpp-provider` is a pure TypeScript/HTTP connector with no runtime dependencies or native binaries. Its Node requirement is >=22.19, satisfied by the installed Node 22.23.2. The current llama.cpp release is exposed through the Windows app alias `llama.exe` rather than the older `llama-server` command. Because the provider only connects to a running endpoint, that launcher distinction does not matter, and it avoids shell/process-management differences between Windows and Unix. Keep llama.cpp bound to `127.0.0.1`; configure authentication before exposing it to another interface.

## Why `pi-llamacpp-provider` wins for this server

1. **No setup beyond installation at the default endpoint.** It discovers the currently running server on Pi startup.
2. **The live result matched llama.cpp's reported model properties.** It registered 38.4K context, reasoning, vision, and a sensible 16.4K output cap.
3. **Its compatibility mapping is llama.cpp-specific.** The source sends a `system` role instead of `developer`, uses `max_tokens`, and drives thinking through chat-template kwargs rather than assuming OpenAI reasoning semantics. [Source](https://github.com/T0mSIlver/pi-llamacpp-provider/blob/main/src/index.ts)
4. **Its security footprint is small.** It has no runtime dependencies, communicates with the configured HTTP endpoint, and only persists a learned chat-template capability cache under the user's cache directory. [npm metadata](https://registry.npmjs.org/pi-llamacpp-provider)

Its limitation is deliberate: it connects Pi to a running server but does not start the server or provide load/unload controls.

## Why not default to `pi-llama-cpp` here

`pi-llama-cpp` has the strongest management UX and is the better package for llama.cpp router presets. However, its current source:

- returns `reasoning: true` for every model because it does not detect that capability; and
- uses the complete context window as `maxTokens`.

Those choices are visible in [`BaseModel`](https://github.com/gsanhueza/pi-llama-cpp/blob/master/src/models/baseModel.ts) and were reflected by the 38.4K live output cap in the smoke test. They are acceptable for some reasoning/router setups but are less robust than `pi-llamacpp-provider` for the current single Qwen server.

## Recommended path

### Keep the existing single-model launch

Installed globally and verified:

```text
pi install npm:pi-llamacpp-provider
```

Reload or restart Pi, then select `llamacpp/ggml-org/Qwen3.8-27B-GGUF:Q8_0` with `/model`.

### Prefer first-party model management

Restart llama.cpp without `-m`, `--model`, or `-hf`, provide `--models-dir`, `--no-models-autoload`, `--jinja`, `--host 127.0.0.1`, and the desired context/GPU flags. Then use:

```text
/login llama.cpp
/llama
/model
```

See [Pi's llama.cpp documentation](https://pi.dev/docs/latest/llama-cpp). This is the recommended long-term configuration.

## Primary sources

- [Pi official llama.cpp docs](https://pi.dev/docs/latest/llama-cpp)
- [Pi official provider docs](https://pi.dev/docs/latest/providers)
- [Pi custom model docs](https://pi.dev/docs/latest/models)
- [`pi-llamacpp-provider` gallery](https://pi.dev/packages/pi-llamacpp-provider), [repository](https://github.com/T0mSIlver/pi-llamacpp-provider), and [source](https://github.com/T0mSIlver/pi-llamacpp-provider/blob/main/src/index.ts)
- [`pi-llama-cpp` gallery](https://pi.dev/packages/pi-llama-cpp), [repository](https://github.com/gsanhueza/pi-llama-cpp), and [source](https://github.com/gsanhueza/pi-llama-cpp/blob/master/src/models/baseModel.ts)
- [`pi-localllm-provider` gallery](https://pi.dev/packages/pi-localllm-provider) and [repository](https://github.com/freeyoung/pi-localllm-provider)
- [`@hypabolic/crossbar` gallery](https://pi.dev/packages/@hypabolic/crossbar) and [package metadata](https://github.com/Hypabolic/Crossbar/blob/main/package.json)
- [`pi-llama-switch` gallery](https://pi.dev/packages/pi-llama-switch)
