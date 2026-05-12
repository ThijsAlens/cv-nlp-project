# utils/

Shared utilities used by all other submodules. Currently contains only `io.py`.

## Modules

| Module | Purpose |
|--------|---------|
| `io.py` | `read_json`, `write_json`, `read_yaml`, `write_yaml`, `ensure_dir` - all file I/O in the project goes through these helpers instead of calling `json`/`yaml` directly. |
