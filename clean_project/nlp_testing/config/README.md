# config/

All user-editable configuration for the NLP assistant.

## Files

| File | Purpose |
|------|---------|
| `nlp_config.yaml` | Main settings: LLM model name, RAG parameters, file paths, debug flag. |
| `sorting_rules.json` | Maps detected material names (lowercase) to disposal bin names. |
| `system_prompt.txt` | LLM system prompt for ongoing chat turns. Contains `{rag_context}` and `{bm25_context}` placeholders filled at runtime. |
| `start_conversation.txt` | LLM prompt for the initial 'start' turn (when detected objects are provided). Contains `{disposal_items}` and `{rag_context}` placeholders. |
| `test_input.json` | Sample input for the 'start' command when testing standalone. Format: `{"labels": ["Metal", "Glass"]}`. |

## Editing sorting_rules.json

Keys are lowercase material names as returned by the detector. Values are the bin name shown to the user.
Add new materials here if the model is retrained with additional classes.

## Editing the prompts

The placeholder names in curly braces (`{rag_context}` etc.) are required and must not be renamed.
Everything else can be freely edited to tune the LLM's tone and behaviour.
