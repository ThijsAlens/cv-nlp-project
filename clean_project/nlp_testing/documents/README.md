# documents/

Knowledge base for the RAG system. Organised into two subdirectories
that feed two different retrieval strategies.

## general/

Belgian waste disposal rules, one `.txt` file per material stream.
These files are indexed by FAISS for **semantic search**.

After adding or editing files here, rebuild the FAISS index:
```bash
uv run python scripts/run_build_index.py
```

Current files:
- `Regels_gft.txt` -- organic/vegetable/garden waste (GFT)
- `Regels_glas.txt` -- glass
- `Regels_papier_karton.txt` -- paper and cardboard
- `Regels_pmd.txt` -- plastic, metal, and drink cartons (PMD)
- `Regels_restafval.txt` -- residual waste

## regions/

Region- and country-specific disposal rules, one `.txt` file per intercommunale
or country. These files are searched by **BM25 keyword matching** when the user
mentions a city or region.

The BM25 index is rebuilt automatically at chatbot startup -- no manual rebuild needed.
Add a new `.txt` file here to support a new region immediately.

Each file follows this format:
```
Region/Intercommunale: <name>
Cities: <comma-separated list>
Country: <country>
----------------------------------------
Waste Bag Colors:
- Residual waste: <description>
- PMD: <description>
...
```
