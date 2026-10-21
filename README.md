# Entrega 3 — RAG local

Sistema end-to-end de recuperación aumentada por generación (RAG) local. Indexa documentos `.txt` y `.md` de `data/` en ChromaDB, recupera los fragmentos pertinentes y responde usando **únicamente** ese contexto.

La generación usa Groq y el vectorstore/embeddings quedan locales. Si la información no aparece en los fragmentos recuperados, el asistente responde exactamente **“No lo sé”**.

## Características

- Ingesta reproducible de archivos `.txt` y `.md`.
- `RecursiveCharacterTextSplitter` con `chunk_size=2000` y `chunk_overlap=200`: aproximación conservadora a 500 y 50 tokens, respectivamente.
- ChromaDB persistente en `vectorstore/` y manifiesto con huella de documentos: evita reindexar si no cambió `data/`.
- El mismo modelo de `HuggingFaceEmbeddings` para indexación y consulta.
- Retriever `top_k=4` (configurable entre 3 y 5).
- Cadena LCEL asíncrona `get_rag_response(query: str)`.
- Respuesta validada con `PydanticOutputParser`: `respuesta` y `referencias`.
- Datos de muestra coherentes: manual operativo del sistema de préstamo de una biblioteca.


## Uso

Primero cargue o actualice el índice:

```powershell
python -m src.ingestion
```

Luego consulte en modo interactivo:

```powershell
python -m src.main
```

O haga una consulta puntual:

```powershell
python -m src.main "¿Cuántas renovaciones permite la biblioteca?"
```

Para forzar una reindexación, por ejemplo tras cambiar parámetros de fragmentación:

```powershell
python -m src.ingestion --force
```

## Pruebas

Las pruebas unitarias no requieren una clave de Groq ni descargan modelos:

```powershell
pytest
```

Incluyen una pregunta respondible y una pregunta trampa que comprueba el comportamiento esperado de groundedness.

## Estructura

```text
data/             Documentos fuente de ejemplo
src/              Ingesta, configuración, cadena LCEL y CLI
tests/            Pruebas unitarias
vectorstore/      Se crea localmente; no se publica
```

## Nota de diseño

El modelo de lenguaje puede ser sustituido mediante `GROQ_MODEL`. El modelo de embeddings debe mantenerse idéntico durante la indexación y las consultas; por eso se centraliza en `src/config.py` y se registra en el manifiesto de Chroma.
