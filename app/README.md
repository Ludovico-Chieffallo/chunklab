---
title: ChunkLab
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app/app.py
pinned: false
license: mit
short_description: Which chunking strategy retrieves your answers best?
---

# ChunkLab — Hugging Face Space

This Space runs the [chunklab](https://github.com/ludovicochieffallo/chunklab) demo.

Upload a document (PDF, DOCX, TXT, MD), enter a few questions with gold snippets
(`question :: gold snippet`), pick which chunking strategies to compare, and see
which one retrieves your answers best — with a downloadable HTML report.

Everything runs on CPU with a small local embedding model; no API key needed.
