# Everwod IA FAQs

Sistema inteligente para generación automática de FAQs utilizando IA, embeddings semánticos, clustering y revisión humana.

---

# Arquitectura del Proyecto

```text
Frontend React (5173)
        ↓
review.py (8004)
        ↓
PostgreSQL
        ↑
suggestion.py (8003)
        ↑
conversations.jsonl
        ↑
ingestion.py (8001)
```

---

# Tecnologías Utilizadas

## Backend
- Python 3.12
- FastAPI
- Uvicorn
- PostgreSQL
- psycopg2

## IA / NLP
- SentenceTransformers
- all-MiniLM-L6-v2
- Transformers
- Qwen2.5-0.5B-Instruct
- DBSCAN
- Scikit-learn

## Frontend
- React
- Vite
- JavaScript

---

# Funcionalidades

- Generación automática de FAQs
- Clustering semántico
- Embeddings NLP
- Filtrado por empresa
- Dashboard de revisión humana
- Aprobar/Rechazar FAQs
- Persistencia en PostgreSQL
- Arquitectura basada en microservicios

---

# Estructura del Proyecto

```text
Everwod_IA_FAQs/
│
├── frontend/
│   └── React Dashboard
│
├── data/
│   ├── conversations.jsonl
│   └── faq_suggestions.json
│
├── ingestion.py
├── suggestion.py
├── review.py
├── embeddings.py
├── faq_core.py
├── faq_model.py
├── faq_scheduler.py
├── validation.py
│
└── README.md
```

---

# Instalación

## 1. Clonar repositorio

```bash
git clone https://github.com/Anyelith-10/Everwod_IA_FAQs.git
```

---

## 2. Entrar al proyecto

```bash
cd Everwod_IA_FAQs
```

---

## 3. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

---

## 4. Instalar frontend

```bash
cd frontend
npm install
```

---

# Inicialización del Sistema

Abrir 4 terminales.

---

## Terminal 1 — ingestion.py

```bash
py -m uvicorn ingestion:app --reload --port 8001
```

---

## Terminal 2 — suggestion.py

```bash
py -m uvicorn suggestion:app --reload --port 8003
```

---

## Terminal 3 — review.py

```bash
py -m uvicorn review:app --reload --port 8004
```

---

## Terminal 4 — Frontend React

```bash
cd frontend
npm run dev
```

---

# Acceso al Sistema

## Frontend

```text
http://localhost:5173
```

---

## API Review

```text
http://127.0.0.1:8004/docs
```

---

## API Suggestions

```text
http://127.0.0.1:8003/docs
```

---

# Flujo del Sistema

1. Se cargan conversaciones
2. Se generan embeddings
3. Se agrupan preguntas similares
4. Se generan FAQs mediante IA
5. FAQs quedan en estado pending
6. Usuario revisa FAQs
7. FAQs aprobadas pasan a producción

---

# Embeddings

Modelo utilizado:

```text
all-MiniLM-L6-v2
```

Utilizado para:
- similitud semántica
- clustering
- detección de duplicados

---

# Clustering

Algoritmo:

```text
DBSCAN
```

Permite:
- agrupación automática
- detección de ruido
- FAQs repetidas

---

# Modelo LLM

Modelo utilizado:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Utilizado para:
- generación de respuestas
- reformulación de preguntas
- FAQs corporativas

---

# Frontend Dashboard

Funciones:
- visualizar FAQs
- aprobar
- rechazar
- generar FAQs
- filtrar por empresa

---

# Estados de FAQs

| Estado | Descripción |
|---|---|
| pending | FAQ pendiente |
| approved | FAQ aprobada |
| rejected | FAQ rechazada |
