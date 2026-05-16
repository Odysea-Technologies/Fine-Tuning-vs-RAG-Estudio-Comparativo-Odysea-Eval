# Fine-Tuning-vs-RAG-Estudio-Comparativo-Odysea-Eval

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Repositorio oficial del estudio **"Fine-Tuning vs RAG: Estudio comparativo de precisión, costo y latencia en 247 pruebas empresariales"** realizado por Odysea Eval entre enero-abril 2025.

## 📊 Resumen del Estudio

Este estudio evalúa cinco arquitecturas diferentes (Fine-Tuning GPT-3.5, Fine-Tuning GPT-4, RAG-Basic, RAG-Advanced, Baseline) en 247 tareas empresariales reales de 8 MYPES del sur del Perú.

**Hallazgo principal:** RAG-Basic supera a fine-tuning en precisión factual (F1: 0.84 vs 0.71) con 68% menos costo, pero fine-tuning domina en tareas de formato específico (0.91 vs 0.79).

### Métricas Clave

| Arquitectura | Precisión (F1) | Latencia p95 | Costo/consulta | Costo 12 meses* |
|--------------|----------------|--------------|----------------|-----------------|
| FT-GPT3.5 | 0.71 (±0.11) | 1,520ms | S/. 0.039 | S/. 3,587 |
| FT-GPT4 | 0.76 (±0.09) | 2,910ms | S/. 0.284 | S/. 18,647 |
| RAG-Basic | **0.84 (±0.07)** | 2,180ms | **S/. 0.048** | **S/. 3,327** |
| RAG-Advanced | 0.89 (±0.05) | 3,560ms | S/. 0.312 | S/. 18,868 |

*Asumiendo 5,000 consultas/mes

## 📁 Estructura del Repositorio

```
ft-vs-rag-benchmark/
├── README.md                          # Este archivo
├── LICENSE                            # Licencia MIT
├── requirements.txt                   # Dependencias Python
├── data/
│   ├── synthetic_dataset.jsonl       # 50 consultas sintéticas representativas
│   ├── schema.json                   # Schema del dataset
│   └── README.md                     # Descripción del dataset
├── configs/
│   ├── ft_gpt35_config.yaml          # Configuración Fine-Tuning GPT-3.5
│   ├── ft_gpt4_config.yaml           # Configuración Fine-Tuning GPT-4
│   ├── rag_basic_config.yaml         # Configuración RAG-Basic
│   └── rag_advanced_config.yaml      # Configuración RAG-Advanced
├── src/
│   ├── evaluation/
│   │   ├── metrics.py                # Cálculo de F1, latencia, costos
│   │   ├── evaluator.py              # Pipeline de evaluación
│   │   └── human_eval.py             # Utilidad para evaluación humana
│   ├── fine_tuning/
│   │   ├── prepare_data.py           # Preparación JSONL para fine-tuning
│   │   ├── train.py                  # Script de entrenamiento
│   │   └── inference.py              # Inferencia con modelos fine-tuned
│   ├── rag/
│   │   ├── indexer.py                # Indexación de documentos
│   │   ├── retriever.py              # Recuperación vectorial
│   │   └── generator.py              # Generación con contexto RAG
│   └── utils/
│       ├── openai_client.py          # Cliente OpenAI con retry logic
│       └── cost_calculator.py        # Cálculo de costos operacionales
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb # Análisis exploratorio del dataset
│   ├── 02_run_experiments.ipynb      # Ejecución de experimentos
│   └── 03_visualizations.ipynb       # Gráficos del paper
├── results/
│   ├── raw_results.csv               # Resultados crudos de 247 pruebas
│   ├── aggregated_metrics.json       # Métricas agregadas
│   └── plots/                        # Gráficos generados
└── docs/
    ├── METHODOLOGY.md                # Metodología detallada
    ├── REPRODUCIBILITY.md            # Guía de reproducción
    └── LIMITATIONS.md                # Limitaciones del estudio
```

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.11+
- API Key de OpenAI
- ChromaDB o Pinecone (para RAG)
- ~2GB de espacio en disco para embeddings

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/Odysea-eval/ft-vs-rag-benchmark.git
cd ft-vs-rag-benchmark

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu OpenAI API Key
```

### Ejecutar Evaluación Rápida

```bash
# Evaluar RAG-Basic en dataset sintético (50 consultas)
python src/evaluation/evaluator.py \
    --config configs/rag_basic_config.yaml \
    --dataset data/synthetic_dataset.jsonl \
    --output results/rag_basic_synthetic.json

# Ver resultados
python src/utils/report_generator.py results/rag_basic_synthetic.json
```

## 📖 Dataset Sintético

Por restricciones de confidencialidad, no podemos compartir las 247 consultas empresariales reales. En su lugar, proveemos:

**`data/synthetic_dataset.jsonl`**: 50 consultas sintéticas que replican la distribución y complejidad del dataset original:
- 12 consultas factuales directas
- 18 consultas de inferencia
- 20 consultas de formato específico

**Formato de cada entrada:**

```json
{
  "id": "query_001",
  "query": "¿Cuál es el plazo de garantía del producto SKU-A847?",
  "context": "Documento de políticas de garantía (fragmento anonimizado)",
  "ground_truth": "12 meses a partir de la fecha de compra",
  "category": "factual_direct",
  "industry": "retail",
  "complexity": "low"
}
```

Ver [`data/README.md`](data/README.md) para descripción completa del schema.

## 🔧 Configuraciones Evaluadas

### Fine-Tuning GPT-3.5

```yaml
model: gpt-3.5-turbo-1106
training_examples: 2847
epochs: 3
learning_rate_multiplier: 0.02
cost_training: S/. 1,247
cost_inference: S/. 0.039/query
```

### RAG-Basic (Configuración Recomendada para MYPES)

```yaml
embeddings: text-embedding-3-small
vector_db: ChromaDB (local)
llm: gpt-3.5-turbo-0125
top_k: 3
similarity_threshold: 0.75
streaming: enabled
cost_setup: S/. 87
cost_inference: S/. 0.048/query
```

Ver [`configs/`](configs/) para todas las configuraciones.

## 📊 Reproducir el Estudio

### Paso 1: Preparar tus Datos

```bash
# Si tienes documentación empresarial propia
python src/rag/indexer.py \
    --docs_path /path/to/your/docs \
    --output_db ./chroma_db \
    --embedding_model text-embedding-3-small
```

### Paso 2: Fine-Tuning (Opcional)

```bash
# Preparar datos de entrenamiento
python src/fine_tuning/prepare_data.py \
    --input data/your_qa_pairs.jsonl \
    --output data/training_data.jsonl

# Entrenar modelo
python src/fine_tuning/train.py \
    --training_file data/training_data.jsonl \
    --model gpt-3.5-turbo-1106 \
    --epochs 3
```

### Paso 3: Ejecutar Evaluación Completa

```bash
# Evaluar todas las configuraciones
python src/evaluation/evaluator.py \
    --mode full \
    --dataset data/synthetic_dataset.jsonl \
    --configs configs/*.yaml \
    --output results/full_evaluation.json

# Generar reporte con gráficos
python src/utils/report_generator.py \
    results/full_evaluation.json \
    --output results/report.html
```

Ver [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) para guía detallada.

## 📈 Resultados Principales

### Precisión por Tipo de Consulta

| Tipo de Consulta | FT-GPT3.5 | RAG-Basic | Delta |
|------------------|-----------|-----------|-------|
| Factual directa | 0.68 | **0.91** | +34% |
| Inferencia | 0.71 | 0.81 | +14% |
| Formato específico | **0.91** | 0.79 | -13% |

### Trade-off Costo vs Precisión

![Costo vs Precisión](results/plots/cost_vs_precision.png)

**Zona óptima para MYPES:** F1 >0.80, Costo <S/. 0.10/consulta  
→ Solo RAG-Basic cumple ambos criterios

### Proyección de Costos 12 Meses

![Proyección Costos](results/plots/cost_projection_12m.png)

RAG-Basic se vuelve más económico que FT-GPT3.5 después del mes 2 por ahorro en setup inicial.

## 🎯 Recomendaciones Basadas en Evidencia

**Para MYPES con corpus <2,000 páginas:**
- ✅ **Usar RAG-Basic** si >60% de consultas son factuales
- ✅ **Usar Fine-Tuning GPT-3.5** si >60% de consultas requieren formato específico

**Para empresas con >10,000 consultas/mes:**
- Evaluar fine-tuning (amortización de costo setup)
- Considerar RAG-Advanced solo si presupuesto >S/. 800/mes

**Evitar:**
- ❌ Fine-Tuning GPT-4 para MYPES (costo 7x superior sin ROI claro)
- ❌ RAG-Advanced para volúmenes <2,000 consultas/mes (no amortiza costo de Pinecone)

## 📄 Citación

Si usas este código o dataset en tu investigación, por favor cita:

```bibtex
@techreport{ramos2025ftvsrag,
  title={Fine-Tuning vs RAG: Estudio comparativo de precisión, costo y latencia en 247 pruebas empresariales},
  author={Ramos, Diego and Equipo Odysea Eval},
  institution={Odysea Eval, Odysea Technologies},
  year={2025},
  month={April},
  address={Arequipa, Perú},
  url={https://github.com/Odysea-eval/ft-vs-rag-benchmark}
}
```

**Paper completo:** [https://odysea.tech/research/ft-vs-rag-2025](https://odysea.tech/research/ft-vs-rag-2025)

## 🤝 Contribuciones

Este es un repositorio de investigación. Aceptamos:
- 🐛 Reportes de bugs en el código de evaluación
- 📊 Resultados de replicación en otros contextos (Latam, otros idiomas)
- 💡 Mejoras en metodología de evaluación

**No aceptamos:** PRs que modifiquen los datos o resultados originales del estudio (integridad científica).

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para lineamientos.

## 📜 Licencia

Este proyecto está licenciado bajo MIT License - ver [LICENSE](LICENSE) para detalles.

**Dataset sintético:** CC BY-SA 4.0  
**Paper/Reporte:** CC BY-NC-SA 4.0 (citar apropiadamente)

## 📧 Contacto

**Odysea Eval - Centro de Investigación Aplicada en IA**

- 🌐 Web: [https://Odyseatechnologies.pe/eval](https://odysea.tech/eval)
- 📧 Email: eval@odysea.tech
- 📍 Ubicación: Arequipa, Perú

**Lead Researcher:** Mauricio Lezama (mauriciolezama@odysea.tech)

## 🙏 Agradecimientos

- A las 8 MYPES de Arequipa, Cusco y Puno que contribuyeron datos anonimizados
- A la comunidad open-source de LangChain, ChromaDB, y OpenAI Python SDK
- A los revisores técnicos que validaron la metodología

---

**Última actualización:** 21 de abril, 2026  
**Versión del repositorio:** v1.0.0  
**DOI:** 10.5281/zenodo.XXXXXX (pendiente)

---

<div align="center">

**Hecho con ❤️ en Arequipa, Perú**

![Odysea Eval](https://odysea.tech)

</div>
