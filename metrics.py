"""
Módulo de cálculo de métricas para evaluación
Odysea Eval - 2025
"""

import re
from typing import Dict, Any
from openai import OpenAI


# Tipos de cambio y costos (abril 2025)
USD_TO_SOLES = 3.75

OPENAI_COSTS_USD = {
    # Embeddings
    'text-embedding-3-small': {
        'input': 0.02 / 1_000_000
    },
    'text-embedding-3-large': {
        'input': 0.13 / 1_000_000
    },
    
    # Fine-tuning
    'gpt-3.5-turbo-1106': {
        'input': 3.00 / 1_000_000,
        'output': 6.00 / 1_000_000,
        'training': 8.00 / 1_000_000  # Por epoch
    },
    'gpt-4-0613': {
        'input': 30.00 / 1_000_000,
        'output': 60.00 / 1_000_000,
        'training': 80.00 / 1_000_000
    },
    
    # Modelos estándar
    'gpt-3.5-turbo-0125': {
        'input': 0.50 / 1_000_000,
        'output': 1.50 / 1_000_000
    },
    'gpt-4o-2024-08-06': {
        'input': 2.50 / 1_000_000,
        'output': 10.00 / 1_000_000
    },
    'gpt-4-turbo-2024-04-09': {
        'input': 10.00 / 1_000_000,
        'output': 30.00 / 1_000_000
    }
}


def compute_f1_score(
    prediction: str,
    ground_truth: str,
    judge_model: str = 'gpt-4o'
) -> float:
    """
    Calcula F1-score usando GPT-4 como juez.
    
    Args:
        prediction: Respuesta generada por el sistema
        ground_truth: Respuesta correcta validada
        judge_model: Modelo a usar como evaluador
        
    Returns:
        F1-score entre 0.0 y 1.0
    """
    
    client = OpenAI()
    
    judge_prompt = f"""Eres un evaluador experto que compara respuestas.

RESPUESTA CORRECTA (ground truth):
{ground_truth}

RESPUESTA DEL MODELO:
{prediction}

Evalúa la respuesta del modelo en una escala de 0.0 a 1.0:
- 1.0 = Completamente correcta, captura toda la información relevante
- 0.5 = Parcialmente correcta, captura algunos elementos pero falta información crítica
- 0.0 = Completamente incorrecta o irrelevante

Considera:
1. Corrección factual (información correcta vs incorrecta)
2. Completitud (información faltante vs presente)
3. Relevancia (información relevante vs irrelevante)

Responde SOLO con un número entre 0.0 y 1.0, sin explicaciones.
"""
    
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "Eres un evaluador preciso y consistente."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        score_text = response.choices[0].message.content.strip()
        
        # Extraer número del texto
        match = re.search(r'(\d+\.?\d*)', score_text)
        if match:
            score = float(match.group(1))
            # Asegurar que esté en rango [0, 1]
            return max(0.0, min(1.0, score))
        else:
            print(f"⚠️  No se pudo parsear score: {score_text}")
            return 0.0
            
    except Exception as e:
        print(f"❌ Error al calcular F1 con juez: {e}")
        return 0.0


def compute_cost(
    tokens_input: int,
    tokens_output: int,
    config: Dict[str, Any]
) -> float:
    """
    Calcula costo en soles de una consulta.
    
    Args:
        tokens_input: Tokens de entrada (prompt + context)
        tokens_output: Tokens de salida (respuesta generada)
        config: Configuración del sistema (para extraer modelo)
        
    Returns:
        Costo en soles peruanos
    """
    
    # Determinar modelo según configuración
    if 'llm' in config:
        model = config['llm']['model']
    elif 'model' in config:
        model = config['model']
    else:
        raise ValueError("No se encontró modelo en config")
    
    # Buscar costos del modelo
    if model not in OPENAI_COSTS_USD:
        print(f"⚠️  Modelo {model} no tiene costos definidos, usando gpt-3.5-turbo-0125")
        model = 'gpt-3.5-turbo-0125'
    
    costs = OPENAI_COSTS_USD[model]
    
    # Calcular costo en USD
    cost_usd = (
        tokens_input * costs['input'] +
        tokens_output * costs['output']
    )
    
    # Si usa embeddings (RAG), agregar ese costo
    if 'embeddings' in config:
        # Asumimos que la query es ~50 tokens en promedio
        embedding_model = config['embeddings']['model']
        embedding_cost = 50 * OPENAI_COSTS_USD[embedding_model]['input']
        cost_usd += embedding_cost
    
    # Convertir a soles
    return cost_usd * USD_TO_SOLES


def compute_latency_stats(latencies: list) -> Dict[str, float]:
    """
    Calcula estadísticas de latencia.
    
    Args:
        latencies: Lista de latencias en milisegundos
        
    Returns:
        Diccionario con percentiles y promedios
    """
    import numpy as np
    
    return {
        'p50': np.percentile(latencies, 50),
        'p75': np.percentile(latencies, 75),
        'p90': np.percentile(latencies, 90),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }


def compute_confidence_interval(scores: list, confidence: float = 0.95) -> tuple:
    """
    Calcula intervalo de confianza mediante bootstrap.
    
    Args:
        scores: Lista de scores (ej: F1 scores)
        confidence: Nivel de confianza (default 0.95)
        
    Returns:
        (lower_bound, upper_bound)
    """
    import numpy as np
    from scipy import stats
    
    n = len(scores)
    mean = np.mean(scores)
    std_error = stats.sem(scores)
    
    # Intervalo de confianza usando distribución t
    ci = stats.t.interval(
        confidence,
        n - 1,
        loc=mean,
        scale=std_error
    )
    
    return ci


def detect_hallucination(
    prediction: str,
    context_docs: list,
    judge_model: str = 'gpt-4o'
) -> bool:
    """
    Detecta si la predicción contiene alucinaciones (información no presente en contexto).
    
    Args:
        prediction: Respuesta generada
        context_docs: Documentos usados como contexto
        judge_model: Modelo evaluador
        
    Returns:
        True si se detecta alucinación, False si no
    """
    
    client = OpenAI()
    
    context_text = "\n\n".join([doc['text'] for doc in context_docs])
    
    prompt = f"""Analiza si la RESPUESTA contiene información que NO está presente en los DOCUMENTOS.

DOCUMENTOS:
{context_text}

RESPUESTA:
{prediction}

¿La respuesta contiene afirmaciones específicas (nombres, números, fechas, hechos) que NO están en los documentos?

Responde SOLO "SÍ" o "NO".
"""
    
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "Eres un detector de alucinaciones preciso."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return 'SÍ' in answer or 'SI' in answer
        
    except Exception as e:
        print(f"❌ Error al detectar alucinación: {e}")
        return False
