#!/usr/bin/env python3
"""
Evaluator para Fine-Tuning vs RAG
Apex Eval - 2025

Ejecuta evaluación comparativa de diferentes arquitecturas sobre un dataset
de consultas empresariales.

Uso:
    python evaluator.py --config configs/rag_basic_config.yaml \
                        --dataset data/synthetic_dataset.jsonl \
                        --output results/rag_basic.json
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import yaml

from tqdm import tqdm
import numpy as np
from openai import OpenAI

from metrics import compute_f1_score, compute_latency_stats, compute_cost
from rag.retriever import RAGRetriever
from rag.generator import RAGGenerator
from fine_tuning.inference import FineTunedModel


@dataclass
class EvaluationResult:
    """Resultado de evaluación individual"""
    query_id: str
    query: str
    prediction: str
    ground_truth: str
    
    # Métricas
    f1_score: float
    latency_ms: float
    tokens_input: int
    tokens_output: int
    cost_soles: float
    
    # Metadata
    category: str
    complexity: str
    timestamp: str


class Evaluator:
    """Evaluador de sistemas de QA"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.client = OpenAI()
        self.results: List[EvaluationResult] = []
        
        # Inicializar sistema según configuración
        self._setup_system()
    
    def _setup_system(self):
        """Inicializa el sistema a evaluar según config"""
        system_type = self.config.get('name', '')
        
        if 'RAG' in system_type:
            print(f"Inicializando sistema RAG: {system_type}")
            self.retriever = RAGRetriever(self.config)
            self.generator = RAGGenerator(self.config)
            self.system_type = 'rag'
            
        elif 'FT' in system_type or 'Fine' in system_type:
            print(f"Inicializando modelo fine-tuned: {system_type}")
            self.model = FineTunedModel(self.config)
            self.system_type = 'fine_tuned'
            
        else:
            raise ValueError(f"Tipo de sistema no reconocido: {system_type}")
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Carga dataset en formato JSONL"""
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))
        
        print(f"Dataset cargado: {len(dataset)} consultas")
        return dataset
    
    def evaluate_query(self, query_data: Dict[str, Any]) -> EvaluationResult:
        """Evalúa una consulta individual"""
        
        start_time = time.time()
        
        # Obtener predicción según tipo de sistema
        if self.system_type == 'rag':
            # Retrieval + Generation
            docs = self.retriever.retrieve(query_data['query'])
            prediction, usage = self.generator.generate(
                query=query_data['query'],
                context_docs=docs
            )
        else:
            # Fine-tuned model
            prediction, usage = self.model.predict(query_data['query'])
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calcular F1-score usando GPT-4 como juez
        f1_score = compute_f1_score(
            prediction=prediction,
            ground_truth=query_data['ground_truth'],
            judge_model='gpt-4o'
        )
        
        # Calcular costo
        cost_soles = compute_cost(
            tokens_input=usage['prompt_tokens'],
            tokens_output=usage['completion_tokens'],
            config=self.config
        )
        
        return EvaluationResult(
            query_id=query_data['id'],
            query=query_data['query'],
            prediction=prediction,
            ground_truth=query_data['ground_truth'],
            f1_score=f1_score,
            latency_ms=latency_ms,
            tokens_input=usage['prompt_tokens'],
            tokens_output=usage['completion_tokens'],
            cost_soles=cost_soles,
            category=query_data.get('category', 'unknown'),
            complexity=query_data.get('complexity', 'unknown'),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def run_evaluation(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ejecuta evaluación completa sobre el dataset"""
        
        print(f"\nEvaluando {len(dataset)} consultas...")
        
        for query_data in tqdm(dataset):
            try:
                result = self.evaluate_query(query_data)
                self.results.append(result)
            except Exception as e:
                print(f"\nError en query {query_data['id']}: {e}")
                continue
        
        # Calcular métricas agregadas
        return self._compute_aggregate_metrics()
    
    def _compute_aggregate_metrics(self) -> Dict[str, Any]:
        """Calcula métricas agregadas de todos los resultados"""
        
        f1_scores = [r.f1_score for r in self.results]
        latencies = [r.latency_ms for r in self.results]
        costs = [r.cost_soles for r in self.results]
        
        # Estadísticas generales
        metrics = {
            'system': self.config['name'],
            'n_queries': len(self.results),
            
            # Precisión
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'f1_median': np.median(f1_scores),
            
            # Latencia
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'latency_mean': np.mean(latencies),
            
            # Costos
            'cost_per_query': np.mean(costs),
            'cost_total': np.sum(costs),
            'cost_monthly_5k': np.mean(costs) * 5000,  # Proyección 5K queries/mes
            
            # Por categoría
            'by_category': self._metrics_by_group('category'),
            'by_complexity': self._metrics_by_group('complexity')
        }
        
        return metrics
    
    def _metrics_by_group(self, group_field: str) -> Dict[str, Dict[str, float]]:
        """Calcula métricas segmentadas por un campo"""
        
        grouped = {}
        for result in self.results:
            group = getattr(result, group_field)
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(result)
        
        metrics = {}
        for group, results in grouped.items():
            f1_scores = [r.f1_score for r in results]
            metrics[group] = {
                'n': len(results),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores)
            }
        
        return metrics
    
    def save_results(self, output_path: str):
        """Guarda resultados en JSON"""
        
        output = {
            'config': self.config,
            'aggregate_metrics': self._compute_aggregate_metrics(),
            'individual_results': [asdict(r) for r in self.results]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Resultados guardados en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluador Fine-Tuning vs RAG - Apex Eval'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Ruta al archivo de configuración YAML'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Ruta al dataset en formato JSONL'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Ruta donde guardar resultados JSON'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limitar evaluación a N queries (útil para testing)'
    )
    
    args = parser.parse_args()
    
    # Inicializar evaluador
    evaluator = Evaluator(args.config)
    
    # Cargar dataset
    dataset = evaluator.load_dataset(args.dataset)
    
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"⚠️  Limitando evaluación a {args.limit} queries")
    
    # Ejecutar evaluación
    metrics = evaluator.run_evaluation(dataset)
    
    # Mostrar resumen
    print("\n" + "="*50)
    print("RESUMEN DE RESULTADOS")
    print("="*50)
    print(f"\nSistema: {metrics['system']}")
    print(f"Queries evaluadas: {metrics['n_queries']}")
    print(f"\nPrecisión (F1):")
    print(f"  Mean: {metrics['f1_mean']:.3f} (±{metrics['f1_std']:.3f})")
    print(f"  Median: {metrics['f1_median']:.3f}")
    print(f"\nLatencia:")
    print(f"  p50: {metrics['latency_p50']:.0f}ms")
    print(f"  p95: {metrics['latency_p95']:.0f}ms")
    print(f"\nCostos:")
    print(f"  Por query: S/. {metrics['cost_per_query']:.4f}")
    print(f"  Proyección 5K queries/mes: S/. {metrics['cost_monthly_5k']:.2f}")
    
    # Guardar resultados
    evaluator.save_results(args.output)


if __name__ == '__main__':
    main()
