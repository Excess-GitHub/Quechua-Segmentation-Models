# Segmentación Morfológica del Quechua

**🌐 Idioma / Language:** [English](README.md) | Español

---

> **Segmentación Morfológica Supervisada para el Quechua Sureño: Priors, Filtros y Aumento con LLM**

Un conjunto de herramientas integral para la segmentación morfológica del quechua sureño, con arquitecturas neuronales aumentadas con priors lingüísticamente informados.

## అ Resumen de Resultados

### Resultados en Conjunto de Prueba (913 palabras)

| Modelo | EM | +Filtro | B-F1 |
|--------|:--:|:-------:|:----:|
| Transformer Seq2Seq | 43.2% | — | — |
| BiLSTM (Carácter) | 52.7% | — | 0.817 |
| BiLSTM (Grafema) | 56.1% | — | 0.840 |
| BiLSTM + Morfessor | 55.1% | — | 0.838 |
| BiLSTM + Prior DT | 54.1% | 64.8% | 0.815 |
| BiLSTM + Prior HMM | 57.4% | 66.6% | 0.822 |
| **BiLSTM + HMM + GPT-4o (200)** | **63.9%** | **74.2%** | **0.898** |

**EM** = Coincidencia Exacta, **+Filtro** = con filtro de rechazo, **B-F1** = F1 de Fronteras (sin filtro)

### Resultados de Validación Cruzada (5 pliegues)

| Modelo | VC EM | VC B-F1 |
|--------|:-----:|:-------:|
| Transformer Seq2Seq | 60.5 ± 1.5% | — |
| BiLSTM-CRF | 84.9 ± 1.5% | — |
| BiLSTM + Prior DT | 84.2 ± 1.6% | 0.960 ± 0.004 |
| BiLSTM + Prior HMM | 85.8 ± 1.2% | 0.952 ± 0.005 |

### Hallazgos Principales

1. **BiLSTM > Transformer**: Los modelos basados en BiLSTM superan sustancialmente a los Transformers (+13.9% EM), confirmando que los sesgos inductivos importan en escenarios de bajos recursos.

2. **La tokenización por grafemas ayuda**: Respetar los dígrafos del quechua (ch, ll, ph, etc.) proporciona mejoras modestas pero consistentes (+3.4% EM).

3. **Los priors lingüísticos mejoran**: El prior HMM de sufijos logra 57.4% vs 56.1% de la línea base de grafemas (+1.3%).

4. **Los filtros de rechazo son cruciales**: Las restricciones morfotácticas duras en la inferencia proporcionan +9.2% de mejora en EM.

5. **El aumento con LLM es efectivo**: GPT-4o con 200 ejemplos sintéticos produce nuestro mejor resultado (74.2% EM).

6. **GPT-4o > GPT-5 para esta tarea**: La fidelidad en el seguimiento de instrucciones importa más que la capacidad bruta para tareas morfológicas específicas.

## ఆ Arquitectura

```
Palabra → Tokenizador de Grafemas → Codificador BiLSTM → Logits de Frontera → Filtro de Rechazo → Palabra Segmentada
                                          ↑
                                  Prior HMM/DT (suave)
```

El sistema combina:
- **Priors suaves** (HMM o Árbol de Decisión) durante el entrenamiento mediante fusión a nivel de logits
- **Restricciones duras** (filtro de rechazo de sufijos) en la inferencia

## ఇ Estructura del Repositorio

```
quechua-segmentation/
├── src/
│   ├── __init__.py          # Exportaciones del paquete
│   ├── preprocessing.py     # Tokenización, normalización
│   ├── models.py            # Arquitecturas neuronales y priors
│   ├── evaluation.py        # Métricas y utilidades de evaluación
│   └── training.py          # Bucles de entrenamiento y checkpoints
├── notebooks/               # Notebooks Jupyter originales
├── images/                  # Figuras y visualizaciones
├── data/                    # Archivos de datos (ver sección Datos)
├── models/                  # Checkpoints de modelos entrenados
└── README.md
```

## ఈ Inicio Rápido

### Instalación

```bash
pip install torch numpy pandas scikit-learn morfessor regex
```

### Uso Básico

```python
from src import (
    to_graphemes, 
    BiLSTMBoundary, 
    HMMSuffixPrior,
    SuffixRejectionFilter,
    apply_boundaries
)

# Tokenizar una palabra
tokens = to_graphemes("rikuchkani")
# ['r', 'i', 'k', 'u', 'ch', 'k', 'a', 'n', 'i']

# Cargar modelo entrenado y predecir
model = BiLSTMBoundary(vocab_size=42, emb_dim=64, hidden_size=128)
# ... cargar pesos ...

# Aplicar prior HMM
hmm_prior = HMMSuffixPrior()
hmm_prior.fit(training_morph_splits)
prior_probs = hmm_prior.predict_probs(tokens)

# Obtener predicciones y aplicar filtro
boundary_labels = [0, 0, 0, 1, 0, 0, 1, 0]  # del modelo
segments = apply_boundaries(tokens, boundary_labels)
# ['riku', 'chka', 'ni']

# Filtro de rechazo
filter = SuffixRejectionFilter(suffix_vocabulary)
filtered = filter.filter("rikuchkani", segments)
```

## ఉ Datos

### Datos de Entrenamiento (Privados)
- **6,896 palabras únicas** de transcripciones de entrevistas con ~70 hablantes adultos de quechua
- Recopilados bajo aprobación IRB con consentimiento informado
- Anotados por dos consultores ancianos de la comunidad
- Disponibles bajo solicitud en forma desidentificada bajo condiciones de acceso controlado

### Datos de Prueba (Públicos)
- **913 palabras únicas** separadas antes del entrenamiento
- Publicados con este repositorio

### Estadísticas del Corpus

Nuestros datos de entrenamiento exhiben una fuerte correlación entre la longitud de palabra y el conteo de morfemas (Pearson r = 0.79, p < 0.001):

![Longitud de Palabra vs Conteo de Morfemas](images/heatmap.png)

*Mapa de calor mostrando la relación entre longitud de palabra (caracteres) y número de morfemas. El patrón diagonal refleja la restricción de raíces bisilábicas del quechua y la sufijación regular.*

![Regresión Lineal](images/regression.png)

*Relación lineal: morfemas ≈ 0.28 × longitud + 0.32 (R² = 0.63)*

Para el corpus público (~2.1M tokens, 206K tipos):

![Ley de Heaps](images/heaps.png)

*El crecimiento del vocabulario sigue la ley de Heaps con β = 0.90 (R² = 0.98), indicando productividad continua.*

![Zipf-Mandelbrot](images/zipf.png)

*Distribución de frecuencia de palabras con ajuste Zipf-Mandelbrot (s = 1.06, q = 6.0).*

## ఊ Modelos

### Etiquetador de Fronteras BiLSTM
BiLSTM a nivel de carácter/grafema con predicción de fronteras por posición.

```python
from src.models import BiLSTMBoundary

model = BiLSTMBoundary(
    vocab_size=42,
    emb_dim=64,
    hidden_size=128,
    num_layers=2,
    dropout=0.1
)
```

### BiLSTM con Priors Lingüísticos
Integra priors HMM o Árbol de Decisión mediante fusión a nivel de logits.

```python
from src.models import BiLSTMWithPrior, HMMSuffixPrior

prior = HMMSuffixPrior(max_suffix_len=8)
prior.fit(morph_splits)

model = BiLSTMWithPrior(vocab_size=42, prior_alpha=1.0)
```

### Filtro de Rechazo
Validación post-procesamiento contra vocabulario de sufijos conocidos.

```python
from src.models import SuffixRejectionFilter

filter = SuffixRejectionFilter(suffix_set)
valid = filter.validate(["riku", "chka", "ni"])  # True
valid = filter.validate(["ri", "ku", "xyz"])     # False
```

## ఋ Aumento con LLM

Usamos GPT-4o para generar ejemplos de entrenamiento sintéticos:

1. **Selección de candidatos**: Palabras del corpus público de quechua que coinciden con patrones de raíces en los datos de entrenamiento
2. **Prompting few-shot**: 37 pares de demostración, temperatura=0
3. **Validación**: Rechazar salidas que contengan palabras en inglés o formato incorrecto
4. **Cantidad óptima**: 200 ejemplos (más puede perjudicar el rendimiento)

El pipeline respeta la gobernanza de datos al no exponer datos de entrenamiento privados a APIs externas.

## ౠ Citación

```bibtex
@inproceedings{anonymous2026quechua,
  title={Supervised Morphological Segmentation for Southern Quechua: 
         Priors, Filters, and LLM Augmentation},
  author={Anonymous},
  booktitle={Proceedings of ACL 2026},
  year={2026}
}
```

## ఎ Ética y Gobernanza de Datos

- Datos de entrenamiento recopilados bajo aprobación IRB con consentimiento informado
- Los ancianos de la comunidad sirvieron como anotadores con compensación justa a tarifas de EE.UU.
- Honramos los principios de soberanía de datos indígenas (CARE)
- Los datos privados no se exponen a APIs externas
- La publicación pública prioriza aplicaciones de mantenimiento del idioma y educación

## ఏ Licencia

El código se publica bajo Licencia MIT. El conjunto de prueba de 913 palabras se publica para fines de investigación.

## ఐ Agradecimientos

Agradecemos a las comunidades quechuahablantes que contribuyeron datos y experiencia lingüística, y a los consultores ancianos de la comunidad que proporcionaron las anotaciones.
