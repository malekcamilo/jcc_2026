## Domain Adaptation of Large Language Models: A Comparison between Full Fine-Tuning and LoRA
---
#### Código fuente que acompaña al artículo presentado en JCC 2026.

El conjunto de datos utilizado en los experimentos se construyó a partir de los documentos pertenecientes al dominio de Ciencias de la Computación del repositorio institucional SEDICI, abarcando un total de 19.974 registros con sus respectivos metadatos. Cada instancia consta del título y el resumen del documento. El tipo de documento se empleó como variable objetivo para la tarea de clasificación supervisada. 

Para solicitar acceso al conjunto de datos, comuníquese a: malek.camilo@gmail.com.

---
#### Librerías utilizadas

| Paquete | Versión |
| :--- | :--- |
| **accelerate** | 1.13.0 |
| **datasets** | 4.8.4 |
| **dpcpp-cpp-rt** | 2025.2.1 |
| **intel-sycl-rt** | 2025.2.1 |
| **numpy** | 2.4.4 |
| **peft** | 0.18.1 |
| **pytorch-triton-xpu** | 3.5.0 |
| **scikit-learn** | 1.8.0 |
| **torch** | 2.9.0+xpu |
| **transformers** | 5.5.0 |
| **trl** | 1.0.0 |

---

#### Troubleshooting

En caso de experimentar errores de falta de memoria (Out of Memory - OOM) durante el entrenamiento Full Fine-Tuning (FFT), es necesario realizar la siguiente modificación en la configuración de la clase `SFTTrainer` (mediante `SFTConfig`):

```python
training_args = SFTConfig(
    ...
    # Reducido de 8 a 1 para evitar OOM
    per_device_train_batch_size=1,
    # Añadido explícitamente para prevenir OOM en evaluación
    per_device_eval_batch_size=1, 
    # Incrementado para mantener el tamaño de batch efectivo global
    gradient_accumulation_steps=8, 
    ...
)
```
