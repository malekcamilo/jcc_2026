## Performance vs. Efficiency in Domain-Specific LLM Adaptation: A Comparative Study of Full Fine-Tuning and LoRA on Real-World Data
---
#### Source code accompanying the paper presented at JCC 2026

The dataset is constructed from all documents in the Computer Science domain of the SEDICI institutional repository, comprising a total of 19,974 records with associated metadata. Each instance is composed of the document title and abstract, while the document type is used as the target variable for the classification task.

To request access to the dataset, please contact: malek.camilo@gmail.com.

---
#### Python version
```
$ python --version
Python 3.11.15
```

#### Used Libraries

| Package | Version |
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

To avoid Out of Memory (OOM) errors during Full Fine-Tuning (FFT), modify the `SFTConfig` parameters passed to the `SFTTrainer` class as follows:

```python
training_args = SFTConfig(
    ...
    # Reduced from 8 to 1 to avoid OOM
    per_device_train_batch_size=1,
    # Added explicitly to prevent OOM during evaluation
    per_device_eval_batch_size=1, 
    # Increased to maintain the effective global batch size
    gradient_accumulation_steps=8, 
    ...
)
```
