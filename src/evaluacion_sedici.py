import argparse
import json
import torch
import unicodedata

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

def format_prompt(example, is_training=False, eos_token="<|eot_id|>"):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Eres un clasificador automático de documentos académicos. "
        "Tu única tarea es asignar la categoría correcta a los registros bibliográficos provistos.\n"
        "Regla estricta: Debes responder ÚNICAMENTE con el nombre exacto de la categoría. No incluyas explicaciones, puntuación adicional ni texto conversacional.\n"
        "Categorías válidas: Articulo, Objeto de conferencia, Tesis, Libro, Otro, Objeto de aprendizaje.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Clasifica el siguiente registro:\n"
        f"<titulo>{example['title']}</titulo>\n"
        f"<resumen>{example['abstract']}</resumen><|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if is_training:
        prompt += example['type'] + eos_token
    return prompt

VALID_CATEGORIES = ["objeto de conferencia", "objeto de aprendizaje", "articulo", "tesis", "libro", "otro"]

def normalizar_texto(texto):
    texto = str(texto).lower().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def extraer_categoria(texto):
    texto_norm = normalizar_texto(texto)
    for category in VALID_CATEGORIES:
        if category in texto_norm:
            return category
    return "sin_clasificar"

def evaluate_model(model, tokenizer, test_dataset):
    model.eval()
    predictions = []
    references = []
    
    exact_matches = 0
    exact_matches_per_class = {cat: 0 for cat in VALID_CATEGORIES}
    total_per_class = {cat: 0 for cat in VALID_CATEGORIES}

    tokenizer.padding_side = "left"
    batch_size = 16 
    
    dataset_list = list(test_dataset)
    
    print(f"Iniciando inferencia vectorizada sobre test (Batch Size: {batch_size})...")
    
    for i in range(0, len(dataset_list), batch_size):
        batch = dataset_list[i:i+batch_size]
        prompts = [format_prompt(item, is_training=False) for item in batch]
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True
            )

        for j, item in enumerate(batch):
            input_length = inputs.input_ids[j].shape[0]
            generated_text = tokenizer.decode(outputs[j][input_length:], skip_special_tokens=True).strip()
            raw_pred = generated_text.split('\n')[0].strip()

            norm_true = normalizar_texto(item['type'])
            
            if norm_true in total_per_class:
                total_per_class[norm_true] += 1
            
            if raw_pred == item['type']:
                exact_matches += 1
                if norm_true in exact_matches_per_class:
                    exact_matches_per_class[norm_true] += 1

            predictions.append(extraer_categoria(generated_text))
            references.append(norm_true)

    tokenizer.padding_side = "right"

    total = len(dataset_list)
    em_global = exact_matches / total if total > 0 else 0.0
    acc = accuracy_score(references, predictions)
    macro_f1 = f1_score(references, predictions, average='macro', zero_division=0)
    
    # Calcular EM por clase
    em_per_class = {}
    for cat in VALID_CATEGORIES:
        if total_per_class[cat] > 0:
            em_per_class[cat] = exact_matches_per_class[cat] / total_per_class[cat]
        else:
            em_per_class[cat] = 0.0
            
    # Obtener F1 por clase (gracias a sklearn)
    report = classification_report(references, predictions, output_dict=True, zero_division=0)
    
    return acc, macro_f1, em_global, em_per_class, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["base", "fft", "lora"], required=True)
    parser.add_argument("--lora_r", type=int, default=8, help="Rango de LoRA (alpha = 2*r)")
    args = parser.parse_args()

    datasets = load_dataset("json", data_files={
        "test": "test_strat_norm.jsonl"
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    suffix = f"{args.method}_r{args.lora_r}" if args.method == "lora" else args.method

    if args.method == "base":
        print("Evaluando modelo base sin fine-tuning...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map={"": "xpu:0"}
        )
    elif args.method == "lora":
        print(f"Cargando modelo base y acoplando adaptador LoRA desde ./resultados_{suffix}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map={"": "xpu:0"}
        )
        model = PeftModel.from_pretrained(base_model, f"./resultados_{suffix}")
    elif args.method == "fft":
        print(f"Cargando modelo FFT completo desde ./resultados_{suffix}")
        model = AutoModelForCausalLM.from_pretrained(
            f"./resultados_{suffix}",
            dtype=torch.bfloat16,
            device_map={"": "xpu:0"}
        )

    acc, macro_f1, em_global, em_per_class, report = evaluate_model(model, tokenizer, datasets["test"])

    method_str = f"LORA (r={args.lora_r})" if args.method == "lora" else args.method.upper()
    print("\n" + "=" * 50)
    print(f"{'REPORTE DE EVALUACIÓN':^50}")
    print("=" * 50)
    print(f"{'Método:':<12} {method_str}")
    print(f"{'Accuracy:':<12} {acc:.4f}")
    print(f"{'Macro F1:':<12} {macro_f1:.4f}")
    print(f"{'Global EM:':<12} {em_global:.4f}")
    
    print("-" * 50)
    print(f"{'MÉTRICAS POR CLASE':^50}")
    print("-" * 50)
    print(f"{'Clase':<25} | {'F1-Score':<10} | {'Exact Match':<10}")
    print("-" * 50)
    for cat in VALID_CATEGORIES:
        f1_c = report.get(cat, {}).get('f1-score', 0.0)
        em_c = em_per_class.get(cat, 0.0)
        print(f"{cat.title():<25} | {f1_c:.4f}     | {em_c:.4f}")
    
    result_path = f"./metricas_eval_{suffix}.json"
    print("=" * 50)
    print(f"{'Archivo:':<12} {result_path}")
    print("=" * 50 + "\n")

    # Preparar el guardado detallado
    resultados_eval = {
        "config": {
            "fase": "eval",
            "method": args.method,
            "lora_r": args.lora_r if args.method == "lora" else None
        },
        "metrics": {
            "global": {
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "exact_match": round(em_global, 4)
            },
            "per_class": {}
        }
    }
    
    for cat in VALID_CATEGORIES:
        resultados_eval["metrics"]["per_class"][cat] = {
            "f1_score": round(report.get(cat, {}).get('f1-score', 0.0), 4),
            "exact_match": round(em_per_class.get(cat, 0.0), 4),
            "precision": round(report.get(cat, {}).get('precision', 0.0), 4),
            "recall": round(report.get(cat, {}).get('recall', 0.0), 4),
            "support": report.get(cat, {}).get('support', 0)
        }

    with open(result_path, 'w') as f:
        json.dump(resultados_eval, f, indent=2)

if __name__ == "__main__":
    main()
