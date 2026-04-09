import argparse
import json
import time
import torch
import unicodedata

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, f1_score

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
                temperature=None,
                top_p=None,
                use_cache=True
            )

        for j, item in enumerate(batch):
            input_length = inputs.input_ids[j].shape[0]
            generated_text = tokenizer.decode(outputs[j][input_length:], skip_special_tokens=True).strip()
            raw_pred = generated_text.split('\n')[0].strip()

            if raw_pred == item['type']:
                exact_matches += 1

            predictions.append(extraer_categoria(generated_text))
            references.append(normalizar_texto(item['type']))

    tokenizer.padding_side = "right"

    total = len(dataset_list)
    em = exact_matches / total if total > 0 else 0.0
    acc = accuracy_score(references, predictions)
    macro_f1 = f1_score(references, predictions, average='macro', zero_division=0)
    
    return acc, macro_f1, em

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["base", "fft", "lora"], required=True)
    parser.add_argument("--lora_r", type=int, default=8, help="Rango de LoRA (alpha = 2*r)")
    parser.add_argument("--fase", type=str, choices=["train", "eval"], required=True)
    args = parser.parse_args()

    datasets = load_dataset("json", data_files={
        "train": "train_strat_norm.jsonl",
        "val": "val_strat_norm.jsonl",
        "test": "test_strat_norm.jsonl"
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def formatting_prompts_func(example):
        return format_prompt(example, is_training=True, eos_token=tokenizer.eos_token)

    suffix = f"{args.method}_r{args.lora_r}" if args.method == "lora" else args.method

    if args.fase == "train":
        if args.method == "base":
            print("El método 'base' no requiere entrenamiento. Finalizando.")
            return

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map={"": "xpu:0"} 
        )

        if args.method == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_r * 2,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            model = get_peft_model(model, peft_config)

        tasa_aprendizaje = 2e-4 if args.method == "lora" else 2e-5

        training_args = SFTConfig(
            output_dir=f"./resultados_{suffix}",
            learning_rate=tasa_aprendizaje,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            num_train_epochs=3,
            bf16=True,
            save_strategy="epoch",
            logging_strategy="epoch",
            max_length=512
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
            args=training_args,
            formatting_func=formatting_prompts_func
        )

        try:
            torch.xpu.reset_peak_memory_stats()
        except AttributeError:
            pass
        start_time = time.time()
        
        trainer.train()
        
        train_time = (time.time() - start_time) / 3600
        try:
            memory_gb = torch.xpu.max_memory_allocated() / (1024**3)
        except AttributeError:
            memory_gb = 0.0

        log_path = f"./loss_{suffix}.json"
        with open(log_path, 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"Historial de loss guardado en {log_path}")

        if args.method == "lora":
            trainer.model.save_pretrained(f"./resultados_{suffix}")
        elif args.method == "fft":
            trainer.save_model(f"./resultados_{suffix}")
        
        print(f"Entrenamiento finalizado y pesos guardados en ./resultados_{suffix}")

        resultados_train = {
            "method": args.method,
            "lora_r": args.lora_r if args.method == "lora" else None,
            "train_time_hours": round(train_time, 2),
            "memory_gb": round(memory_gb, 2)
        }
        result_path = f"./metricas_train_{suffix}.json"
        with open(result_path, 'w') as f:
            json.dump(resultados_train, f, indent=2)

    elif args.fase == "eval":
        if args.method == "base":
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

        acc, macro_f1, em = evaluate_model(model, tokenizer, datasets["test"])

        print("-" * 50)
        print(f"Resultados para el método: {args.method.upper()}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Macro F1:  {macro_f1:.4f}")
        print(f"EM:        {em:.4f}")
        print("-" * 50)

        resultados_eval = {
            "method": args.method,
            "lora_r": args.lora_r if args.method == "lora" else None,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "em": round(em, 4)
        }
        result_path = f"./resultados_{suffix}.json"
        with open(result_path, 'w') as f:
            json.dump(resultados_eval, f, indent=2)
        print(f"Resultados de evaluación guardados en {result_path}")

if __name__ == "__main__":
    main()
