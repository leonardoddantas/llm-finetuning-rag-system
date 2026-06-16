# LLM Fine-Tuning RAG System

Sistema desenvolvido para a disciplina **Tópicos Avançados em Inteligência Artificial**, com foco na geração de datasets utilizando RAG, fine-tuning de Large Language Models (LLMs) com LoRA, avaliação de desempenho e disponibilização dos modelos através de uma API RESTful utilizando FastAPI.

## Objetivo

O projeto tem como objetivo especializar modelos de linguagem no domínio de **Neurociência Cognitiva**, utilizando um documento científico como base de conhecimento para:

* Gerar automaticamente um dataset de instruções e respostas;
* Realizar fine-tuning com a técnica LoRA;
* Avaliar quantitativa e qualitativamente os modelos treinados;
* Disponibilizar os modelos através de uma API RESTful.

---

## Pipeline do Projeto

O desenvolvimento foi dividido em quatro etapas principais:

### Etapa 1 — Geração do Dataset

Notebook:

```bash
notebooks/01_rag_generation.ipynb
```

Nesta etapa foi realizado:

* Extração de texto do PDF;
* Divisão do conteúdo em chunks;
* Geração automática de pares instrução-resposta utilizando o modelo Phi-4 Mini Instruct;
* Curadoria manual dos exemplos gerados.

Resultado:

* 277 pares gerados;
* 253 pares válidos após curadoria.

---

### Etapa 2 — Fine-Tuning com LoRA

Foram treinados quatro modelos:

#### Modelos Causais

* Phi-4 Mini Instruct
* Llama 3.2 3B Instruct

#### Modelos Seq2Seq

* Flan-T5-XL
* LaMini-Flan-T5-783M

Os treinamentos foram realizados utilizando a técnica LoRA (Low-Rank Adaptation), reduzindo significativamente o número de parâmetros treináveis.

---

### Etapa 3 — Avaliação dos Modelos

Foram calculadas as seguintes métricas:

* Perplexidade (PPL)
* BLEU
* ROUGE-1
* ROUGE-2
* ROUGE-L
* Faithfulness
* Answer Relevance
* Plan Adherence

Além da análise quantitativa, foi realizada uma comparação qualitativa entre os modelos base e fine-tunados.

---

### Etapa 4 — Integração via API RESTful

Implementação de uma API utilizando FastAPI para disponibilizar os modelos treinados.

Endpoints principais:

```http
GET /health
```

Verifica o status da API.

```http
GET /modelos
```

Lista os modelos disponíveis.

```http
POST /chat
```

Recebe uma pergunta e retorna a resposta gerada pelo modelo selecionado.

---

## Estrutura do Projeto

```text
llm-finetuning-rag-system/
│
├── static/
│   └── index.html
│
├── src/
│   └── dataset_generation/
│       └── generator.py
│
├── data/
│   ├── raw/
│   │   └── 20260045290.pdf
│   │
│   └── processed/
│       ├── dataset_gerado.jsonl
│       └── dataset_curado.jsonl
│
├── lora_models/
│   ├── causal_model_1/
│   ├── causal_model_2/
│   ├── seq2seq_model_1/
│   └── seq2seq_model_2/
│
├── notebooks/
│   ├── 01_rag_generation.ipynb
│   ├── 02_lora_finetuning_causal_phi4.ipynb
│   ├── 02_lora_finetuning_causal_llama32.ipynb
│   ├── 02_lora_finetuning_seq2seq_flant5xl.ipynb
│   ├── 02_lora_finetuning_seq2seq_LaMiniFlanT5.ipynb
│   ├── 03_avaliacao_phi4_mini.ipynb
│   ├── 03_avaliacao_llama32.ipynb
│   ├── 03_avaliacao_flant5xl.ipynb
│   └── 03_avaliacao_lamini_flant5.ipynb
│
├── .venv/
├── .gitignore
└── main.py
```

---

## Principais Resultados

| Modelo         | PPL | BLEU | ROUGE-L |
| -------------- | --- | ---- | ------- |
| Phi-4 Mini     | 8.9 | 1.3  | 0.088   |
| Llama 3.2 3B   | 5.8 | 1.3  | 0.103   |
| Flan-T5-XL     | 5.7 | 5.5  | 0.206   |
| LaMini-Flan-T5 | 9.7 | 4.9  | 0.198   |

O modelo **Flan-T5-XL** apresentou o melhor desempenho geral, alcançando os maiores valores de BLEU e ROUGE, demonstrando maior capacidade de especialização para o domínio estudado.

---

## Tecnologias Utilizadas

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* FastAPI
* LangChain
* FAISS
* Pandas
* Matplotlib
* Scikit-learn

---

## Autor

**Leonardo**

Curso de Engenharia da Computação

Universidade Federal do Rio Grande do Norte (UFRN)

Disciplina: Tópicos Avançados em Inteligência Artificial
