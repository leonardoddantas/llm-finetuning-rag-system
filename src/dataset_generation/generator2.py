import json
import os
import pdfplumber
import torch

from tqdm import tqdm
from transformers import pipeline, logging

logging.set_verbosity_error()


def extract_text_from_file(file_path):
    """
    Extrai texto de um arquivo .pdf ou .txt.
    Retorna uma string com todo o conteúdo.
    """
    if file_path.lower().endswith('.pdf'):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    else:
        raise ValueError("Formato não suportado. Use .pdf ou .txt")


def split_text(text, max_chunk_length=1000):
    """
    Divide o texto em chunks.
    """
    paragraphs = text.split("\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:

        if len(current_chunk) + len(para) < max_chunk_length:
            current_chunk += para + "\n"

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = para + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def generate_instruction_response(chunk, hf_pipeline):

    prompt = f"""
Você é um gerador de datasets para fine-tuning de modelos de linguagem.

Com base exclusivamente no conteúdo fornecido:

1. Gere uma pergunta objetiva e informativa.
2. Gere uma resposta curta e precisa.
3. Utilize apenas informações presentes no texto.
4. Não utilize conhecimento externo.
5. Gere a pergunta e a resposta no mesmo idioma do conteúdo.
6. Priorize conceitos, definições, processos e explicações.
7. Não gere perguntas sobre:
   - título do livro
   - nome do autor
   - capítulos
   - sumário
   - ISBN
   - informações editoriais

Limites:
- Pergunta: máximo de 20 palavras.
- Resposta: máximo de 30 palavras.

Use exatamente o formato:

INSTRUCTION: <pergunta>
RESPONSE: <resposta>

Conteúdo:
\"\"\"
{chunk}
\"\"\"
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:

        outputs = hf_pipeline(
            messages,
            max_new_tokens=150,
            max_length=None,
            return_full_text=False
        )

        content = outputs[0]["generated_text"]

        instruction = (
            content
            .split("INSTRUCTION:")[1]
            .split("RESPONSE:")[0]
            .strip()
        )

        answer = (
            content
            .split("RESPONSE:")[1]
            .strip()
        )

        return instruction, answer

    except Exception:
        return None, None


def append_to_jsonl(instruction, answer, output_file):
    """
    Salva um único exemplo imediatamente.
    """

    example = {
        "Instruction": instruction,
        "Output": answer
    }

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                example,
                ensure_ascii=False
            ) + "\n"
        )


def generate_dataset(
    file_path,
    model_id="google/gemma-3-1b-it",
    output_file="/content/drive/MyDrive/llm-project/dataset_gerado.jsonl",
    max_chunks=None
):

    print(f"🔄 Carregando modelo: {model_id} ...")

    hf_pipeline = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print("📄 Extraindo texto do arquivo...")
    text = extract_text_from_file(file_path)

    print("✂️ Dividindo em chunks...")
    chunks = split_text(text)

    if max_chunks:
        chunks = chunks[:max_chunks]

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    # Limpa o arquivo antigo
    with open(output_file, "w", encoding="utf-8"):
        pass

    generated_count = 0

    print(
        f"🧠 Gerando pares para {len(chunks)} chunks..."
    )

    for chunk in tqdm(
        chunks,
        desc="Processando chunks"
    ):

        if len(chunk.strip()) < 10:
            continue

        instruction, answer = generate_instruction_response(
            chunk,
            hf_pipeline
        )

        if instruction and answer:

            append_to_jsonl(
                instruction,
                answer,
                output_file
            )

            generated_count += 1

            if generated_count % 10 == 0:
                print(
                    f"💾 {generated_count} exemplos já salvos"
                )

    print(
        f"\n✅ Dataset salvo em: {output_file}"
    )

    print(
        f"📊 Total de exemplos: {generated_count}"
    )