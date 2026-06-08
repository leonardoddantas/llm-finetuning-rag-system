import json
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
    Divide o texto em chunks com base em quebras de linha.
    Cada chunk terá no máximo max_chunk_length caracteres.
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
    """
    Dado um chunk de texto, usa o pipeline de texto para gerar uma pergunta curta
    e uma resposta direta (ambas com menos de 10 palavras).
    """
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

        Content:
        \"\"\"
        {chunk}
        \"\"\"
        """

    messages = [{"role": "user", "content": prompt}]

    try:
        outputs = hf_pipeline(
            messages,
            max_new_tokens=150,
            max_length=None,  # evita warning
            return_full_text=False
        )
        content = outputs[0]["generated_text"]

        # Extrai os campos esperados
        instr_part = content.split("INSTRUCTION:")[1].split("RESPONSE:")[0].strip()
        answer = content.split("RESPONSE:")[1].strip()
        return instr_part, answer
    except Exception as e:
        # Em caso de erro, retorna None, None (o par será ignorado)
        return None, None

def save_to_jsonl(pairs, output_file="dataset.jsonl"):
    """
    pairs: lista de tuplas (instrução, resposta)
    output_file: nome do arquivo de saída
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for instruction, answer in pairs:
            if instruction and answer:
                example = {
                    "Instruction": instruction,
                    "Output": answer
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

def generate_dataset(file_path, model_id="Qwen/Qwen2.5-1.5B-Instruct",
                    output_file="dataset.jsonl", max_chunks=None):
    """
    file_path: caminho para o arquivo .pdf ou .txt
    model_id: identificador do modelo no Hugging Face Hub
    output_file: nome do arquivo de saída (JSONL)
    max_chunks: se especificado, limita a quantidade de chunks processados
    """
    print(f"🔄 Carregando modelo: {model_id} ...")

    # Pipeline para geração de texto
    hf_pipeline = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",        # usa GPU se disponível
        torch_dtype=torch.bfloat16 # reduz consumo de memória
    )

    print("📄 Extraindo texto do arquivo...")
    text = extract_text_from_file(file_path)

    print("✂️  Dividindo em chunks...")
    chunks = split_text(text)

    if max_chunks:
        chunks = chunks[:max_chunks]

    print(f"🧠 Gerando pares (instrução + resposta) para {len(chunks)} chunks...")
    pairs = []

    for chunk in tqdm(chunks, desc="Processando chunks"):
        if len(chunk.strip()) < 10: 
            continue
        instruction, answer = generate_instruction_response(chunk, hf_pipeline)
        if instruction and answer:
            pairs.append((instruction, answer))

    save_to_jsonl(pairs, output_file)
    print(f"\n✅ Dataset salvo em: {output_file} ({len(pairs)} exemplos gerados)")
