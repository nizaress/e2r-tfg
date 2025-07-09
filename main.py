from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import torch.optim as optim
from huggingface_hub import hf_hub_download
from transformers import (AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer)
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, model_name='t5-small'):
        super(Generator, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None):
        if labels is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            return outputs
        else:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=2048,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )
            return outputs

    def generate_simplified(self, original_text):
        prompt = f"simplify: {original_text}"
        inputs = self.tokenizer(
            prompt,
            max_length=2048,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=2048,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )
        simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return simplified


class TextSimplificationGAN:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Usando dispositivo: {self.device}")

        self.generator = Generator().to(device)
        self.gen_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.gen_optimizer = optim.AdamW(self.generator.parameters(), lr=2e-5)

    def to_device(self, tensor_dict):
        if isinstance(tensor_dict, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tensor_dict.items()}
        elif isinstance(tensor_dict, torch.Tensor):
            return tensor_dict.to(self.device)
        else:
            return tensor_dict

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        print(f"Modelo cargado desde: {path}")



PROMPT_LECTURA_FACIL = (
    "Eres un experto en accesibilidad y simplificación de textos. Tu tarea es transformar el siguiente texto para que sea comprensible por cualquier persona, incluyendo personas con discapacidad intelectual, dificultades lectoras o bajo nivel de alfabetización.\n"
    "Debes aplicar TODAS las pautas de Lectura Fácil de forma estricta y priorizar la claridad, sencillez y accesibilidad. No omitas ni ignores ninguna regla.\n"
    "No repitas el texto original ni añadas explicaciones. Responde únicamente con el texto simplificado.\n"
    "\nPAUTAS DE ORTOTIPOGRAFÍA:\n"
    "- No uses mayúsculas en palabras o frases completas, excepto en siglas.\n"
    "- Usa mayúscula inicial solo al inicio de párrafos, títulos, después de punto o en nombres propios.\n"
    "- Separa ideas diferentes con punto y aparte, no con coma.\n"
    "- Usa punto y aparte o conjunciones en vez de punto y seguido o coma para ideas relacionadas.\n"
    "- Usa dos puntos (:) para listas de más de tres elementos.\n"
    "- No uses punto y coma (;).\n"
    "- Evita paréntesis, corchetes y signos poco habituales (% & / ...).\n"
    "- No uses etcétera ni puntos suspensivos (...), reemplázalos por 'entre otros' o 'y muchos más'.\n"
    "- Evita comillas; si las usas, acompáñalas de explicación.\n"
    "\nPAUTAS DE VOCABULARIO:\n"
    "- Usa lenguaje sencillo y frecuente, adaptado al público objetivo.\n"
    "- Evita términos abstractos, técnicos o complejos.\n"
    "- Sustituye palabras homófonas/homógrafas por sinónimos.\n"
    "- Evita palabras largas o con sílabas complejas.\n"
    "- Evita adverbios terminados en -mente.\n"
    "- Evita superlativos, usa 'muy' + adjetivo.\n"
    "- Elimina palabras redundantes o innecesarias.\n"
    "- Evita palabras en otros idiomas salvo uso común (ej. wifi).\n"
    "- No uses abreviaturas ni siglas sin explicar la primera vez.\n"
    "- Evita frases nominales y lenguaje figurado (o explícalo).\n"
    "- Usa siempre la misma palabra para el mismo referente.\n"
    "- Evita palabras indeterminadas (cosa, algo).\n"
    "- Escribe los números con cifras; para números grandes, usa comparaciones cualitativas.\n"
    "- Separa los dígitos de teléfonos por bloques.\n"
    "- Evita números ordinales, usa cardinales.\n"
    "- Evita fracciones y porcentajes, usa descripciones equivalentes.\n"
    "- Escribe fechas completas (ej. 'el 1 de enero de 2023').\n"
    "- Usa el formato de 12 horas con texto (ej. 'las 3 de la tarde').\n"
    "- Evita números romanos, escríbelos como se leen.\n"
    "\nPAUTAS DE ORACIONES:\n"
    "- Usa frases sencillas, evita oraciones complejas.\n"
    "- Usa presente de indicativo siempre que sea posible.\n"
    "- Evita tiempos compuestos, condicionales y subjuntivos.\n"
    "- Usa voz activa, evita la pasiva y la pasiva refleja.\n"
    "- Usa imperativo solo en contextos claros, aclarando a quién se dirige.\n"
    "- Evita oraciones impersonales y con gerundio.\n"
    "- Evita verbos consecutivos salvo perífrasis con deber, querer, saber o poder.\n"
    "- Prefiere oraciones afirmativas, evita doble negación.\n"
    "- No uses elipsis, expresa todas las ideas claramente.\n"
    "- Evita explicaciones entre comas o aposiciones que corten el ritmo.\n"
    "- Limita las oraciones a dos ideas por frase como máximo.\n"
    "- Usa conectores simples, evita conectores complejos como 'por lo tanto' o 'sin embargo'.\n"
    "\n\nTexto: \n{texto}\nSimplificación: "
)

class InferenceInput(BaseModel):
    text: str
    model: str
    hf_token: str | None = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_response(input_text: str):
    return {"response": f"Prueba: {input_text}"}


def simplificar_texto_llama3_8b(texto_original: str, hf_token: str | None = None) -> str:
    try:
        client = InferenceClient(token=hf_token)
        client.provider = "nebius"
        max_tokens = 2 * len(texto_original.split())
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente experto que ayuda a simplificar textos aplicando correctamente las pautas de Lectura Fácil."
                },
                {
                    "role": "user",
                    "content": PROMPT_LECTURA_FACIL.format(texto=texto_original)
                }
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error en la respuesta del modelo Llama3: {str(e)}"


def cargar_modelo(repo_adaptador: str, hf_token: str | None = None):
    config = PeftConfig.from_pretrained(repo_adaptador, token=hf_token)
    modelo_base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, token=hf_token)
    modelo_peft = PeftModel.from_pretrained(modelo_base, repo_adaptador, token=hf_token)
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, token=hf_token)

    return modelo_peft.eval(), tokenizer


def simplificar_con_gan(texto: str, hf_token: str | None = None) -> str:
    local_ckpt = hf_hub_download(repo_id="Nizaress/e2r-gan", filename="e2r_gan.pth")
    print(f"Checkpoint descargado en: {local_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = TextSimplificationGAN(device=device)
    gan.load_model(local_ckpt)

    return gan.generator.generate_simplified(texto)


@app.post("/llama")
async def llama_endpoint(data: InferenceInput):
    texto = data.text
    token = data.hf_token
    resultado = simplificar_texto_llama3_8b(texto, hf_token=token)
    return {"response": resultado}


@app.post("/finetune")
async def finetune_endpoint(data: InferenceInput):
    texto_original = data.text
    token = data.hf_token

    try:
        model, tokenizer = cargar_modelo("Nizaress/e2r-finetuned", hf_token=token)

        inputs = tokenizer(PROMPT_LECTURA_FACIL.format(texto=texto_original), return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"response": generated[len(PROMPT_LECTURA_FACIL.format(texto=texto_original)):]}
    
    except Exception as e:
        return {"response": f"Error al generar con modelo PEFT: {str(e)}"}
    

@app.post("/gan")
async def gan_endpoint(data: InferenceInput):
    try:
        texto = data.text
        token = data.hf_token
        resultado = simplificar_con_gan(texto, token)
        return {"response": resultado}
    except Exception as e:
        return {"response": f"Error al simplificar con GAN: {str(e)}"}