# MarianMT
#from transformers import MarianMTModel, MarianTokenizer
# NLLB
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5EncoderModel
# langdetect
from langdetect import detect
from flask import Flask, request, jsonify, render_template
from diffusers import BitsAndBytesConfig, StableDiffusion3Pipeline, SD3Transformer2DModel
import torch
from io import BytesIO
import base64
import os
import requests
from torch.nn import DataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

app = Flask(__name__)

# IPFS API URL
IPFS_API_URL = "http://147.47.122.200:5998/api/v0/add"

# 모델 로드 및 최적화
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)
t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    text_encoder_3=t5_nf4,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda:0")  # 기본 GPU 설정
#print(pipe.components.keys())
#pipe.components["transformer"].to("cuda:1")
#pipe.components["vae"].to("cuda:2") 

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# 모델과 토크나이저 로드
model_name = "NHNDQ/nllb-finetuned-ko2en"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPU 1번 사용 설정
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 모델을 GPU 1번으로 이동
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def translate_korean_to_english(korean_text):
    # 입력 텍스트를 토큰화하고 GPU로 이동
    inputs = tokenizer(korean_text, return_tensors="pt", padding=True, truncation=True).to(device)
    # 모델에 입력 후 출력 생성
    outputs = model.generate(**inputs, max_length=512)
    # 토큰을 텍스트로 변환 (출력은 CPU에서 처리)
    translated_text = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    return translated_text


def translate_to_english(korean_text):
    inputs = tokenizer(korean_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors")
    detected_language = detect(prompt)
    print(f"Detected Language: {detected_language}")
    if detected_language == "ko":
        prompt = translate_korean_to_english(prompt)
    print(prompt)
    guidance_scale = float(data.get("guidance_scale", 7))
    num_inference_steps = int(data.get("num_inference_steps", 18))
    #print(torch.cuda.memory_summary(device="cuda:0"))
    # 이미지 생성
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # 이미지를 로컬 파일로 저장
    img_io = BytesIO()
    image.save(img_io, "PNG")
    img_io.seek(0)
    img_bytes = img_io.getvalue()

    # IPFS에 업로드
    try:
        response = requests.post(IPFS_API_URL, files={"file": img_bytes})
        response.raise_for_status()
        ipfs_hash = response.json().get("Hash")
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to upload to IPFS: {e}"}), 500

    # IPFS Gateway URL 생성
    ipfs_url = f"http://192.168.1.131:8080/ipfs/{ipfs_hash}"

    # 이미지를 base64로 인코딩
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    print(ipfs_hash)
    print(ipfs_url)
    # 결과 반환
    return jsonify({
        "image": img_base64,
        "ipfs_hash": ipfs_hash,
        "ipfs_url": ipfs_url
    })


@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
