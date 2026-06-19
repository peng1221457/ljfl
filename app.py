# app.py
import io
import os
import json
from typing import List, Tuple
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# Config: adjust if needed
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/checkpoint_best.pth")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "outputs/class_names.json")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 5

app = FastAPI(title="Garbage Classifier API")

# Allow CORS for testing; lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility: build the same model architecture used in training
def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name == "mobilenet_v2":
        if pretrained:
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            m = models.mobilenet_v2(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        if pretrained:
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unsupported model: choose mobilenet_v2 or resnet50")
    return m

# Load class names
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"class names not found: {CLASS_NAMES_PATH}")
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

# Build and load model weights
MODEL_NAME = os.environ.get("MODEL_NAME", "mobilenet_v2")
model = build_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
ck = torch.load(MODEL_PATH, map_location="cpu")
state = ck.get('model_state') if isinstance(ck, dict) and 'model_state' in ck else ck
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# Preprocess transform (same normalization used during training)
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

def predict_image(img: Image.Image, topk: int = TOPK) -> List[Tuple[int, float]]:
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().squeeze(0)
        topk_probs, topk_idxs = torch.topk(probs, min(topk, probs.numel()))
        results = [(int(idx.item()), float(prob.item())) for idx, prob in zip(topk_idxs, topk_probs)]
    return results

# 首页：全新可视化页面（图片预览 + 概率柱状图）
@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>垃圾分类识别系统</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: "Microsoft YaHei", sans-serif;
        }
        body {
            background-color: #f5f7fa;
            padding: 30px;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            background: #fff;
            border-radius: 14px;
            padding: 30px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }
        h1 {
            text-align: center;
            color: #2d3748;
            margin-bottom: 30px;
        }
        .row {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        .col-left, .col-right {
            flex: 1;
            min-width: 420px;
        }
        .card {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .card h3 {
            color: #4a5568;
            margin-bottom: 16px;
            font-size: 18px;
        }
        #imgPreview {
            max-width: 100%;
            max-height: 360px;
            border: 1px dashed #cbd5e0;
            border-radius: 8px;
            display: block;
            margin: 10px auto;
        }
        input[type="file"] {
            margin-bottom: 15px;
        }
        button {
            background-color: #3182ce;
            color: white;
            border: none;
            padding: 10px 26px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #2b6cb0;
        }
        #resultTop1 {
            background-color: #bee3f8;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        #chartBox {
            width: 100%;
            height: 320px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px;
        }
        .bar-row {
            display: flex;
            align-items: center;
            margin: 10px 0;
            gap:10px;
        }
        .bar-label {
            width: 160px;
            font-size:14px;
        }
        .bar-wrap {
            flex:1;
            height:24px;
            background:#e8f4fc;
            border-radius:12px;
            overflow:hidden;
        }
        .bar-fill {
            height:100%;
            background:#3182ce;
            border-radius:12px;
            transition: width 0.4s ease;
        }
        .bar-score {
            width:70px;
            text-align:right;
            font-size:14px;
            color:#2d3748;
        }
        .hidden {
            display:none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>垃圾分类智能识别系统</h1>
        <div class="row">
            <!-- 左侧：上传图片区域 -->
            <div class="col-left">
                <div class="card">
                    <h3>1. 上传待识别图片</h3>
                    <form id="uploadForm">
                        <input type="file" id="fileInput" accept="image/*" required>
                        <br>
                        <button type="submit">开始识别</button>
                    </form>
                    <div id="previewWrap" class="hidden">
                        <h4 style="margin-top:16px;">图片预览</h4>
                        <img id="imgPreview" />
                    </div>
                </div>
            </div>
            <!-- 右侧：识别结果 + 概率柱状图 -->
            <div class="col-right">
                <div class="card">
                    <h3>2. 识别结果</h3>
                    <div id="resultWrap" class="hidden">
                        <div id="resultTop1"></div>
                        <h4>Top5 类别概率分布</h4>
                        <div id="chartBox"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById("uploadForm");
        const fileInput = document.getElementById("fileInput");
        const previewWrap = document.getElementById("previewWrap");
        const imgPreview = document.getElementById("imgPreview");
        const resultWrap = document.getElementById("resultWrap");
        const resultTop1 = document.getElementById("resultTop1");
        const chartBox = document.getElementById("chartBox");

        // 本地图片预览
        fileInput.addEventListener("change", function(e){
            const file = e.target.files[0];
            if(!file) return;
            const reader = new FileReader();
            reader.onload = function(ev){
                imgPreview.src = ev.target.result;
                previewWrap.classList.remove("hidden");
            }
            reader.readAsDataURL(file);
        })

        // 表单提交识别
        form.addEventListener("submit", async function(e){
            e.preventDefault();
            const file = fileInput.files[0];
            if(!file) return;
            const formData = new FormData();
            formData.append("file", file);

            try{
                const res = await fetch("/predict", {
                    method: "POST",
                    body: formData
                })
                const data = await res.json();
                renderResult(data);
            }catch(err){
                alert("识别失败：" + err)
            }
        })

        // 渲染结果与概率柱状图
        function renderResult(data){
            resultWrap.classList.remove("hidden");
            const top1 = data.top1;
            // 第一名展示
            resultTop1.innerHTML = `
                <h4>最佳匹配类别</h4>
                <p style="font-size:20px;margin:8px 0;color:#2b6cb0">${top1.label_name}</p>
                <p>置信概率：${(top1.score * 100).toFixed(2)} %</p>
            `
            // 清空柱状图容器
            chartBox.innerHTML = "";
            // 绘制每条概率条
            data.topk.forEach(item => {
                const percent = (item.score * 100).toFixed(2);
                const divRow = document.createElement("div");
                divRow.className = "bar-row";
                divRow.innerHTML = `
                    <div class="bar-label">${item.label_name}</div>
                    <div class="bar-wrap">
                        <div class="bar-fill" style="width:${percent}%"></div>
                    </div>
                    <div class="bar-score">${percent}%</div>
                `
                chartBox.appendChild(divRow);
            })
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot read image: {e}")
    preds = predict_image(img, topk=TOPK)
    # Build response
    resp = []
    for idx, prob in preds:
        name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else str(idx)
        resp.append({"label_idx": idx, "label_name": name, "score": round(prob, 6)})
    # Also return top-1 in a friendly way
    top1 = resp[0] if resp else None
    return JSONResponse({"top1": top1, "topk": resp})

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME, "num_classes": NUM_CLASSES}