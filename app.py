"""
app.py — Sugarcane Disease Detection · MoE v4
Gradio 5 compatible · HuggingFace Spaces deployment
"""

import os, time, json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import gradio as gr
from PIL import Image
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CANONICAL   = ["Mosaic", "Rust", "RedRot", "YellowLeaf", "Healthy"]
NUM_CLASSES = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR  = Path("models")
E1_ZERO     = [0, 1]
E2_ZERO     = [2, 3]
HF_REPO_ID  = "vishnu0107/sugarcane_moe"

_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

CLASS_COLORS = ["#fbbf24", "#fb923c", "#f87171", "#facc15", "#34d399"]
DISEASE_INFO = {
    "Mosaic":     ("#fbbf24", "Mosaic Virus",        "SCMV",
                   "Mosaic / streaking on leaves, stunted growth.",
                   "Use resistant varieties; remove infected plants immediately."),
    "Rust":       ("#fb923c", "Rust Disease",         "Puccinia melanocephala",
                   "Orange-brown pustules on leaf undersides.",
                   "Apply fungicide spray; remove infected leaf debris."),
    "RedRot":     ("#f87171", "Red Rot",              "Colletotrichum falcatum",
                   "Red discolouration inside stalks, foul odour.",
                   "Destroy infected stools; use hot-water treated sets."),
    "YellowLeaf": ("#facc15", "Yellow Leaf Virus",    "ScYLV",
                   "Yellowing of midrib and leaf lamina.",
                   "Use certified disease-free planting material."),
    "Healthy":    ("#34d399", "Healthy Plant",        "",
                   "No disease signs detected.",
                   "Continue regular monitoring and preventive practices."),
}

# ══════════════════════════════════════════════════════════════
# AUTO-DOWNLOAD MODELS
# ══════════════════════════════════════════════════════════════
def maybe_download_models():
    if HF_REPO_ID is None:
        return
    try:
        from huggingface_hub import hf_hub_download
        MODELS_DIR.mkdir(exist_ok=True)
        required = ["expert1B_best.pt", "expert2B_best.pt", "expert3_best.pt",
                    "gate_best.pt", "temperature.pt", "ood_config.pt"]
        for f in required:
            dest = MODELS_DIR / f
            if not dest.exists():
                print(f"Downloading {f} from {HF_REPO_ID}...")
                hf_hub_download(repo_id=HF_REPO_ID, filename=f,
                                local_dir=str(MODELS_DIR))
                print(f"  OK {f} saved to {dest}")
    except Exception as e:
        print(f"Download error: {e}")

maybe_download_models()

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════
class SwinWithPool(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head     = nn.Linear(self.backbone.num_features, num_classes)
    @property
    def num_features(self): return self.backbone.num_features
    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() == 4:   feat = feat.mean([1, 2])
        elif feat.dim() == 3: feat = feat.mean(1)
        return self.head(feat)

class AttentionGate(nn.Module):
    D = 64
    def __init__(self, n_experts=3):
        super().__init__()
        D = self.D
        self.ctx_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=4, padding=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, D), nn.LayerNorm(D),
        )
        self.expert_proj = nn.ModuleList([nn.Linear(5, D) for _ in range(n_experts)])
        self.cross_attn  = nn.MultiheadAttention(D, num_heads=4, batch_first=True, dropout=0.1)
        self.out_head    = nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, n_experts))
        self.n = n_experts
    def forward(self, img, experts):
        q  = self.ctx_cnn(img).unsqueeze(1)
        kv = torch.stack([self.expert_proj[i](experts[i]) for i in range(self.n)], dim=1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        return F.softmax(self.out_head(attn_out.squeeze(1)), dim=-1), None

# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
def _load(path, builder=None):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    ht = ck.get("head_type", "simple")
    if ht == "swin_wrapped":
        m = SwinWithPool(ck["model_name"], ck["num_classes"], pretrained=False)
    else:
        m = timm.create_model(ck["model_name"], pretrained=False, num_classes=ck["num_classes"])
        if builder: m = builder(m, m.num_features)
    m.load_state_dict(ck["model_state_dict"])
    m = m.to(DEVICE).eval()
    for p in m.parameters(): p.requires_grad = False
    print(f"  OK {ck['model_name']:40s}  val_f1={ck['val_f1']:.4f}  [{ht}]")
    return m, ck.get("val_f1", 0.0)

def _e1b_head(model, nf):
    model.classifier = nn.Sequential(
        nn.Linear(nf, 512), nn.BatchNorm1d(512), nn.GELU(),
        nn.Dropout(0.4), nn.Linear(512, 128), nn.GELU(),
        nn.Dropout(0.2), nn.Linear(128, NUM_CLASSES),
    ); return model

def _e2b_head(model, nf):
    model.head.fc = nn.Sequential(
        nn.Linear(nf, 256), nn.GELU(),
        nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES),
    ); return model

def load_all_models():
    print(f"Loading models on {DEVICE}...")
    E1B, _ = _load(MODELS_DIR / "expert1B_best.pt", _e1b_head)
    E2B, _ = _load(MODELS_DIR / "expert2B_best.pt", _e2b_head)
    E3,  _ = _load(MODELS_DIR / "expert3_best.pt")
    gate   = AttentionGate(n_experts=3).to(DEVICE).eval()
    gate.load_state_dict(torch.load(MODELS_DIR / "gate_best.pt",
                                    map_location=DEVICE, weights_only=True))
    for p in gate.parameters(): p.requires_grad = False
    cal = torch.load(MODELS_DIR / "temperature.pt", map_location="cpu", weights_only=True)
    ood = torch.load(MODELS_DIR / "ood_config.pt",  map_location="cpu", weights_only=True)
    T   = float(cal["temperature"])
    thr = float(ood["ood_threshold"])
    print(f"  Temperature T={T:.3f}  OOD threshold={thr:.3f}")
    print(f"  All models loaded on {DEVICE.upper()}")
    return E1B, E2B, E3, gate, T, thr

E1B, E2B, E3, GATE, TEMPERATURE, OOD_THRESHOLD = load_all_models()

# ══════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════
def preprocess(pil_img, size=224):
    tf = transforms.Compose([transforms.Resize((size, size)),
                              transforms.ToTensor(), _NORM])
    return tf(pil_img.convert("RGB")).unsqueeze(0)

def _mn(s, zi):
    m = s.clone(); m[:, zi] = 0.0
    return m / m.sum(-1, keepdim=True).clamp(min=1e-8)

@torch.no_grad()
def _single_pass(img224):
    i320 = F.interpolate(img224, (320, 320), mode="bilinear", align_corners=False)
    i256 = F.interpolate(img224, (256, 256), mode="bilinear", align_corners=False)
    s1   = F.softmax(E1B(i320), -1)
    s2   = F.softmax(E2B(i256), -1)
    e3l  = E3(img224)
    s3   = F.softmax(e3l, -1)
    s1m  = _mn(s1, E1_ZERO); s2m = _mn(s2, E2_ZERO)
    gw, _ = GATE(img224, [s1m, s2m, s3])
    c1 = s1m.max(-1, keepdim=True)[0]
    c2 = s2m.max(-1, keepdim=True)[0]
    c3 = s3.max(-1,  keepdim=True)[0]
    ew1 = gw[:, 0:1]*c1; ew2 = gw[:, 1:2]*c2; ew3 = gw[:, 2:3]*c3
    ews = (ew1+ew2+ew3).clamp(min=1e-8)
    ens = (ew1/ews)*s1m + (ew2/ews)*s2m + (ew3/ews)*s3
    return ens, gw.squeeze(0), e3l

@torch.no_grad()
def run_inference(pil_img, use_tta=True):
    t0  = time.perf_counter()
    img = preprocess(pil_img).to(DEVICE)
    H = W = 224; cs = int(H * 0.875)

    if use_tta:
        crops = [
            img,
            torch.flip(img, [3]), torch.flip(img, [2]), torch.flip(img, [2,3]),
            F.interpolate(img[:,:,:cs,:cs],     (H,W), mode="bilinear", align_corners=False),
            F.interpolate(img[:,:,H-cs:,:cs],   (H,W), mode="bilinear", align_corners=False),
            F.interpolate(img[:,:,:cs,W-cs:],   (H,W), mode="bilinear", align_corners=False),
            F.interpolate(img[:,:,H-cs:,W-cs:], (H,W), mode="bilinear", align_corners=False),
        ]
        results  = [_single_pass(c) for c in crops]
        ens_avg  = torch.stack([r[0] for r in results]).mean(0).squeeze(0)
        e3l_ood  = results[0][2]
    else:
        ens_raw, _, e3l_ood = _single_pass(img)
        ens_avg = ens_raw.squeeze(0)

    cal    = F.softmax(torch.log(ens_avg.clamp(min=1e-8)) / TEMPERATURE, dim=-1)
    energy = float((-TEMPERATURE * torch.logsumexp(e3l_ood / TEMPERATURE, -1)).item())

    return {
        "probs":      cal.cpu().numpy(),
        "energy":     energy,
        "is_ood":     energy > OOD_THRESHOLD,
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
        "tta_crops":  8 if use_tta else 1,
    }

# ══════════════════════════════════════════════════════════════
# SEGMENTATION
# ══════════════════════════════════════════════════════════════
def extract_disease_hsv(pil_img):
    """
    Highlights non-green/disease areas using an HSV mask.
    The background is converted to dark grayscale to make the disease pop.
    """
    img_np = np.array(pil_img.convert("RGB"))
    hsv = mcolors.rgb_to_hsv(img_np / 255.0)
    
    H = hsv[:, :, 0] * 360
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    
    # Healthy green hue range
    green_mask = (H > 55) & (H < 170)
    # Ignore very dark or unsaturated background pixels
    bg_mask = (S < 0.2) | (V < 0.2)
    
    disease_mask = ~(green_mask | bg_mask)
    
    # Create a dark grayscale version of the image for the background
    gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
    gray_3d = np.stack((gray, gray, gray), axis=-1) * 0.4
    
    # Where disease is detected, overlay a vivid red/pink color
    final = gray_3d.copy()
    final[disease_mask] = [239, 68, 68] # Tailwind Red-500
    
    return Image.fromarray(final.astype(np.uint8))

# ══════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════
_PLACEHOLDER = (
    "<div style='display:flex;align-items:center;justify-content:center;"
    "height:260px;color:#475569;font-family:Georgia,serif;font-size:1em;"
    "border:2px dashed #334155;border-radius:14px'>"
    "Upload a leaf image and click Diagnose</div>"
)

def predict(image, use_tta):
    if image is None:
        return _PLACEHOLDER, None

    res = run_inference(image, use_tta=use_tta)

    # ── OOD: hard stop ────────────────────────────────────────
    if res["is_ood"]:
        return ("""
<div style="font-family:'Georgia',serif;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:18px;padding:36px 28px;
            background:linear-gradient(135deg,#450a0a 0%,#1e293b 100%);
            border:1.5px solid #ef4444;border-radius:16px;text-align:center">
  <div style="font-size:2.8em;line-height:1">&#9888;&#65039;</div>
  <div style="font-size:1.4em;font-weight:700;color:#fca5a5;letter-spacing:.01em">
    Not a sugarcane leaf image
  </div>
  <div style="color:#fecaca;font-size:0.95em;max-width:360px;line-height:1.65">
    This image does not resemble the sugarcane leaf photos the model was trained on.
    Please upload a <strong>clear, close-up photo of a sugarcane leaf</strong>
    for an accurate diagnosis.
  </div>
  <div style="margin-top:6px;padding:9px 24px;background:#ef444433;
              border:1px solid #ef4444;border-radius:8px;color:#fca5a5;
              font-size:0.82em;letter-spacing:.04em;font-family:monospace">
    Please upload a different image
  </div>
</div>""", None)

    # ── Normal result ─────────────────────────────────────────
    probs = res["probs"]
    top   = int(probs.argmax())
    conf  = float(probs[top])
    key   = CANONICAL[top]
    color, common, pathogen, symptoms, advice = DISEASE_INFO[key]
    tta_label = f"{res['tta_crops']}-crop TTA" if res["tta_crops"] > 1 else "Single pass"

    if key == "Healthy":
        sev_label, sev_col, sev_bg = "HEALTHY", "#34d399", "#064e3b"
    elif conf > 0.80:
        sev_label, sev_col, sev_bg = "HIGH CONFIDENCE", color, f"{color}33"
    elif conf > 0.55:
        sev_label, sev_col, sev_bg = "MODERATE", color, f"{color}22"
    else:
        sev_label, sev_col, sev_bg = "LOW CONFIDENCE", "#94a3b8", "#334155"

    pathogen_line = (
        f"<div style='color:{color};font-size:0.82em;font-style:italic;"
        f"margin-bottom:14px;letter-spacing:.02em'>{pathogen}</div>"
        if pathogen else ""
    )

    consult_line = (
        "<div style='margin-top:10px;padding-top:10px;border-top:1px dashed #334155;color:#94a3b8;font-size:0.85em;line-height:1.4'>"
        "<em>Note: Consult a local agriculturalist or extension officer for further advice.</em></div>"
    ) if key != "Healthy" else ""

    result_html = f"""
<div style="font-family:'Georgia',serif;">
  <div style="display:flex;align-items:center;justify-content:space-between;
              margin-bottom:14px;flex-wrap:wrap;gap:8px">
    <div style="display:flex;align-items:center;gap:10px">
      <div style="width:14px;height:14px;border-radius:50%;background:{color};
                  box-shadow:0 0 12px {color}aa;flex-shrink:0"></div>
      <div style="font-size:1.6em;font-weight:700;color:{color};
                  letter-spacing:-.01em;line-height:1.2">{common}</div>
    </div>
    <div style="padding:5px 14px;border-radius:20px;font-size:0.7em;font-weight:700;
                letter-spacing:.08em;font-family:monospace;background:{sev_bg};
                color:{sev_col};border:1px solid {sev_col}88">{sev_label}</div>
  </div>

  {pathogen_line}

  <div style="margin-bottom:18px">
    <div style="display:flex;justify-content:space-between;color:#94a3b8;
                font-size:0.78em;margin-bottom:6px;font-family:monospace;letter-spacing:.04em">
      <span>CONFIDENCE</span>
      <span style="color:{color};font-weight:700">{conf*100:.1f}%</span>
    </div>
    <div style="background:#0f172a;border-radius:6px;height:8px;overflow:hidden;border:1px solid #334155">
      <div style="background:linear-gradient(90deg,{color}88,{color});height:100%;
                  width:{conf*100:.1f}%;border-radius:6px"></div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
    <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px">
      <div style="color:#64748b;font-size:0.7em;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:6px;font-family:monospace">Symptoms</div>
      <div style="color:#f8fafc;font-size:0.9em;line-height:1.55">{symptoms}</div>
    </div>
    <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px">
      <div style="color:#64748b;font-size:0.7em;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:6px;font-family:monospace">Actions Required</div>
      <div style="color:#f8fafc;font-size:0.9em;line-height:1.55">{advice}</div>
      {consult_line}
    </div>
  </div>

  <div style="display:flex;gap:14px;flex-wrap:wrap;padding:10px 14px;background:#0f172a;
              border-radius:8px;color:#94a3b8;font-size:0.75em;font-family:monospace;
              letter-spacing:.04em;border:1px solid #334155">
    <span>&#9201; {res['elapsed_ms']:.0f} ms</span>
    <span>&#183;</span><span>{tta_label}</span>
    <span>&#183;</span><span>{DEVICE.upper()}</span>
  </div>
</div>"""

    return result_html, extract_disease_hsv(image)


# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root, .dark {
    --background-fill-primary: #0f172a !important; /* slate-900 */
    --background-fill-secondary: #1e293b !important; /* slate-800 */
    --border-color-primary: #334155 !important; /* slate-700 */
    --body-text-color: #f8fafc !important; /* slate-50 */
    --block-background-fill: #1e293b !important;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .gradio-container, .main {
    background: #0f172a !important;
    color: #f8fafc !important;
    font-family: 'DM Sans', sans-serif !important;
}
.main .block-container {
    max-width: 1100px !important;
    padding: 0 16px !important;
}

.gr-form, .gr-box, .gr-panel, .gr-accordion, .gr-checkbox {
    background-color: #1e293b !important;
    border-color: #334155 !important;
}

[data-testid="image"] {
    border: 2px dashed #334155 !important;
    border-radius: 14px !important;
    background: #0f172a !important;
    transition: border-color .25s !important;
}
[data-testid="image"]:hover { border-color: #10b981 !important; }

button.lg.primary {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: .06em !important;
    font-size: 0.95em !important;
    color: #ffffff !important;
    box-shadow: 0 4px 18px #10b98133 !important;
    transition: all .2s !important;
    padding: 12px 24px !important;
}
button.lg.primary:hover {
    background: linear-gradient(135deg, #047857, #34d399) !important;
    box-shadow: 0 6px 26px #10b98155 !important;
    transform: translateY(-1px) !important;
}

label.svelte-1f354aw, .gr-checkbox label {
    color: #94a3b8 !important;
    font-size: 0.85em !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: .03em !important;
    background: transparent !important;
}

details {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
details summary {
    color: #cbd5e1 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85em !important;
    letter-spacing: .05em !important;
    padding: 12px 16px !important;
}

.gr-plot { background: transparent !important; border: none !important; }
footer { display: none !important; }
"""

# ══════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════
ARCH_MD = """
| Expert | Backbone | Resolution | Speciality |
|--------|----------|------------|------------|
| E1B | EfficientNetV2-M | 320 px | RedRot, YellowLeaf |
| E2B | ConvNeXt-Base | 256 px | Mosaic, Rust |
| E3 | Swin-Base | 224 px | All 5 classes |
| Gate | AttentionGate | 224 px | Routing |

**Training:** FocalLoss · LLRD · CutMix / MixUp &nbsp;&nbsp;**Test Macro F1 (TTA):** `0.9836`
"""

_LABEL_STYLE = ("display:block;font-family:'DM Mono',monospace;font-size:0.75em;"
                "color:#94a3b8;letter-spacing:.1em;text-transform:uppercase;"
                "margin-bottom:8px;margin-top:4px")

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Sugarcane Disease Detection") as demo:

    gr.HTML(f"""
<div style="padding:32px 8px 24px;text-align:center;
            border-bottom:1px solid #334155;margin-bottom:28px">
  <div style="font-family:'DM Serif Display',Georgia,serif;font-size:2.4em;
              color:#f8fafc;letter-spacing:-.02em;margin-bottom:8px">
    &#127807; Sugarcane Disease Detection
  </div>
  <div style="font-family:'DM Sans',sans-serif;font-size:0.95em;color:#94a3b8">
    Upload a close-up photo of a sugarcane leaf to identify disease
  </div>
</div>
""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=280):
            gr.HTML(f"<span style='{_LABEL_STYLE}'>Leaf Image</span>")
            inp = gr.Image(type="pil", height=280, show_label=False)
            tta_toggle = gr.Checkbox(
                label="8-crop TTA  (more accurate, ~8x slower)",
                value=True,
            )
            btn = gr.Button("🔍  Diagnose", variant="primary", size="lg")
            with gr.Accordion("Model Architecture", open=False):
                gr.Markdown(ARCH_MD)

        with gr.Column(scale=6, min_width=340):
            gr.HTML(f"<span style='{_LABEL_STYLE}'>Diagnosis</span>")
            out_result = gr.HTML(value=_PLACEHOLDER)

    gr.HTML("<div style='height:1px;background:#334155;margin:28px 0'></div>")
    gr.HTML(f"<span style='{_LABEL_STYLE}'>Disease Localization (HSV Mask)</span>")
    out_seg = gr.Image(type="pil", height=400, show_label=False, interactive=False)

    gr.HTML("""
<div style="margin-top:32px;padding:16px 8px;border-top:1px solid #334155;
            display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px">
  <span style="font-family:'DM Mono',monospace;font-size:0.75em;
               color:#64748b;letter-spacing:.04em">
    Built with PyTorch &#183; timm &#183; Gradio
  </span>
  <span style="font-family:'DM Mono',monospace;font-size:0.75em;
               color:#64748b;letter-spacing:.04em">
    Test Macro F1 (TTA): 0.9836
  </span>
</div>
""")

    _outs = [out_result, out_seg]
    btn.click(fn=predict, inputs=[inp, tta_toggle], outputs=_outs)
    inp.change(fn=predict, inputs=[inp, tta_toggle], outputs=_outs)

if __name__ == "__main__":
    demo.launch(share=False, ssr_mode=False)
