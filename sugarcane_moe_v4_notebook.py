# ──────────────────────────────────────────────────────────────
# CELL 1 — Install deps
# ──────────────────────────────────────────────────────────────
import subprocess, sys, os

subprocess.run([sys.executable, '-m', 'pip', 'install',
                'timm', 'onnx', 'onnxruntime', 'scikit-learn',
                'albumentations', 'huggingface_hub', '-q'], check=False)

# ──────────────────────────────────────────────────────────────
# CELL 2 — CONFIG
# ──────────────────────────────────────────────────────────────
import math

DATA_DIR  = '/kaggle/input/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset'
SPLIT_DIR = '/kaggle/input/datasets/chintavishnuv/sc-expert1-out'
OUT       = '/kaggle/working'

FORCE_E1B  = False
FORCE_E2B  = False
FORCE_E3   = False
FORCE_GATE = False
FORCE_CAL  = False

E1B_EPOCHS  = 80
E2B_EPOCHS  = 80
E3_EPOCHS   = 100
GATE_EPOCHS = 80
PATIENCE    = 15
ACCUM_STEPS = 4

DEVICE = 'cuda'

os.makedirs(OUT, exist_ok=True)
for label, path in [('DATA_DIR', DATA_DIR), ('SPLIT_DIR', SPLIT_DIR)]:
    print(f'{label}: {path}  exists={os.path.isdir(path)}')
for f in ['train_idx.npy', 'val_idx.npy', 'test_idx.npy']:
    p = os.path.join(SPLIT_DIR, f)
    print(f'  {f}: {chr(9989) if os.path.exists(p) else chr(10060)+" MISSING"}')

# ──────────────────────────────────────────────────────────────
# CELL 3 — SHARED UTILITIES  (FIX: SwinWithPool + load_expert)
# ──────────────────────────────────────────────────────────────
import random, warnings, types
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import timm
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings('ignore')

CANONICAL   = ['Mosaic', 'Rust', 'RedRot', 'YellowLeaf', 'Healthy']
NUM_CLASSES = 5
_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

def get_data_root(data_dir):
    for sub in ['train', 'Train', 'images', 'Images', '']:
        p = os.path.join(data_dir, sub) if sub else data_dir
        if os.path.isdir(p) and sum(os.path.isdir(os.path.join(p, d))
                                    for d in os.listdir(p)) >= 4:
            return p
    return data_dir

def build_dataset(root, transform):
    ds = datasets.ImageFolder(root, transform=transform)
    folder_to_canon = {}
    for name in ds.classes:
        nl = name.lower()
        for c in CANONICAL:
            if c.lower() in nl or nl in c.lower():
                folder_to_canon[name] = c; break
        else:
            folder_to_canon[name] = name
    old_to_new = {
        old_idx: CANONICAL.index(folder_to_canon.get(name, name))
        if folder_to_canon.get(name, name) in CANONICAL else old_idx
        for old_idx, name in enumerate(ds.classes)
    }
    ds.targets = [old_to_new.get(t, t) for t in ds.targets]
    ds.samples = [(s, old_to_new.get(l, l)) for s, l in ds.samples]
    return ds

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.ls = label_smoothing
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.ls, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

def cutmix(x, y, alpha=1.0):
    if alpha <= 0 or x.size(0) < 2: return x, y, y, 1.0
    lam  = float(np.random.beta(alpha, alpha))
    idx  = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cr   = np.sqrt(1.0 - lam)
    ch, cw = int(H * cr), int(W * cr)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    out = x.clone(); out[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(H * W)
    return out, y, y[idx], lam

def mixup(x, y, alpha=0.4):
    if alpha <= 0 or x.size(0) < 2: return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cosine_warmup_lr(optimizer, epoch, warmup, total, base_lrs, min_lr=1e-7):
    if epoch < warmup:
        scale = (epoch + 1) / max(1, warmup)
    else:
        t = (epoch - warmup) / max(1, total - warmup)
        scale = min_lr / max(base_lrs) + 0.5 * (1 - min_lr / max(base_lrs)) * (1 + math.cos(math.pi * t))
    for pg, blr in zip(optimizer.param_groups, base_lrs):
        pg['lr'] = blr * scale

class SwinWithPool(nn.Module):
    """Swin backbone (num_classes=0 → guaranteed (B,C) output) + head."""
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        nf = self.backbone.num_features
        self.head = nn.Linear(nf, num_classes)

    @property
    def num_features(self):
        return self.backbone.num_features

    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() == 4:
            feat = feat.mean([1, 2])
        elif feat.dim() == 3:
            feat = feat.mean(1)
        return self.head(feat)

def load_expert(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    ht = ck.get('head_type', 'simple')
    model_name = ck['model_name']

    if ht == 'swin_wrapped':
        m = SwinWithPool(model_name, ck['num_classes'], pretrained=False)
        m.load_state_dict(ck['model_state_dict'])
    else:
        m = timm.create_model(model_name, pretrained=False, num_classes=ck['num_classes'])
        nf = m.num_features
        if ht == 'efficientnet_deep':
            m.classifier = nn.Sequential(
                nn.Linear(nf, 512), nn.BatchNorm1d(512), nn.GELU(),
                nn.Dropout(0.4), nn.Linear(512, 128), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(128, ck['num_classes'])
            )
        elif ht == 'convnext_deep':
            m.head.fc = nn.Sequential(
                nn.Linear(nf, 256), nn.GELU(),
                nn.Dropout(0.3), nn.Linear(256, ck['num_classes'])
            )
        m.load_state_dict(ck['model_state_dict'])

    m = m.to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    print(f'  Loaded {model_name:40s}  val_f1={ck["val_f1"]:.4f}')
    return m

try:
    _et = transforms.ElasticTransform(alpha=1.0, sigma=1.0)
    HAS_ELASTIC = True; print('✅ ElasticTransform available')
except AttributeError:
    HAS_ELASTIC = False; print('⚠️  ElasticTransform unavailable')

try:
    import albumentations as A
    import albumentations.pytorch as Ap
    HAS_ALBUMENTATION = True; print('✅ albumentations available')
except ImportError:
    HAS_ALBUMENTATION = False; print('⚠️  albumentations not found')

set_seed(42)
print(f'\n✅ Utilities ready  |  DEVICE={DEVICE}  |  CANONICAL={CANONICAL}')

# ──────────────────────────────────────────────────────────────
# CELL 4 — STAGE 1: Expert 1B (EfficientNetV2-M 320px)
# ──────────────────────────────────────────────────────────────
E1B_PATH = os.path.join(OUT, 'expert1B_best.pt')

if os.path.exists(E1B_PATH) and not FORCE_E1B:
    print('⏭  E1B exists — skipping.')
else:
    print('=' * 62)
    print('  STAGE 1: Expert 1B — EfficientNetV2-M 320px')
    print('=' * 62)
    set_seed(42)

    E1B_MODEL  = 'tf_efficientnetv2_m'
    E1B_SPEC   = [2, 3, 4]
    E1B_IMG    = 320
    E1B_LR     = 2e-4
    E1B_WD     = 1e-2
    E1B_WARMUP = 5

    WMAP1 = {'RedRot': 6.0, 'YellowLeaf': 7.0, 'Healthy': 1.5,
             'Mosaic': 0.3, 'Rust': 0.3}

    _aug1 = [
        transforms.Resize((E1B_IMG + 48, E1B_IMG + 48)),
        transforms.RandomResizedCrop(E1B_IMG, scale=(0.60, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
        transforms.RandomRotation(30),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5))], p=0.35),
    ]
    if HAS_ELASTIC:
        _aug1.append(transforms.RandomApply(
            [transforms.ElasticTransform(alpha=60.0, sigma=6.0)], p=0.4))
    _aug1 += [transforms.ToTensor(), _NORM,
              transforms.RandomErasing(p=0.3, scale=(0.02, 0.20))]
    aug1    = transforms.Compose(_aug1)
    val_tf1 = transforms.Compose([transforms.Resize((E1B_IMG, E1B_IMG)),
                                   transforms.ToTensor(), _NORM])

    dr1     = get_data_root(DATA_DIR)
    full_v1 = build_dataset(dr1, val_tf1)
    full_t1 = build_dataset(dr1, aug1)
    lbl1    = np.array(full_v1.targets)
    tr_idx  = np.load(os.path.join(SPLIT_DIR, 'train_idx.npy'))
    vl_idx  = np.load(os.path.join(SPLIT_DIR, 'val_idx.npy'))

    spec_mask1 = np.isin(lbl1[tr_idx], E1B_SPEC)
    tr_spec1   = tr_idx[spec_mask1]
    print(f'  Specialty train samples: {len(tr_spec1)}')
    for c in E1B_SPEC:
        print(f'    {CANONICAL[c]:12s}: {(lbl1[tr_spec1]==c).sum()}')

    tr_ld1 = DataLoader(Subset(full_t1, tr_spec1), batch_size=12,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    vl_ld1 = DataLoader(Subset(full_v1, vl_idx), batch_size=32,
                        shuffle=False, num_workers=2)

    model1 = timm.create_model(E1B_MODEL, pretrained=True, num_classes=NUM_CLASSES)
    nf1    = model1.num_features
    model1.classifier = nn.Sequential(
        nn.Linear(nf1, 512), nn.BatchNorm1d(512), nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128), nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, NUM_CLASSES)
    )
    model1 = model1.to(DEVICE)
    print(f'  E1B num_features={nf1}')
    print(f'  Head: {nf1}→512→128→{NUM_CLASSES} (BN+GELU+Dropout)')

    cw1   = torch.tensor([WMAP1.get(c, 1.0) for c in CANONICAL],
                          dtype=torch.float, device=DEVICE)
    crit1 = FocalLoss(gamma=2.0, weight=cw1, label_smoothing=0.15)
    opt1  = torch.optim.AdamW(model1.parameters(), lr=E1B_LR, weight_decay=E1B_WD)
    base_lrs1 = [E1B_LR] * len(opt1.param_groups)
    scaler1   = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))

    def spec_f1(model, loader, spec_idx):
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().tolist())
                gts.extend(lbls.tolist())
        pairs = [(p, l) for p, l in zip(preds, gts) if l in spec_idx]
        if not pairs: return 0.0
        pp, ll = zip(*pairs)
        return f1_score(list(ll), list(pp), average='macro', zero_division=0)

    best1 = 0.0; pat1 = 0
    print(f'\nTraining E1B ({E1B_EPOCHS} epochs, accum={ACCUM_STEPS}x, warmup={E1B_WARMUP} ep)...')
    for epoch in range(1, E1B_EPOCHS + 1):
        model1.train(); tl = tot = 0
        cosine_warmup_lr(opt1, epoch - 1, E1B_WARMUP, E1B_EPOCHS, base_lrs1)
        opt1.zero_grad()
        for step, (imgs, lbls) in enumerate(tr_ld1):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            xm, ya, yb, lam = cutmix(imgs, lbls, alpha=1.5)
            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                out  = model1(xm)
                loss = (lam * crit1(out, ya) + (1 - lam) * crit1(out, yb)) / ACCUM_STEPS
            scaler1.scale(loss).backward()
            tl += loss.item() * ACCUM_STEPS * imgs.size(0); tot += imgs.size(0)
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(tr_ld1):
                scaler1.unscale_(opt1)
                torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)
                scaler1.step(opt1); scaler1.update(); opt1.zero_grad()
        vf = spec_f1(model1, vl_ld1, E1B_SPEC)
        lr_now = opt1.param_groups[0]['lr']
        if vf > best1:
            best1 = vf
            torch.save({
                'model_state_dict': model1.state_dict(),
                'model_name':       E1B_MODEL,
                'num_classes':      NUM_CLASSES,
                'class_names':      CANONICAL,
                'val_f1':           best1,
                'epoch':            epoch,
                'head_type':        'efficientnet_deep',
                'specialty':        E1B_SPEC,
                'img_size':         E1B_IMG,
            }, E1B_PATH)
            pat1 = 0; mark = ' ✓'
        else:
            pat1 += 1; mark = ''
        if epoch % 5 == 0 or mark:
            print(f'  Ep{epoch:3d}/{E1B_EPOCHS}  loss:{tl/tot:.4f}'
                  f'  spec_f1:{vf:.4f}  lr:{lr_now:.2e}{mark}')
        if pat1 >= PATIENCE:
            print(f'  Early stop ep{epoch}'); break

    print(f'\n✅ E1B done — best spec F1: {best1:.4f}  →  {E1B_PATH}')

# ──────────────────────────────────────────────────────────────
# CELL 5 — STAGE 2: Expert 2B (ConvNeXt-Base 256px)
# ──────────────────────────────────────────────────────────────
E2B_PATH = os.path.join(OUT, 'expert2B_best.pt')

if os.path.exists(E2B_PATH) and not FORCE_E2B:
    print('⏭  E2B exists — skipping.')
else:
    print('=' * 62)
    print('  STAGE 2: Expert 2B — ConvNeXt-Base 256px')
    print('=' * 62)
    set_seed(42)

    E2B_MODEL  = 'convnext_base'
    E2B_SPEC   = [0, 1, 4]
    E2B_IMG    = 256
    E2B_LR     = 2e-4
    E2B_WD     = 1e-2
    E2B_WARMUP = 5

    WMAP2 = {'Mosaic': 5.0, 'Rust': 5.0, 'Healthy': 1.0,
             'RedRot': 0.3, 'YellowLeaf': 0.3}

    aug2 = transforms.Compose([
        transforms.Resize((E2B_IMG + 32, E2B_IMG + 32)),
        transforms.RandomResizedCrop(E2B_IMG, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08),
        transforms.RandomRotation(30),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(), _NORM,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    val_tf2 = transforms.Compose([transforms.Resize((E2B_IMG, E2B_IMG)),
                                   transforms.ToTensor(), _NORM])

    dr2       = get_data_root(DATA_DIR)
    full_v2   = build_dataset(dr2, val_tf2)
    full_t2   = build_dataset(dr2, aug2)
    lbl2      = np.array(full_v2.targets)
    tr_idx2   = np.load(os.path.join(SPLIT_DIR, 'train_idx.npy'))
    vl_idx2   = np.load(os.path.join(SPLIT_DIR, 'val_idx.npy'))

    spec_mask2 = np.isin(lbl2[tr_idx2], E2B_SPEC)
    tr_spec2   = tr_idx2[spec_mask2]
    print(f'  Specialty train samples: {len(tr_spec2)}')
    for c in E2B_SPEC:
        print(f'    {CANONICAL[c]:12s}: {(lbl2[tr_spec2]==c).sum()}')

    tr_ld2 = DataLoader(Subset(full_t2, tr_spec2), batch_size=16,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    vl_ld2 = DataLoader(Subset(full_v2, vl_idx2), batch_size=32,
                        shuffle=False, num_workers=2)

    model2 = timm.create_model(E2B_MODEL, pretrained=True, num_classes=NUM_CLASSES)
    nf2    = model2.num_features
    model2.head.fc = nn.Sequential(
        nn.Linear(nf2, 256), nn.GELU(),
        nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES)
    )
    model2 = model2.to(DEVICE)
    print(f'  E2B num_features={nf2}')

    cw2   = torch.tensor([WMAP2.get(c, 1.0) for c in CANONICAL],
                          dtype=torch.float, device=DEVICE)
    crit2 = nn.CrossEntropyLoss(weight=cw2, label_smoothing=0.10)
    opt2  = torch.optim.AdamW(model2.parameters(), lr=E2B_LR, weight_decay=E2B_WD)
    base_lrs2 = [E2B_LR] * len(opt2.param_groups)
    scaler2   = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))

    def spec_f1_2(model, loader, spec_idx):
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().tolist())
                gts.extend(lbls.tolist())
        pairs = [(p, l) for p, l in zip(preds, gts) if l in spec_idx]
        if not pairs: return 0.0
        pp, ll = zip(*pairs)
        return f1_score(list(ll), list(pp), average='macro', zero_division=0)

    best2 = 0.0; pat2 = 0
    print(f'\nTraining E2B ({E2B_EPOCHS} epochs, warmup={E2B_WARMUP} ep)...')
    for epoch in range(1, E2B_EPOCHS + 1):
        model2.train(); tl = tot = 0
        cosine_warmup_lr(opt2, epoch - 1, E2B_WARMUP, E2B_EPOCHS, base_lrs2)
        for imgs, lbls in tr_ld2:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            xm, ya, yb, lam = mixup(imgs, lbls, alpha=0.4)
            opt2.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                out  = model2(xm)
                loss = lam * crit2(out, ya) + (1 - lam) * crit2(out, yb)
            scaler2.scale(loss).backward()
            scaler2.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
            scaler2.step(opt2); scaler2.update()
            tl += loss.item() * imgs.size(0); tot += imgs.size(0)
        vf = spec_f1_2(model2, vl_ld2, E2B_SPEC)
        lr_now = opt2.param_groups[0]['lr']
        if vf > best2:
            best2 = vf
            torch.save({
                'model_state_dict': model2.state_dict(),
                'model_name':       E2B_MODEL,
                'num_classes':      NUM_CLASSES,
                'class_names':      CANONICAL,
                'val_f1':           best2,
                'epoch':            epoch,
                'head_type':        'convnext_deep',
                'specialty':        E2B_SPEC,
                'img_size':         E2B_IMG,
            }, E2B_PATH)
            pat2 = 0; mark = ' ✓'
        else:
            pat2 += 1; mark = ''
        if epoch % 5 == 0 or mark:
            print(f'  Ep{epoch:3d}/{E2B_EPOCHS}  loss:{tl/tot:.4f}'
                  f'  spec_f1:{vf:.4f}  lr:{lr_now:.2e}{mark}')
        if pat2 >= PATIENCE:
            print(f'  Early stop ep{epoch}'); break

    print(f'\n✅ E2B done — best spec F1: {best2:.4f}  →  {E2B_PATH}')

# ──────────────────────────────────────────────────────────────
# CELL 6 — STAGE 3: Expert 3 (Swin-Base 224px) — FIXED
# ──────────────────────────────────────────────────────────────
E3_PATH = os.path.join(OUT, 'expert3_best.pt')

if os.path.exists(E3_PATH) and not FORCE_E3:
    print('⏭  E3 exists — skipping.')
else:
    print('=' * 62)
    print('  STAGE 3: Expert 3 — Swin-Base 224px (5-class generalist)')
    print('=' * 62)
    set_seed(42)

    E3_MODEL  = 'swin_base_patch4_window7_224'
    E3_IMG    = 224
    E3_LR     = 3e-4
    E3_WD     = 5e-2
    E3_WARMUP = 10
    LLRD      = 0.75

    aug3 = transforms.Compose([
        transforms.Resize((E3_IMG + 32, E3_IMG + 32)),
        transforms.RandomResizedCrop(E3_IMG, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), _NORM,
        transforms.RandomErasing(p=0.25),
    ])
    val_tf3 = transforms.Compose([transforms.Resize((E3_IMG, E3_IMG)),
                                   transforms.ToTensor(), _NORM])

    dr3     = get_data_root(DATA_DIR)
    full_v3 = build_dataset(dr3, val_tf3)
    full_t3 = build_dataset(dr3, aug3)
    tr_idx3 = np.load(os.path.join(SPLIT_DIR, 'train_idx.npy'))
    vl_idx3 = np.load(os.path.join(SPLIT_DIR, 'val_idx.npy'))

    tr_ld3 = DataLoader(Subset(full_t3, tr_idx3), batch_size=24,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    vl_ld3 = DataLoader(Subset(full_v3, vl_idx3), batch_size=32,
                        shuffle=False, num_workers=2)
    print(f'  Train: {len(tr_idx3)}  Val: {len(vl_idx3)}')

    model3 = SwinWithPool(E3_MODEL, NUM_CLASSES, pretrained=True).to(DEVICE)
    nf3    = model3.num_features
    print(f'  E3 num_features={nf3}')
    with torch.no_grad():
        _dummy = torch.randn(2, 3, E3_IMG, E3_IMG).to(DEVICE)
        _out   = model3(_dummy)
        assert _out.shape == (2, NUM_CLASSES), \
            f'SwinWithPool output {_out.shape} — expected (2, {NUM_CLASSES})'
        print(f'  ✅ Output shape verified: {tuple(_out.shape)}')
    del _dummy, _out

    n_stages    = len(model3.backbone.layers)
    param_groups = []
    param_groups.append({
        'params': list(model3.backbone.patch_embed.parameters()),
        'lr': E3_LR * LLRD ** (n_stages + 1),
        'weight_decay': E3_WD
    })
    for i, layer in enumerate(model3.backbone.layers):
        param_groups.append({
            'params': list(layer.parameters()),
            'lr': E3_LR * LLRD ** (n_stages - i),
            'weight_decay': E3_WD
        })
    head_params = (list(model3.backbone.norm.parameters())
                   + list(model3.head.parameters()))
    param_groups.append({'params': head_params, 'lr': E3_LR, 'weight_decay': 0.0})
    base_lrs3 = [pg['lr'] for pg in param_groups]

    print('  LLRD param groups:')
    for i, pg in enumerate(param_groups):
        nparams = sum(p.numel() for p in pg['params'])
        print(f'    group {i}: lr={pg["lr"]:.2e}  params={nparams:,}')

    opt3    = torch.optim.AdamW(param_groups)
    crit3   = nn.CrossEntropyLoss(label_smoothing=0.10)
    scaler3 = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))

    def full_f1(model, loader):
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                out = model(imgs.to(DEVICE))
                preds.extend(out.argmax(1).cpu().tolist())
                gts.extend(lbls.tolist())
        return f1_score(gts, preds, average='macro', zero_division=0)

    best3 = 0.0; pat3 = 0
    print(f'\nTraining E3 ({E3_EPOCHS} epochs, LLRD={LLRD}, warmup={E3_WARMUP} ep)...')
    for epoch in range(1, E3_EPOCHS + 1):
        model3.train(); tl = tot = 0
        cosine_warmup_lr(opt3, epoch - 1, E3_WARMUP, E3_EPOCHS, base_lrs3)
        for imgs, lbls in tr_ld3:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt3.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                out3 = model3(imgs)
                loss = crit3(out3, lbls)
            scaler3.scale(loss).backward()
            scaler3.unscale_(opt3)
            torch.nn.utils.clip_grad_norm_(model3.parameters(), 1.0)
            scaler3.step(opt3); scaler3.update()
            tl += loss.item() * imgs.size(0); tot += imgs.size(0)
        vf = full_f1(model3, vl_ld3)
        lr_now = opt3.param_groups[-1]['lr']
        if vf > best3:
            best3 = vf
            torch.save({
                'model_state_dict': model3.state_dict(),
                'model_name':       E3_MODEL,
                'num_classes':      NUM_CLASSES,
                'class_names':      CANONICAL,
                'val_f1':           best3,
                'epoch':            epoch,
                'head_type':        'swin_wrapped',
                'img_size':         E3_IMG,
            }, E3_PATH)
            pat3 = 0; mark = ' ✓'
        else:
            pat3 += 1; mark = ''
        if epoch % 5 == 0 or mark:
            print(f'  Ep{epoch:3d}/{E3_EPOCHS}  loss:{tl/tot:.4f}'
                  f'  val_f1:{vf:.4f}  head_lr:{lr_now:.2e}{mark}')
        if pat3 >= PATIENCE:
            print(f'  Early stop ep{epoch}'); break

    print(f'\n✅ E3 done — best macro F1: {best3:.4f}  →  {E3_PATH}')

# ──────────────────────────────────────────────────────────────
# CELL 7 — STAGE 4: AttentionGate
# ──────────────────────────────────────────────────────────────
GATE_PATH = os.path.join(OUT, 'gate_best.pt')
CTX_PATH  = os.path.join(OUT, 'ctx_best.pt')

if os.path.exists(GATE_PATH) and os.path.exists(CTX_PATH) and not FORCE_GATE:
    print('⏭  Gate exists — skipping.')
else:
    print('=' * 62)
    print('  STAGE 4: AttentionGate (cross-attention over expert tokens)')
    print('=' * 62)
    set_seed(42)

    E1_ZERO = [0, 1]; E2_ZERO = [2, 3]
    GATE_BS = 16; GATE_LR = 2e-4; GATE_WD = 1e-4; DIV_W = 0.15
    GATE_WARMUP = 5

    print('Loading experts...')
    _E1B = load_expert(E1B_PATH, DEVICE)
    _E2B = load_expert(E2B_PATH, DEVICE)
    _E3  = load_expert(E3_PATH,  DEVICE)

    tf224g = transforms.Compose([transforms.Resize((224, 224)),
                                  transforms.ToTensor(), _NORM])
    tf320g = transforms.Compose([transforms.Resize((320, 320)),
                                  transforms.ToTensor(), _NORM])
    tf256g = transforms.Compose([transforms.Resize((256, 256)),
                                  transforms.ToTensor(), _NORM])

    drG      = get_data_root(DATA_DIR)
    full224G = build_dataset(drG, tf224g)
    full320G = build_dataset(drG, tf320g)
    full256G = build_dataset(drG, tf256g)
    tr_idxG  = np.load(os.path.join(SPLIT_DIR, 'train_idx.npy'))
    vl_idxG  = np.load(os.path.join(SPLIT_DIR, 'val_idx.npy'))
    te_idxG  = np.load(os.path.join(SPLIT_DIR, 'test_idx.npy'))

    def mask_norm(s, zi):
        m = s.clone(); m[:, zi] = 0.0
        return m / m.sum(-1, keepdim=True).clamp(min=1e-8)

    @torch.no_grad()
    def cache_all(idx, tag):
        print(f'  Caching {tag}...', end=' ', flush=True)
        s1l, s2l, s3l, ll = [], [], [], []
        ld320 = DataLoader(Subset(full320G, idx), batch_size=32, shuffle=False, num_workers=2)
        ld256 = DataLoader(Subset(full256G, idx), batch_size=32, shuffle=False, num_workers=2)
        ld224 = DataLoader(Subset(full224G, idx), batch_size=32, shuffle=False, num_workers=2)
        for (i320, lbls), (i256, _), (i224, _) in zip(ld320, ld256, ld224):
            s1l.append(F.softmax(_E1B(i320.to(DEVICE)), -1).cpu())
            s2l.append(F.softmax(_E2B(i256.to(DEVICE)), -1).cpu())
            s3l.append(F.softmax(_E3( i224.to(DEVICE)), -1).cpu())
            ll.append(lbls)
        s1  = torch.cat(s1l); s2 = torch.cat(s2l)
        s3  = torch.cat(s3l); lbl = torch.cat(ll)
        s1m = mask_norm(s1, E1_ZERO)
        s2m = mask_norm(s2, E2_ZERO)
        print(f'done — {len(lbl)} samples')
        return s1m, s2m, s3, lbl

    print('\nCaching expert outputs...')
    tr_s1m, tr_s2m, tr_s3, tr_lbl = cache_all(tr_idxG, 'train')
    vl_s1m, vl_s2m, vl_s3, vl_lbl = cache_all(vl_idxG, 'val')
    te_s1m, te_s2m, te_s3, te_lbl = cache_all(te_idxG, 'test')

    print('\nExpert solo F1 (val, macro):')
    for nm, s, lbl in [('E1B(masked)', vl_s1m, vl_lbl),
                       ('E2B(masked)', vl_s2m, vl_lbl),
                       ('E3',          vl_s3,  vl_lbl)]:
        f = f1_score(lbl.tolist(), s.argmax(1).tolist(), average='macro', zero_division=0)
        print(f'  {nm:16s}: {f:.4f}')

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
                nn.Linear(64, D), nn.LayerNorm(D)
            )
            self.expert_proj = nn.ModuleList([nn.Linear(5, D) for _ in range(n_experts)])
            self.cross_attn  = nn.MultiheadAttention(D, num_heads=4, batch_first=True, dropout=0.1)
            self.out_head    = nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, n_experts))
            self.n = n_experts

        def forward(self, img, experts):
            q  = self.ctx_cnn(img).unsqueeze(1)
            kv = torch.stack([self.expert_proj[i](experts[i])
                               for i in range(self.n)], dim=1)
            attn_out, _ = self.cross_attn(q, kv, kv)
            return F.softmax(self.out_head(attn_out.squeeze(1)), dim=-1)

    gate_net = AttentionGate(n_experts=3).to(DEVICE)
    params   = list(gate_net.parameters())
    gate_opt = torch.optim.AdamW(params, lr=GATE_LR, weight_decay=GATE_WD)
    gate_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        gate_opt, T_0=20, T_mult=2, eta_min=1e-6
    )

    class JointDS(Dataset):
        def __init__(self, imgs, s1m, s2m, s3, lbl):
            self.imgs = imgs
            self.s1m, self.s2m, self.s3 = s1m, s2m, s3
            self.lbl  = lbl
        def __len__(self): return len(self.lbl)
        def __getitem__(self, i):
            img, _ = self.imgs[i]
            return img, self.s1m[i], self.s2m[i], self.s3[i], self.lbl[i]

    tr_jl = DataLoader(JointDS(Subset(full224G, tr_idxG), tr_s1m, tr_s2m, tr_s3, tr_lbl),
                       batch_size=GATE_BS, shuffle=True, num_workers=2)
    vl_jl = DataLoader(JointDS(Subset(full224G, vl_idxG), vl_s1m, vl_s2m, vl_s3, vl_lbl),
                       batch_size=GATE_BS, shuffle=False, num_workers=2)
    te_jl = DataLoader(JointDS(Subset(full224G, te_idxG), te_s1m, te_s2m, te_s3, te_lbl),
                       batch_size=GATE_BS, shuffle=False, num_workers=2)

    def gate_fwd(img, s1m, s2m, s3):
        img = img.to(DEVICE)
        s1m, s2m, s3 = s1m.to(DEVICE), s2m.to(DEVICE), s3.to(DEVICE)
        gw  = gate_net(img, [s1m, s2m, s3])
        c1  = s1m.max(-1, keepdim=True)[0]
        c2  = s2m.max(-1, keepdim=True)[0]
        c3  = s3.max(-1,  keepdim=True)[0]
        ew1 = gw[:, 0:1] * c1; ew2 = gw[:, 1:2] * c2; ew3 = gw[:, 2:3] * c3
        ew_s = (ew1 + ew2 + ew3).clamp(min=1e-8)
        ens  = (ew1 / ew_s) * s1m + (ew2 / ew_s) * s2m + (ew3 / ew_s) * s3
        return ens, gw

    def div_loss(gw):
        mw  = gw.mean(0)
        ent = -(mw * (mw + 1e-8).log()).sum()
        return (torch.log(torch.tensor(3.0)) - ent) / torch.log(torch.tensor(3.0))

    @torch.no_grad()
    def eval_gate(loader):
        gate_net.eval(); ap, al, agw = [], [], []
        for img, s1m, s2m, s3, lbls in loader:
            ens, gw = gate_fwd(img, s1m, s2m, s3)
            ap.extend(ens.argmax(1).cpu().tolist())
            al.extend(lbls.tolist())
            agw.append(gw.cpu())
        return (f1_score(al, ap, average='macro', zero_division=0),
                torch.cat(agw).mean(0))

    best_gf1 = 0.0; patg = 0
    print(f'\nTraining AttentionGate ({GATE_EPOCHS} epochs)...')
    for epoch in range(1, GATE_EPOCHS + 1):
        gate_net.train(); tl = tot = 0
        for img, s1m, s2m, s3, lbls in tr_jl:
            lbls = lbls.to(DEVICE)
            gate_opt.zero_grad()
            ens, gw = gate_fwd(img, s1m, s2m, s3)
            ce   = F.nll_loss(torch.log(ens.clamp(min=1e-8)), lbls)
            loss = ce + DIV_W * div_loss(gw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            gate_opt.step()
            tl += loss.item() * img.size(0); tot += img.size(0)
        vf1, mgw = eval_gate(vl_jl)
        gate_sch.step()
        if vf1 > best_gf1:
            best_gf1 = vf1
            torch.save(gate_net.state_dict(), GATE_PATH)
            patg = 0; mark = ' ✓'
        else:
            patg += 1; mark = ''
        gws = '  '.join([f'E{i+1}:{mgw[i]:.3f}' for i in range(3)])
        if epoch % 5 == 0 or mark:
            print(f'  Ep{epoch:3d}/{GATE_EPOCHS}  loss:{tl/tot:.4f}'
                  f'  val_f1:{vf1:.4f}  [{gws}]{mark}')
        if patg >= PATIENCE:
            print(f'  Early stop ep{epoch}'); break

    gate_net.load_state_dict(torch.load(GATE_PATH, map_location=DEVICE))
    tf1, tgw = eval_gate(te_jl)
    e3_only  = f1_score(te_lbl.tolist(), te_s3.argmax(1).tolist(),
                        average='macro', zero_division=0)
    print(f'\n✅ Gate done')
    print(f'   Test Macro F1 : {tf1:.4f}')
    print(f'   Gate weights  : E1B={tgw[0]:.3f}  E2B={tgw[1]:.3f}  E3={tgw[2]:.3f}')
    print(f'   E3 alone      : {e3_only:.4f}')
    print(f'   MoE gain      : {tf1 - e3_only:+.4f}')

# ──────────────────────────────────────────────────────────────
# CELL 8 — STAGE 5: Calibration + 8-crop TTA + ONNX
# ──────────────────────────────────────────────────────────────
import time, json, onnx, onnxruntime as ort

TEMP_PATH = os.path.join(OUT, 'temperature.pt')
OOD_PATH  = os.path.join(OUT, 'ood_config.pt')
ONNX_PATH = os.path.join(OUT, 'moe_pipeline.onnx')

if os.path.exists(TEMP_PATH) and os.path.exists(OOD_PATH) and not FORCE_CAL:
    print('⏭  Calibration done.')
else:
    print('=' * 62)
    print('  STAGE 5: Calibration + TTA + ONNX')
    print('=' * 62)
    set_seed(42)

    E1_Z = [0, 1]; E2_Z = [2, 3]

    cE1B = load_expert(E1B_PATH, DEVICE)
    cE2B = load_expert(E2B_PATH, DEVICE)
    cE3  = load_expert(E3_PATH,  DEVICE)
    print()

    class _AttentionGate(nn.Module):
        D = 64
        def __init__(self, n_experts=3):
            super().__init__()
            D = self.D
            self.ctx_cnn = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=4, padding=2), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, D), nn.LayerNorm(D)
            )
            self.expert_proj = nn.ModuleList([nn.Linear(5, D) for _ in range(n_experts)])
            self.cross_attn  = nn.MultiheadAttention(D, num_heads=4, batch_first=True, dropout=0.1)
            self.out_head    = nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, n_experts))
            self.n = n_experts
        def forward(self, img, experts):
            q  = self.ctx_cnn(img).unsqueeze(1)
            kv = torch.stack([self.expert_proj[i](experts[i])
                               for i in range(self.n)], dim=1)
            attn_out, _ = self.cross_attn(q, kv, kv)
            return F.softmax(self.out_head(attn_out.squeeze(1)), dim=-1)

    cGate = _AttentionGate().to(DEVICE).eval()
    cGate.load_state_dict(torch.load(GATE_PATH, map_location=DEVICE))

    def _mn(s, zi):
        m = s.clone(); m[:, zi] = 0.0
        return m / m.sum(-1, keepdim=True).clamp(min=1e-8)

    @torch.no_grad()
    def moe_infer(imgs224):
        i320 = F.interpolate(imgs224, (320, 320), mode='bilinear', align_corners=False)
        i256 = F.interpolate(imgs224, (256, 256), mode='bilinear', align_corners=False)
        s1   = F.softmax(cE1B(i320), -1)
        s2   = F.softmax(cE2B(i256), -1)
        s3   = F.softmax(cE3(imgs224), -1)
        s1m  = _mn(s1, E1_Z); s2m = _mn(s2, E2_Z)
        gw   = cGate(imgs224, [s1m, s2m, s3])
        c1   = s1m.max(-1, keepdim=True)[0]
        c2   = s2m.max(-1, keepdim=True)[0]
        c3   = s3.max(-1,  keepdim=True)[0]
        ew1  = gw[:, 0:1] * c1; ew2 = gw[:, 1:2] * c2; ew3 = gw[:, 2:3] * c3
        ew_s = (ew1 + ew2 + ew3).clamp(min=1e-8)
        return (ew1/ew_s)*s1m + (ew2/ew_s)*s2m + (ew3/ew_s)*s3

    @torch.no_grad()
    def moe_infer_tta(imgs224):
        H, W = imgs224.shape[2], imgs224.shape[3]
        cs   = int(H * 0.875)
        crops = [
            imgs224,
            torch.flip(imgs224, dims=[3]),
            torch.flip(imgs224, dims=[2]),
            torch.flip(imgs224, dims=[2, 3]),
            F.interpolate(imgs224[:, :, :cs, :cs],   (H, W), mode='bilinear', align_corners=False),
            F.interpolate(imgs224[:, :, H-cs:, :cs], (H, W), mode='bilinear', align_corners=False),
            F.interpolate(imgs224[:, :, :cs, W-cs:], (H, W), mode='bilinear', align_corners=False),
            F.interpolate(imgs224[:, :, H-cs:, W-cs:], (H, W), mode='bilinear', align_corners=False),
        ]
        return torch.stack([moe_infer(c) for c in crops]).mean(0)

    tf224c   = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(), _NORM])
    drC      = get_data_root(DATA_DIR)
    full224C = build_dataset(drC, tf224c)
    val_idxC = np.load(os.path.join(SPLIT_DIR, 'val_idx.npy'))
    tr_idxC  = np.load(os.path.join(SPLIT_DIR, 'train_idx.npy'))
    te_idxC  = np.load(os.path.join(SPLIT_DIR, 'test_idx.npy'))
    def ldrc(idx, bs=32):
        return DataLoader(Subset(full224C, idx), batch_size=bs, shuffle=False, num_workers=2)

    print('\n── Temperature scaling...')
    all_log, all_lbl = [], []
    with torch.no_grad():
        for imgs, lbls in ldrc(val_idxC):
            ens = moe_infer(imgs.to(DEVICE))
            all_log.append(torch.log(ens.clamp(min=1e-8)).cpu())
            all_lbl.append(lbls)
    lp = torch.cat(all_log); la = torch.cat(all_lbl)
    T  = nn.Parameter(torch.tensor(1.5))
    to = torch.optim.LBFGS([T], lr=0.05, max_iter=500, tolerance_change=1e-9)
    def tcl():
        to.zero_grad()
        F.cross_entropy(lp / T.clamp(min=0.5), la).backward()
        return F.cross_entropy(lp / T.clamp(min=0.5), la)
    to.step(tcl)
    T_val = max(float(T.item()), 1.0)
    print(f'   T = {T_val:.4f}')
    torch.save({'temperature': T_val}, TEMP_PATH)

    print('── OOD energy threshold...')
    es = []
    with torch.no_grad():
        for imgs, _ in ldrc(tr_idxC):
            logits = cE3(imgs.to(DEVICE))
            es.extend((-T_val * torch.logsumexp(logits / T_val, -1)).cpu().tolist())
    thr = float(np.percentile(np.array(es), 95))
    print(f'   95th-pct threshold = {thr:.4f}')
    torch.save({'ood_threshold': thr, 'temperature': T_val}, OOD_PATH)

    print('── Calibrated test accuracy (8-crop TTA)...')
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, lbls in ldrc(te_idxC):
            ens = moe_infer_tta(imgs.to(DEVICE))
            cal = F.softmax(torch.log(ens.clamp(min=1e-8)) / T_val, -1)
            all_p.extend(cal.argmax(1).cpu().tolist())
            all_l.extend(lbls.tolist())
    tf1_cal = f1_score(all_l, all_p, average='macro', zero_division=0)
    print(f'   Calibrated Test Macro F1 (8-crop TTA): {tf1_cal:.4f}')
    print(classification_report(all_l, all_p,
                                labels=list(range(len(CANONICAL))),
                                target_names=CANONICAL, zero_division=0))

    print('── ONNX export (single-pass, opset 17)...')
    class FullPipeline(nn.Module):
        def __init__(self, e1, e2, e3, gate, e1z, e2z, T):
            super().__init__()
            self.e1, self.e2, self.e3, self.gate = e1, e2, e3, gate
            self.T = float(T)
            m1 = torch.zeros(NUM_CLASSES); m1[e1z] = 1.0
            m2 = torch.zeros(NUM_CLASSES); m2[e2z] = 1.0
            self.register_buffer('m1', m1)
            self.register_buffer('m2', m2)
        def forward(self, x):
            i320 = F.interpolate(x, (320, 320), mode='bilinear', align_corners=False)
            i256 = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
            s1   = F.softmax(self.e1(i320), -1)
            s2   = F.softmax(self.e2(i256), -1)
            s3   = F.softmax(self.e3(x),    -1)
            s1m  = s1 * (1 - self.m1)
            s1m  = s1m / s1m.sum(-1, keepdim=True).clamp(min=1e-8)
            s2m  = s2 * (1 - self.m2)
            s2m  = s2m / s2m.sum(-1, keepdim=True).clamp(min=1e-8)
            gw   = self.gate(x, [s1m, s2m, s3])
            c1   = s1m.max(-1, keepdim=True)[0]
            c2   = s2m.max(-1, keepdim=True)[0]
            c3   = s3.max(-1,  keepdim=True)[0]
            ew1  = gw[:, 0:1] * c1; ew2 = gw[:, 1:2] * c2; ew3 = gw[:, 2:3] * c3
            ew_s = (ew1 + ew2 + ew3).clamp(min=1e-8)
            ens  = (ew1/ew_s)*s1m + (ew2/ew_s)*s2m + (ew3/ew_s)*s3
            return F.softmax(torch.log(ens.clamp(min=1e-8)) / self.T, -1)

    pipeline = FullPipeline(cE1B, cE2B, cE3, cGate, E1_Z, E2_Z, T_val).to(DEVICE).eval()
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    warnings.filterwarnings('ignore')
    torch.onnx.export(
        pipeline, dummy, ONNX_PATH,
        export_params=True, opset_version=17, dynamo=False,
        do_constant_folding=True,
        input_names=['image'], output_names=['probs'],
        dynamic_axes={'image': {0: 'B'}, 'probs': {0: 'B'}}
    )
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print(f'   ✅ ONNX saved ({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)')

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        qp = os.path.join(OUT, 'moe_pipeline_int8.onnx')
        quantize_dynamic(ONNX_PATH, qp, weight_type=QuantType.QInt8)
        print(f'   ✅ INT8: {os.path.getsize(qp)/1e6:.1f} MB')
    except Exception as e:
        print(f'   INT8 skipped: {e}')

    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    d    = np.random.randn(1, 3, 224, 224).astype(np.float32)
    nm   = sess.get_inputs()[0].name
    for _ in range(3): sess.run(None, {nm: d})
    t = time.perf_counter()
    for _ in range(20): sess.run(None, {nm: d})
    print(f'   Benchmark: {(time.perf_counter()-t)/20*1000:.1f} ms/img (CPU)')

    json.dump({
        'method':      'specialty_moe_v4',
        'test_f1_tta': tf1_cal,
        'e1_zero': E1_Z, 'e2_zero': E2_Z,
        'temperature': T_val,
    }, open(os.path.join(OUT, 'final_config.json'), 'w'))
    print(f'\n✅ Stage 5 complete — Final MoE F1 (8-crop TTA): {tf1_cal:.4f}')

# ──────────────────────────────────────────────────────────────
# CELL 9 — Summary
# ──────────────────────────────────────────────────────────────
import json
print('\n' + '=' * 62)
print('  🎉  MoE PIPELINE v4 COMPLETE')
print('=' * 62)

files = [
    ('expert1B_best.pt',       'E1B EfficientNetV2-M 320px (RedRot+YellowLeaf)'),
    ('expert2B_best.pt',       'E2B ConvNeXt-Base 256px (Mosaic+Rust)'),
    ('expert3_best.pt',        'E3  Swin-Base 224px [SwinWithPool] (5-class generalist)'),
    ('gate_best.pt',           'AttentionGate (cross-attention)'),
    ('temperature.pt',         'Temperature scaling'),
    ('ood_config.pt',          'OOD threshold'),
    ('moe_pipeline.onnx',      'Full pipeline ONNX (single-pass)'),
    ('moe_pipeline_int8.onnx', 'Full pipeline ONNX INT8 quantized'),
    ('final_config.json',      'Pipeline config'),
]
for fname, desc in files:
    p  = os.path.join(OUT, fname)
    sz = f'{os.path.getsize(p)/1e6:.1f} MB' if os.path.exists(p) else 'MISSING ❌'
    print(f'  {chr(9989) if os.path.exists(p) else chr(10060)}  {fname:30s} {sz:10s} — {desc}')

print()
try:
    cfg = json.load(open(os.path.join(OUT, 'final_config.json')))
    print(f'  Final Test Macro F1 (TTA): {cfg["test_f1_tta"]:.4f}')
    print(f'  Temperature T            : {cfg["temperature"]:.4f}')
except Exception:
    pass