"""Notebook 24 - Multimodal models: CLIP and a vision-language model, measured

Builds 24_multimodal.ipynb. Run: python tools/build_notebooks.py 24
Read it after notebook 21 (embeddings). Result commentary cells contain {{R_...}} markers filled after execution.
"""
from _common import *  # noqa: F401,F403

nb = NB()
nb.md(r'''
# 24 · Multimodal models: images as embeddings and as tokens

Everything so far was text. Two families of models extend it to images, and they are used for different jobs:

| family | what it produces | used for | here |
|---|---|---|---|
| **contrastive image-text encoders** (CLIP, SigLIP) | one vector per image and per text, in a shared space | zero-shot classification, image search, deduplication, filtering training data | CLIP ViT-B/32 |
| **vision-language models** (VLMs: GPT-4o, Claude, Gemini, Qwen2.5-VL, Llama 3.2 Vision) | text, conditioned on images | describing, answering questions about, and extracting data from images and documents | `qwen2.5vl:3b` in Ollama |

This notebook measures both on the same tasks:
- fine-grained classification of 37 cat and dog breeds, zero-shot and with a few labelled examples;
- what an image costs a VLM in tokens and time, as resolution grows;
- extracting structured fields from invoice images;
- a well-known failure: text written on an image fools CLIP.

Read it **after notebook 21**: CLIP is trained with the same contrastive loss as text embedding models, with images on one side.

**Sources.** Alammar and Grootendorst, *Hands-On Large Language Models*, ch. 9 (multimodal LLMs); Huyen, *AI Engineering*, ch. 1–2 (multimodal foundation models); Stanford CME 295, vision-language models.
''')
nb.md(r'''
## Background: how a transformer sees

### Images as sequences of patches

The **Vision Transformer** (ViT; Dosovitskiy et al., 2021) cuts an image into fixed-size patches (16×16 or 32×32 pixels). Each patch is flattened and linearly projected to a vector, and the vectors are processed as a sequence of tokens by the same transformer blocks as text (notebook 02), with position embeddings for the 2-D layout. A 224×224 image with 32×32 patches is a sequence of 7 × 7 = 49 tokens, plus a class token whose final vector represents the whole image.

### CLIP: learning a shared space from captions

CLIP (Radford et al., 2021) trains an image encoder and a text encoder together, on 400 million (image, caption) pairs from the web, with the contrastive InfoNCE loss of notebook 21. In a batch, each image must be closer to its own caption than to every other caption, and vice versa. Nobody labels classes, yet the space it learns supports **zero-shot classification**:
1. write one sentence per class ("a photo of a beagle");
2. embed the sentences;
3. assign each image to the class whose sentence is most similar.

The wording of those sentences matters, as prompts do for LLMs: CLIP's authors averaged 80 templates per class (prompt ensembling). With a few labelled images per class, a **linear probe** (logistic regression on the frozen image embeddings) usually beats zero-shot. That is the image version of notebook 17's embeddings + classifier. SigLIP (Zhai et al., 2023) replaces the batch-wide softmax with a per-pair sigmoid loss, which trains better and is the encoder inside many current VLMs.

### VLMs: an image encoder wired into an LLM

A VLM feeds image information into a language model. Two designs dominate:

- **Projected visual tokens** (LLaVA, Liu et al., 2023; Qwen-VL, InternVL). A vision encoder (ViT/SigLIP) turns the image into patch vectors, a small projector maps them into the LLM's embedding space, and they are inserted into the prompt as ordinary tokens. The LLM attends to them like words. Training is in two steps: first align the projector on captions, then instruction-tune on visual question answering.
- **Cross-attention** (Flamingo, Alayrac et al., 2022; Llama 3.2 Vision). The LLM's layers get extra cross-attention blocks that read the image features, as in the encoder-decoder of notebook 17, so images do not occupy the context.

Qwen2.5-VL uses the first design with **dynamic resolution**: the number of visual tokens grows with the number of pixels (one token per 28×28 area after merging), instead of resizing every image to a fixed size. Detail is preserved, and the bill grows with it: **an image is paid for in tokens**, which means prefill time, context length and money (notebook 23).

### What goes wrong

- **Hallucination**: describing objects that are not there.
- **Fine-grained confusion**: telling an American pit bull terrier from a Staffordshire bull terrier.
- **OCR errors** on small or degraded text.
- **Typographic attacks**: Goh et al. (2021) found that CLIP reads text in images so well that a label written on an apple ("iPod") makes it classify the apple as an iPod. We test this.

### Key papers

- Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT).
- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- Zhai et al. (2023), *Sigmoid Loss for Language Image Pre-Training* (SigLIP).
- Alayrac et al. (2022), *Flamingo: a Visual Language Model for Few-Shot Learning*.
- Liu et al. (2023), *Visual Instruction Tuning* (LLaVA).
- Bai et al. (2025), *Qwen2.5-VL Technical Report*.
- Goh et al. (2021), *Multimodal Neurons in Artificial Neural Networks* (typographic attacks).
- Parkhi et al. (2012), *Cats and Dogs* (the Oxford-IIIT Pet dataset).
''')
nb.code(setup_apps("llm-edu-suite-24-multimodal"))
nb.code(r'''
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import io, re, base64, ollama, datasets, transformers
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub.utils import logging as hf_logging; hf_logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error(); transformers.utils.logging.disable_progress_bar(); datasets.disable_progress_bars()
from sklearn.linear_model import LogisticRegression
from llm_edu_suite.evaluation import bootstrap_ci, paired_bootstrap_diff
device = "cuda" if torch.cuda.is_available() else "cpu"
VLM = "qwen2.5vl:3b"
ollama_client = ollama.Client(host=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

train = load_dataset("timm/oxford-iiit-pet", split="train")
test = load_dataset("timm/oxford-iiit-pet", split="test").shuffle(seed=0).select(range(600))
NAMES = [n.replace("_", " ") for n in train.features["label"].names]
print(f"{len(NAMES)} breeds | train {len(train)} images | test sample {len(test)} | e.g. {NAMES[:6]}")
y_test = np.array(test["label"])
''')
nb.md(r'''
## 1. CLIP zero-shot: the classes are just sentences

We compute CLIP's image and text embeddings explicitly: the vision transformer's pooled output projected into the shared space, and the same for the text transformer. Then we compare three ways of writing the class sentences: the bare breed name, a single template, and an ensemble of templates averaged per class.
''')
nb.code(r'''
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def image_embeddings(images, batch=64):
    out = []
    for i in range(0, len(images), batch):
        px = proc(images=[im.convert("RGB") for im in images[i:i + batch]], return_tensors="pt")["pixel_values"].to(device)
        out.append(clip.visual_projection(clip.vision_model(pixel_values=px).pooler_output))     # ViT [CLS] -> shared space
    return F.normalize(torch.cat(out), dim=-1)

@torch.no_grad()
def text_embeddings(texts):
    tk = proc(text=texts, return_tensors="pt", padding=True).to(device)
    return F.normalize(clip.text_projection(clip.text_model(input_ids=tk["input_ids"], attention_mask=tk["attention_mask"]).pooler_output), dim=-1)

t0 = time.time(); E_test = image_embeddings(test["image"]); ms_img = 1000 * (time.time() - t0) / len(test)
TEMPLATES = ["a photo of a {}, a type of pet.", "a close-up photo of a {}.", "a photo of the {}, a breed of {}.",
             "a blurry photo of a {}.", "a cute photo of a {}."]
CATS = {"abyssinian", "bengal", "birman", "bombay", "british shorthair", "egyptian mau", "maine coon", "persian", "ragdoll",
        "russian blue", "siamese", "sphynx"}                           # the 12 cat breeds of Oxford-IIIT Pet; the other 25 are dogs
species = ["cat" if n in CATS else "dog" for n in NAMES]
def class_embedding(templates):
    return F.normalize(torch.stack([text_embeddings([t.format(n, s) for t in templates]).mean(0) for n, s in zip(NAMES, species)]), dim=-1)

ZS = {}
for label, W in [("bare name", text_embeddings(NAMES)),
                 ("one template", class_embedding(TEMPLATES[:1])),
                 ("ensemble of 5 templates", class_embedding(TEMPLATES))]:
    pred = (E_test @ W.T).argmax(1).cpu().numpy()
    ZS[label] = (pred == y_test).astype(float)
    m, lo, hi = bootstrap_ci(ZS[label])
    print(f"zero-shot, {label:24s} accuracy {m:.1%} [{lo:.1%}, {hi:.1%}]")
print(f"CLIP image encoding: {ms_img:.1f} ms per image (batch of 64, {device})")
''')
nb.md(r'''
**What the numbers say.** With no labelled pet images at all, CLIP ViT-B/32 names the breed of **86.3 %** of the test images among 37 options. The prompt is part of the model:

- the bare breed name gives 80.0 %;
- one template ("a photo of a {}, a type of pet.") gives 85.2 %;
- averaging five templates gives 86.3 %.

The template tells CLIP what kind of image to expect, the way a caption on the web would describe it, and it disambiguates names ("boxer" the dog, not the athlete). Averaging templates smooths out the quirks of any single wording. Encoding costs 5.9 ms per image in batches on this GPU, so CLIP can label or index a million images in under two hours.

## 2. A few labelled images: the linear probe

Logistic regression on frozen CLIP image embeddings, trained with 1, 2, 4, 8, 16 and all (about 100) labelled images per breed, compared with zero-shot.
''')
nb.code(r'''
E_train = image_embeddings(train["image"]).cpu().numpy()
y_train = np.array(train["label"])
rng = np.random.default_rng(0)
PROBE = {}
for k in [1, 2, 4, 8, 16, None]:
    idx = np.concatenate([rng.permutation(np.where(y_train == c)[0])[:k] for c in range(len(NAMES))]) if k else np.arange(len(y_train))
    lr = LogisticRegression(max_iter=2000, C=10.0).fit(E_train[idx], y_train[idx])
    PROBE[k or "all"] = (lr.predict(E_test.cpu().numpy()) == y_test).astype(float)
    print(f"linear probe, {str(k or 'all'):>3s} images per breed ({len(idx):4d} labels): accuracy {PROBE[k or 'all'].mean():.1%}")

fig, ax = plt.subplots(figsize=(7, 3.8))
ks = [1, 2, 4, 8, 16]
ax.plot(ks + [100], [PROBE[k].mean() * 100 for k in ks] + [PROBE["all"].mean() * 100], "o-", label="linear probe on CLIP embeddings")
ax.axhline(ZS["ensemble of 5 templates"].mean() * 100, color="tab:green", ls="--", label="zero-shot (template ensemble)")
ax.set_xscale("log", base=2); ax.set_xticks(ks + [100], [str(k) for k in ks] + ["all"])
ax.set_xlabel("labelled images per breed"); ax.set_ylabel("accuracy on 600 test images (%)"); ax.grid(alpha=.3); ax.legend()
ax.set_title("37 pet breeds: zero-shot vs a few labels")
plt.tight_layout(); plt.show()
''')
nb.md(r'''
**What the numbers say.**

| labelled images per breed | 1 | 2 | 4 | 8 | 16 | all (~100) | zero-shot |
|---|---|---|---|---|---|---|---|
| linear probe accuracy | 34.3 % | 56.2 % | 61.8 % | 71.2 % | 79.5 % | **89.0 %** | 86.3 % |

**Zero-shot is worth about 30 labelled images per class here.** The probe needs all ~100 images per breed to pass it, and with 16 it is still 7 points behind. This is the opposite of notebook 17's text classification, where a few labels per intent beat zero-shot LLM prompting. Cat and dog breeds are common, well-described categories in CLIP's web-scale training captions. Its text side already knows what a "Bengal" looks like, so a one-shot probe, which has to learn that from one image, is far worse. For niche categories that captions rarely describe (your product SKUs, a defect type), the balance tips toward labels quickly. Measure both.

## 3. A vision-language model on the same task

`qwen2.5vl:3b` gets each image and the list of 37 breeds, and must answer with one. We measure accuracy on 150 test images, latency, and the number of prompt tokens the image adds. Then we repeat on 40 images at three resolutions, to see how tokens, time and accuracy move with the number of pixels.
''')
nb.code(r'''
def b64(img, max_side=None):
    img = img.convert("RGB")
    if max_side:
        img = img.copy(); img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode()

OPTIONS = ", ".join(NAMES)
def vlm_breed(img, max_side=None):
    t0 = time.time()
    r = ollama_client.chat(model=VLM, messages=[{"role": "user", "images": [b64(img, max_side)],
          "content": f"Which breed is the pet in this image? Choose exactly one of: {OPTIONS}. Answer with the breed name only."}],
          options={"temperature": 0, "num_predict": 12})
    text = r["message"]["content"].strip().lower().replace("_", " ")
    hit = [i for i, n in enumerate(NAMES) if n in text]
    return (max(hit, key=lambda i: len(NAMES[i])) if hit else -1), time.time() - t0, r["prompt_eval_count"], text

no_image = ollama_client.chat(model=VLM, messages=[{"role": "user", "content": f"Which breed? Choose exactly one of: {OPTIONS}."}],
                              options={"num_predict": 1})["prompt_eval_count"]
N_VLM = 150
vlm = [vlm_breed(img) for img in test.select(range(N_VLM))["image"]]
vlm_ok = np.array([p == y for (p, *_), y in zip(vlm, y_test[:N_VLM])], dtype=float)
m, lo, hi = bootstrap_ci(vlm_ok)
print(f"{VLM}: accuracy {m:.1%} [{lo:.1%}, {hi:.1%}] on {N_VLM} images | {np.median([v[1] for v in vlm]):.2f} s per image | "
      f"image tokens ≈ {np.median([v[2] for v in vlm]) - no_image:.0f} (prompt {np.median([v[2] for v in vlm]):.0f} vs {no_image} without the image) | unparsed {sum(v[0] == -1 for v in vlm)}")
zs_same = ZS["ensemble of 5 templates"][:N_VLM]; probe_same = PROBE["all"][:N_VLM]
for label, ref in [("CLIP zero-shot", zs_same), ("CLIP linear probe (all labels)", probe_same)]:
    d, lo, hi = paired_bootstrap_diff(ref, vlm_ok)
    print(f"  {VLM} vs {label} ({ref.mean():.1%} on the same images): {d:+.1%} [{lo:+.1%}, {hi:+.1%}]")

RES = {}
sub = test.select(range(40))
for side in [224, 448, 896]:
    out = [vlm_breed(img, side) for img in sub["image"]]
    RES[side] = {"acc": np.mean([p == y for (p, *_), y in zip(out, y_test[:40])]), "sec": np.median([o[1] for o in out]),
                 "tokens": np.median([o[2] for o in out]) - no_image}
    print(f"longest side ≤ {side:3d} px: image tokens ≈ {RES[side]['tokens']:5.0f} | {RES[side]['sec']:.2f} s per image | accuracy {RES[side]['acc']:.0%} on 40")
''')
nb.md(r'''
**What the numbers say.**

- **The VLM is less accurate and far more expensive for this job.** `qwen2.5vl:3b` names the right breed in 72.0 % of 150 images. On the same images CLIP zero-shot gets 83.3 % (VLM −11.3 points [−18.7, −4.7]) and the probe 90.0 % (−18.0 points). The VLM takes 1.45 s per image against 6 ms. A generative model choosing from a list of 37 names in its prompt is a worse classifier than an embedding model built for similarity.
- **An image costs about 1,071 prompt tokens.** The prompt is 1,253 tokens with the image and 182 without. In a provider API that is billed input, and here it is prefill time and context (notebook 23).
- **The token count did not change with resolution.** It stayed at about 1,080 tokens whether the image was capped at 224, 448 or 896 pixels. Qwen2.5-VL supports dynamic resolution, but in this serving path (Ollama's implementation) every image is resized to a fixed pixel budget, so cost is constant. What resolution changed was quality: shrinking the image to 224 px first lost detail that resizing back up cannot recover, and accuracy fell to 55 %. 448 and 896 gave the same 72 %, because the dataset's images are about 500 px and are not enlarged. Check how *your* server preprocesses images: the model card's behaviour and the serving stack's can differ.

**When to use which.** Use CLIP-style embeddings to classify, search, cluster or filter images at scale. Use a VLM when the answer is open-ended (describe, explain, compare, read text, answer a question), when it must follow instructions, or when there is no fixed label set.

## 4. Extracting structured data from documents

A common production use of VLMs is reading documents: invoices, receipts, forms, screenshots. We render 40 synthetic invoices with known fields (vendor, invoice number, date, currency, total), so every answer can be checked exactly. The VLM must return JSON matching a schema: Ollama's `format=` turns on constrained decoding, as in notebook 11. We run it on clean images and on degraded copies (rotated 4°, blurred, saved as JPEG quality 15), as a phone photo or a fax would be.
''')
nb.code(r'''
from pydantic import BaseModel
class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    date: str          # YYYY-MM-DD
    currency: str      # ISO code, e.g. EUR
    total: float

VENDORS = ["Nordic Parts AB", "Casa Verde S.L.", "Blue Harbor Ltd", "Kyoto Tools KK", "Andes Coffee SAC", "Rhein Logistik GmbH",
           "Maple Print Inc", "Porto Textiles Lda"]
ITEMS = ["Steel bolts M8", "Shipping", "Consulting hour", "Paper A4 box", "Coffee beans 1kg", "Pallet rental", "Ink cartridge", "Cotton fabric"]
font = lambda s: ImageFont.load_default(size=s)

def render_invoice(r):
    vendor = r.choice(VENDORS); cur = r.choice(["EUR", "USD", "GBP", "JPY"])
    lines = [(r.choice(ITEMS), r.randint(1, 9), round(r.uniform(3, 400), 2)) for _ in range(r.randint(2, 5))]
    total = round(sum(q * p for _, q, p in lines), 2)
    gold = {"vendor": vendor, "invoice_number": f"INV-{r.randint(10000, 99999)}",
            "date": f"2025-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}", "currency": cur, "total": total}
    img = Image.new("RGB", (800, 560), "white"); d = ImageDraw.Draw(img)
    d.text((40, 30), gold["vendor"], fill="black", font=font(34))
    d.text((40, 90), f"Invoice no. {gold['invoice_number']}", fill="black", font=font(22))
    d.text((520, 90), f"Date: {gold['date']}", fill="black", font=font(22))
    y = 160
    d.text((40, y), "Item", fill="gray", font=font(20)); d.text((420, y), "Qty", fill="gray", font=font(20)); d.text((560, y), "Amount", fill="gray", font=font(20))
    for name, q, p in lines:
        y += 40
        d.text((40, y), name, fill="black", font=font(20)); d.text((430, y), str(q), fill="black", font=font(20))
        d.text((560, y), f"{q * p:,.2f}", fill="black", font=font(20))
    d.line((40, y + 40, 760, y + 40), fill="black", width=2)
    d.text((420, y + 60), f"TOTAL {cur} {total:,.2f}", fill="black", font=font(28))
    return img, gold

def degrade(img):
    img = img.rotate(4, expand=True, fillcolor="white").filter(ImageFilter.GaussianBlur(1.2))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=15); return Image.open(io.BytesIO(buf.getvalue()))

rnd = random.Random(0)
DOCS = [render_invoice(rnd) for _ in range(40)]
def extract(img):
    t0 = time.time()
    r = ollama_client.chat(model=VLM, format=Invoice.model_json_schema(), options={"temperature": 0, "num_predict": 200},
                           messages=[{"role": "user", "images": [b64(img)], "content": "Extract the invoice fields as JSON. Date as YYYY-MM-DD, currency as an ISO code."}])
    try:
        return Invoice.model_validate_json(r["message"]["content"]).model_dump(), time.time() - t0
    except Exception:
        return None, time.time() - t0

def field_ok(pred, gold, k):
    if pred is None:
        return False
    if k == "total":
        return abs(float(pred[k]) - gold[k]) < 0.01
    return str(pred[k]).strip().lower() == str(gold[k]).strip().lower()

EXTRACT = {}
for cond, f in [("clean", lambda x: x), ("degraded", degrade)]:
    outs = [extract(f(img)) for img, _ in DOCS]
    EXTRACT[cond] = {k: np.array([field_ok(p, g, k) for (p, _), (_, g) in zip(outs, DOCS)], dtype=float) for k in Invoice.model_fields}
    EXTRACT[cond]["all fields"] = np.all([EXTRACT[cond][k] for k in Invoice.model_fields], axis=0).astype(float)
    print(f"{cond:9s} " + " | ".join(f"{k} {v.mean():.0%}" for k, v in EXTRACT[cond].items()) + f" | {np.median([t for _, t in outs]):.1f} s per document")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].imshow(DOCS[0][0]); ax[0].set_title("a synthetic invoice"); ax[0].axis("off")
ax[1].imshow(degrade(DOCS[0][0])); ax[1].set_title("the same, degraded"); ax[1].axis("off")
plt.tight_layout(); plt.show()
''')
nb.md(r'''
**What the numbers say.** All five fields of all 40 invoices were extracted exactly, clean or degraded, in about 2.3 s per document. The schema made every reply valid JSON with the right types, and the model read the vendor, number, date, currency and total even through a 4° rotation, blur and heavy JPEG compression.

Be careful with that 100 %. It says our synthetic invoices are **too easy**: printed with a clean font, laid out with one total, no handwriting, no tables spanning pages, no stamps or logos, and labels ("TOTAL EUR") that state the answer. Real document extraction fails on exactly what we did not test:
- several candidate totals (subtotal, tax, total due);
- dates in local formats (03/04/25);
- handwritten corrections;
- multi-page layouts;
- low-contrast scans.

A production evaluation needs a few hundred real documents labelled by hand, scored per field as here. Most of the work is in the edge cases that the first 95 % of documents do not contain. Document AI pipelines often combine a VLM with OCR (Tesseract, PaddleOCR, cloud OCR) and layout models, and validate the extracted fields with business rules: the line items must add up to the total.

## 5. A typographic attack

Following Goh et al. (2021), we write the name of a *wrong* breed on 60 test images, in a white box in a corner, and measure how often each model now answers that breed instead of the true one.
''')
nb.code(r'''
def stamp(img, text):
    img = img.convert("RGB").copy(); w, h = img.size; s = max(18, w // 12)
    d = ImageDraw.Draw(img); box = d.textbbox((0, 0), text, font=font(s))
    d.rectangle((8, 8, 16 + box[2], 16 + box[3]), fill="white"); d.text((12, 10), text, fill="black", font=font(s))
    return img

atk = test.select(range(60))
wrong = [(y + 7) % len(NAMES) for y in y_test[:60]]                   # a fixed wrong breed per image
stamped = [stamp(img, NAMES[w]) for img, w in zip(atk["image"], wrong)]
W = class_embedding(TEMPLATES)
clip_clean = (image_embeddings(atk["image"]) @ W.T).argmax(1).cpu().numpy()
clip_atk = (image_embeddings(stamped) @ W.T).argmax(1).cpu().numpy()
vlm_atk = [vlm_breed(img)[0] for img in stamped]
vlm_clean = [v[0] for v in vlm[:60]]
for name, clean, attacked in [("CLIP zero-shot", clip_clean, clip_atk), (VLM, np.array(vlm_clean), np.array(vlm_atk))]:
    print(f"{name:16s} accuracy clean {np.mean(clean == y_test[:60]):.0%} -> with the written label {np.mean(attacked == y_test[:60]):.0%} "
          f"| answers the written (wrong) breed: {np.mean(attacked == np.array(wrong)):.0%}")
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
for a, i in zip(ax, range(3)):
    a.imshow(stamped[i]); a.axis("off")
    a.set_title(f"true: {NAMES[y_test[i]]}\nCLIP: {NAMES[clip_atk[i]]}", fontsize=8)
plt.tight_layout(); plt.show()
''')
nb.md(r'''
**What the numbers say.** Writing a *wrong* breed name on the image:

| model | accuracy, clean | accuracy, with the written label | answers the written wrong breed |
|---|---|---|---|
| CLIP zero-shot | 87 % | 77 % | 8 % |
| `qwen2.5vl:3b` | 73 % | **17 %** | **72 %** |

- **CLIP is affected, as Goh et al. found**: accuracy fell 10 points and 8 % of answers copied the label. Its image embedding encodes the text in the picture as well as the animal.
- **The VLM was taken over completely.** Shown a dog with "persian" written in a corner, it answered "persian" 72 % of the time. A VLM is an LLM reading its visual tokens, and text it can read in the image becomes part of what it treats as information, often more salient than the animal itself.

This is **prompt injection through pixels**. A screenshot, a scanned document, a product photo or a web page seen by a browsing agent can carry text the model will read and may obey ("ignore the previous instructions..."). Everything notebooks 13, 14 and 22 said about untrusted text applies to untrusted images:
- treat their content as data;
- do not let an agent that reads images hold secrets or dangerous tools;
- test your own pipelines with adversarial images.
''')
nb.md(r'''
## Results

| question | measured (CLIP ViT-B/32 and `qwen2.5vl:3b`, Oxford-IIIT Pet, 37 breeds) | lesson |
|---|---|---|
| CLIP zero-shot | 80.0 % (bare names) → 85.2 % (one template) → **86.3 %** (5 templates); 5.9 ms per image | the class prompts are part of the model |
| linear probe on CLIP embeddings | 34.3 % (1 per breed) … 79.5 % (16) … **89.0 %** (all ~100) | zero-shot is worth ~30 labels per class on common categories |
| VLM as a classifier | 72.0 % (−11.3 points vs CLIP zero-shot), 1.45 s per image, ~1,071 tokens per image | use embeddings to classify, VLMs to describe and read |
| image resolution in this serving path | tokens constant (~1,080); 224 px loses accuracy (55 %) | check your server's image preprocessing |
| invoice extraction with a JSON schema | 100 % of fields, clean and degraded, 2.3 s per document | valid on synthetic invoices only; real documents need a real test set |
| typographic attack | CLIP: 8 % follow the written label; VLM: **72 %** | text in images is prompt injection through pixels |

What this establishes: the two multimodal families solve different problems, and they fail differently. Embeddings are fast and strong for closed sets; VLMs are flexible and expensive, and they read and believe text in the image.

## In production

**Choosing the model.**
- For image search, deduplication, content filtering and closed-set classification, use CLIP or SigLIP embeddings in a vector index (notebook 21). Add a linear probe or a fine-tune when you have labels and the categories are specific to you.
- For captioning, visual question answering, document reading and agents that look at screens, use a VLM: GPT-4o, Claude and Gemini through APIs, or Qwen2.5-VL, InternVL, Llama 3.2 Vision and Gemma 3 locally. Use structured output for anything a program consumes.
- For high-volume documents, a pipeline of OCR + layout analysis + an LLM or VLM, with rule-based validation of the fields, is often cheaper and more controllable than a VLM alone.

**Cost and latency.**
- An image is about a thousand tokens or more. Resize to the smallest resolution that keeps the needed detail, and measure, because both cost and accuracy depend on how the server preprocesses images.
- Cache results for repeated images.
- Batch offline jobs.

**Safety.**
- Treat text inside images as untrusted input.
- Test with adversarial images (written labels, injected instructions).
- Moderate image inputs and outputs (notebook 22's guards have multimodal versions, such as Llama Guard 3 Vision).
- Respect privacy: faces, IDs and documents in images are personal data.

**What to monitor.**
- Accuracy per class or per field on a labelled sample.
- Latency per image and image tokens per request.
- The share of unparsed or schema-invalid replies.
- Drift in image sources (new cameras, new document templates).

**Failure modes.**
- Fine-grained confusion between similar classes.
- Lost detail from downscaling.
- Hallucinated objects or fields.
- Typographic and injected-text attacks.
- An evaluation set cleaner than production data (our invoices).

**Next:** notebook 10 for retrieval-augmented generation over documents, or notebook 14 for agents, where a VLM can read screenshots and documents as tools.
''')
nb.save(notebook_path("24_multimodal.ipynb"))
print("24_multimodal.ipynb written")
