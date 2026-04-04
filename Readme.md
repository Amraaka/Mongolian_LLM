# F.CS332 Гүн Сургалт Хичээлийн Курсын Ажлын Удирдамж

**Оюутны курсын ажил – Монгол хэл дээрх LLM загвар сургах, сайжруулах, хэрэглээний түвшинд ашиглах**

---

## 1. Ерөнхий Танилцуулга

Энэхүү курсын ажил нь **гүн сургалтын орчин үеийн хэлний загварууд (Large Language Models – LLM)**-ыг сонгон авч, **монгол хэл дээр сургах, сайжруулах, мөн хэрэглээний түвшинд ашиглах** чадварыг хөгжүүлэх зорилготой.

Оюутнууд **0.8B–4B параметр бүхий нээлттэй LLM загварыг** сонгон авч дараах дараалсан сургалтын үе шатуудыг хэрэгжүүлнэ:

1. Монгол хэлний өгөгдөл ашиглан **pretraining / continued pretraining**
2. **Асуулт-хариултын fine-tuning**
3. **DPO (Direct Preference Optimization)** ашиглан хариултыг сайжруулах
4. Өөрсдийн үүсгэсэн dataset ашиглан **instruction fine-tuning**

Сургалтын өмнөх болон дараах загварын гүйцэтгэлийг харьцуулж **дүгнэлт гаргана**.

### Зорилго
- Том хэлний загваруудын сургалтын pipeline ойлгох
- Монгол хэлний NLP загвар боловсруулах
- Fine-tuning болон alignment аргуудыг турших
- Загварын гүйцэтгэлийг үнэлэх чадвар эзэмших

### Ашиглах Загварууд
Оюутан дараах хэмжээний LLM-үүдээс **нэгийг** сонгож ашиглана.

**Параметрийн хэмжээ:** 0.8B – 4B

**Жишээ загварууд:**
- **Qwen(2.5,3,3.5)-0.8B / 3B**
- **Gemma(2,3)-2B / 4B**
- **Llama-3-1B**
- **Phi-3 Mini**

### Шаардлагатай Хэрэгслүүд

**Програмчлалын хэл**
- Python

**Орчин**
- Google Colab / 317 тоот өрөөний GPU workstation / Kaggle / Server

**Сангууд**
- PyTorch
- HuggingFace Transformers
- Datasets
- TRL (DPO training)
- Accelerate / PEFT (LoRA)
- Evaluate

### Хугацаа ба Үнэлгээ

**Хугацаа:** 3–6 долоо хоног

**Ирүүлэх материал**
- Код (GitHub)
- Тайлан (PDF) – ШУТИС МХТС Эрдэм шинжилгээний өгүүлэл бичих стандартын дагуу
- Загварын checkpoint
- Танилцуулга (7-10 минут)

**Нийт үнэлгээ**

| Хэсэг                    | Хувь  |
|--------------------------|-------|
| Pretraining              | 20%   |
| QA Fine-tuning           | 20%   |
| DPO Training             | 20%   |
| Instruction Fine-tuning  | 20%   |
| Үр дүн ба анализ         | 20%   |

---

## 2. Үндсэн Даалгаврууд

Даалгавруудыг **дарааллаар** гүйцэтгэж, код болон үр дүнг тайланд оруулна.

### 2.1 Монгол хэлний Pretraining (Continued Pretraining)

**Даалгавар**  
Сонгосон LLM загварыг монгол хэлний текст dataset ашиглан дахин сургах.

**Dataset жишээ**
- Wikipedia MN
- Common Crawl MN
- OSCAR Mongolian
- News dataset
- Open subtitles
- [SharePoint dataset](https://mustedumnmy.sharepoint.com/:f:/g/personal/orgilsict_must_edu_mn/IgCZdPYaSyToSK5cO2cMLCw5AU_dkVGnvG6PrNfZvTfaSDE?e=ptqdQW)

**Алхам**
1. Dataset цуглуулах
2. Текст цэвэрлэх
3. Tokenization
4. Training

**Сургалтын арга**
- Continued Pretraining
- Causal Language Modeling

**Үнэлгээ**
- Perplexity
- Loss

**Үр дүн**  
Сургалтын өмнөх ба дараах **perplexity**-г харьцуулах.

### 2.2 Асуулт Хариултын Fine-Tuning

**Даалгавар**  
Загварыг асуултанд хариулдаг болгох.

**Dataset жишээ**
- Mongolian QA dataset
- Translated SQuAD
- Self-generated QA

**Dataset бүтэц**
```json
{
  "question": "...",
  "answer": "..."
}