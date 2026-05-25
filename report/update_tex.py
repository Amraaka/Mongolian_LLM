import os

file_path = "/home/gantumur/DL/Mongolian_LLM/report/report.tex"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove babel
content = content.replace("\\usepackage[mongolian]{babel}", "% \\usepackage[mongolian]{babel}")

# 2. Update Abstract
old_abstract = """\t\t\t\\textbf{\\textit{Хураангуй:}} \\textbf{Энэхүү курсын ажлын хүрээнд Qwen3.5-2B-Base суурь загварыг Монгол хэлний онцлогт тохируулан Continued Pretraining, QA SFT, DPO, болон Instruction tuning гэсэн 4 үе шаттайгаар сургасан үр дүнг тайлагнав. Тооцооллын нөөцийн хязгаарлалтад нийцүүлэн Unsloth сан болон 4-bit QLoRA ($r=16, \\alpha=16$) аргачлалыг 8-bit AdamW оптимизатортай хослуулан ашигласан. Сургалтын 1-р шатанд 60,000 өгөгдөл дээр 25,000 алхам урьдчилан сургалт хийж, загварын хэл зүйн бүтцийг ойлгох чадварыг сайжруулан Perplexity утгыг ~3.5 түвшинд тогтворжуулсан. 2-р шатанд 98,000 асуулт хариултын өгөгдлийг ChatML форматаар 3-4 epoch сургаснаар F1 оноо 0.7537-д хүрсэн. 3-р шатанд (DPO, $\\beta=0.1$) 7,496 хос өгөгдөл дээр 1,500 алхам сургахад баталгаажуулалтын алдаа 0.0353-д хүрч, хүний хариултыг ялгах Reward accuracy 99.04\\% болсон. Эцсийн шатанд Tavily API ашигласан 900 өгөгдлөөр функц дуудах (function-calling) чадварыг суулгасан. Туршилтаар QLoRA нь VRAM зарцуулалтыг үр дүнтэй бууруулсан хэдий ч квантчлалын нөлөөгөөр үүсэх мэдээллийн алдагдал, DPO шатны хэт өндөр урамшууллын онооноос шалтгаалсан overfitting үүсэх эрсдэл зэрэг техникийн хязгаарлалтууд үүссэн болно.}"""

new_abstract = """\t\t\t\\textbf{\\textit{Хураангуй:}} \\textbf{Энэхүү курсын ажлын хүрээнд Qwen3.5-2B-Base суурь загварыг Монгол хэлний онцлогт тохируулан Continued Pretraining, QA SFT, DPO, болон Instruction tuning гэсэн дөрвөн үе шаттайгаар сургасан үр дүнг тайлагнав. Тооцооллын нөөцийн хязгаарлалтад нийцүүлэн Unsloth сан болон QLoRA аргачлалыг AdamW оптимизатортай хослуулан ашигласан. Сургалтын эхний шатанд их хэмжээний өгөгдөл дээр урьдчилан сургалт хийж, загварын хэл зүйн бүтцийг ойлгох чадварыг сайжруулан Perplexity утгыг мэдэгдэхүйц багасгаж тогтворжуулсан. Хоёрдугаар шатанд асуулт хариултын өгөгдлийг ChatML форматаар сургаснаар F1 оноо өндөр түвшинд хүрсэн. Гуравдугаар шатанд (DPO) хос өгөгдөл дээр сургахад баталгаажуулалтын алдаа багасаж, хүний хариултыг ялгах чадвар (Reward accuracy) маш өндөр хувьтай болсон. Эцсийн шатанд Tavily API ашиглан тусгайлан бэлтгэсэн өгөгдлөөр функц дуудах (function-calling) чадварыг суулгасан. Туршилтаар QLoRA нь график санах ойн зарцуулалтыг үр дүнтэй бууруулсан хэдий ч квантчлалын нөлөөгөөр үүсэх мэдээллийн алдагдал, DPO шатны хэт өндөр урамшууллын онооноос шалтгаалсан хэт цээжлэх (overfitting) эрсдэл зэрэг техникийн хязгаарлалтууд үүссэн болно.}"""

if old_abstract in content:
    content = content.replace(old_abstract, new_abstract)
else:
    print("Warning: Abstract not matched exactly. Trying robust replace.")
    # More robust logic for abstract
    start_str = "\\textbf{\\textit{Хураангуй:}}"
    end_str = "\\vspace{0.3cm}"
    s_idx = content.find(start_str)
    e_idx = content.find(end_str)
    if s_idx != -1 and e_idx != -1:
        content = content[:s_idx] + new_abstract + "\n\t\t\t\n\t\t\t" + content[e_idx:]


# 3. Update Intro
intro_old = "үржигдэхүүнийг урьдчилан таамаглах зарчмаар ажилладаг. \n\t\n\tTransformers архитектурын"
intro_new = "үржигдэхүүнийг урьдчилан таамаглах зарчмаар ажилладаг. \n\t\n\tЭнэхүү судалгааны ажилд суурь болгон ашигласан Qwen3.5-2B-Base загвар нь 2 тэрбум параметртэй, олон хэлний чадавх сайтай, нээлттэй эхийн том хэлний загвар бөгөөд текст үүсгэх, ойлгох, болон төрөл бүрийн хэл шинжлэлийн даалгавруудыг өндөр нарийвчлалтай гүйцэтгэх чадвартай юм \\cite{qwen}. Уг загварыг суурь болгон ашиглах нь тооцооллын нөөцийн хязгаарлагдмал байдалд тохирохоос гадна Монгол хэлний онцлогт нийцүүлэн нэмэлт сургалт хийхэд оновчтой сонголт болдог.\n\t\n\tTransformers архитектурын"
content = content.replace(intro_old, intro_new)


# 4. Update Methodology - QLoRA
qlora_old = "Бүх 4 шатны сургалтад Qwen3.5-2B-Base загвар дээр Unsloth сан болон QLoRA (4-bit) аргыг ашигласан. LoRA-ийн тохиргоонд $r=16, \\alpha=16$, dropout $0$ байх ба attention болон MLP давхаргуудыг сургасан. Бүх шатанд 8-bit AdamW оптимизатор болон cosine learning rate хуваарийг ашигласан бөгөөд gradient checkpointing-ийг санах ой хэмнэх зорилгоор идэвхжүүлсэн."
qlora_new = "Бүх 4 шатны сургалтад Qwen3.5-2B-Base загвар дээр Unsloth сан \\cite{unsloth} болон QLoRA (4-bit) аргыг \\cite{qlora} ашигласан. LoRA-ийн тохиргоонд $r=16, \\alpha=16$, dropout $0$ байх ба attention болон MLP давхаргуудыг сургасан. QLoRA нь загварын параметрүүдийг квантчлалаар шахаж, зөвхөн бага хэмжээст нэмэлт матрицуудыг сургаснаар график санах ойн (VRAM) хэрэгцээг эрс багасган, тооцооллын үр ашгийг нэмэгдүүлдэг шийдэл юм \\cite{qlora}. Бүх шатанд 8-bit AdamW оптимизатор болон cosine learning rate хуваарийг ашигласан бөгөөд gradient checkpointing-ийг санах ой хэмнэх зорилгоор идэвхжүүлсэн."
content = content.replace(qlora_old, qlora_new)


# 5. Update Methodology - DPO
dpo_old = "Загварын хариултын чанарыг сайжруулахын тулд DPO (Direct Preference Optimization) аргаар сургасан. DPO-ийн $\\beta=0.1$ параметр ашигласан ба сургалтын хурд $5 \\times 10^{-6}$, багцын хэмжээ 16 байхаар тохируулсан."
dpo_new = "Загварын хариултын чанарыг сайжруулахын тулд DPO (Direct Preference Optimization) аргаар \\cite{dpo} сургасан. DPO нь хүн шиг сэтгэх, зөв болон буруу хариултыг ялгах чадварыг урамшууллын загвар (reward model) ашиглахгүйгээр шууд оновчилж, хүний хүлээлтэд нийцсэн чанартай хариулт гаргах үйл явцыг тогтворжуулдаг аргачлал юм \\cite{dpo}. DPO-ийн $\\beta=0.1$ параметр ашигласан ба сургалтын хурд $5 \\times 10^{-6}$, багцын хэмжээ 16 байхаар тохируулсан."
content = content.replace(dpo_old, dpo_new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updates applied.")
