from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,FrameBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER
from reportlab.lib import colors
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
fontdir=Path('/System/Library/Fonts/Supplemental')
for name,file in [('Times-Roman','Times New Roman.ttf'),('Times-Bold','Times New Roman Bold.ttf'),('Times-Italic','Times New Roman Italic.ttf'),('Times-BoldItalic','Times New Roman Bold Italic.ttf')]:
 if (fontdir/file).exists():
  pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('Times-Roman',normal='Times-Roman',bold='Times-Bold',italic='Times-Italic',boldItalic='Times-BoldItalic')
W,H=612,792; margin=49; gap=14; col=(W-2*margin-gap)/2
out=str(Path(__file__).with_name('proposal.pdf'))
body=ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=11.5,alignment=TA_JUSTIFY,firstLineIndent=10,spaceAfter=2)
abstract=ParagraphStyle('abstract',parent=body,fontName='Times-Bold',fontSize=9.5,leading=10.5,firstLineIndent=10,spaceAfter=8)
head=ParagraphStyle('head',fontName='Times-Roman',fontSize=10,leading=12,alignment=TA_CENTER,spaceBefore=8,spaceAfter=5,keepWithNext=True)
ref=ParagraphStyle('ref',parent=body,fontSize=8,leading=9,firstLineIndent=0,leftIndent=14,spaceAfter=3)
eq=ParagraphStyle('eq',fontName='Times-Italic',fontSize=10,leading=14,alignment=TA_CENTER,spaceBefore=3,spaceAfter=4)
story=[]
def p(s):story.append(Paragraph(s,body))
def h(s):story.append(Paragraph(s,head))
def e(s):story.append(Paragraph(s,eq))
story.append(Paragraph('<i>Abstract</i>—Just image Transformers (JiT) generate images directly in pixel space, but sampling requires repeated evaluations of a large neural network. This project studies whether existing numerical solvers and fixed timestep schedules can reduce generation latency without retraining. Using one pretrained JiT-B/16 checkpoint at 256 × 256 resolution, I compare Euler, Heun, and second-order Adams-Bashforth sampling under controlled network-evaluation budgets. Uniform and nonuniform schedules are evaluated separately to isolate the effect of timestep placement. The primary outcome is the image-quality versus latency frontier. An initial target is a 2× reduction in latency while keeping FID within 10% of the reference sampler; this is an experimental objective, not an assumed result. The study will deliver reproducible sampler implementations, measurements, and visual comparisons.',abstract))
h('I. RESEARCH QUESTION AND HYPOTHESES')
p('<i>How much can pretrained JiT inference be accelerated through numerical solvers and timestep scheduling without substantially degrading generated image quality?</i>')
p('<b>H1 (solver efficiency).</b> A second-order multistep method can provide a better quality-latency trade-off than the reference Heun sampler by reusing previous velocity evaluations instead of computing a new correction at every step. The comparison is made at matched realized network-evaluation budgets and measured latency, rather than nominal solver steps.')
p('<b>H2 (timestep placement).</b> A fixed nonuniform schedule can improve image quality relative to uniform spacing under the same solver and computational budget. Schedule parameters are tuned on separate validation seeds and frozen before final evaluation.')
p('<b>Success criterion.</b> Seek a configuration with latency at most half that of the reproduced reference and FID no greater than 1.10 times its FID, using an identical evaluation protocol. Failure to meet this target is still informative: the final result will quantify the quality cost of faster sampling and identify the best measured operating points.')
h('II. MOTIVATION AND RELATED WORK')
p('JiT predicts clean images directly from noisy pixel-space inputs, avoiding a separately trained latent autoencoder [1]. Its simple architecture still requires many transformer evaluations to turn noise into an image. Numerical sampling therefore offers an inference-only optimization opportunity without changing the checkpoint or architecture.')
p('Euler uses one velocity evaluation per update. Heun usually uses a second evaluation to correct an Euler prediction, improving numerical accuracy at additional cost. A multistep method instead reuses the preceding velocity to obtain a higher-order update with only one new evaluation after startup. Fewer evaluations can reduce latency, although overhead and guidance mean the relationship must be measured.')
p('DPM-Solver++ demonstrates the value of efficient high-order integration for guided diffusion sampling [2]. Align Your Steps shows that timestep placement also affects sampling efficiency [3]. These works motivate the study; the project does not claim a new numerical solver or direct transfer of their reported gains to JiT. Its contribution is a controlled empirical comparison on a pretrained pixel-space transformer.')
h('III. METHODOLOGY')
p('<b>Model and implementation.</b> Use the public JiT-B/16 checkpoint and PyTorch implementation [4] for class-conditional ImageNet generation at 256 × 256 pixels. Experiments run on an NVIDIA RTX 5080, subject to initial compatibility, throughput, and memory checks. Model weights, architecture, and training remain unchanged. Record the checkpoint, code revision, software versions, precision, batch size, guidance settings, labels, and seeds.')
p('<b>Velocity construction.</b> For noisy state z at time t and predicted clean image x<sub>pred</sub>, follow the repository velocity parameterization:')
e('v(z, t) = (x<sub>pred</sub>(z, t) − z) / max(1 − t, ε)')
p('The endpoint convention and denominator safeguard are fixed across methods. The original repository sampler is retained as a separate reference with its exact settings and measured evaluation count.')
p('<b>Euler.</b> With h<sub>n</sub> = t<sub>n+1</sub> − t<sub>n</sub>, the first-order baseline is')
e('z<sub>n+1</sub> = z<sub>n</sub> + h<sub>n</sub>v<sub>n</sub>')
p('<b>Heun.</b> Form z* = z<sub>n</sub> + h<sub>n</sub>v<sub>n</sub>, evaluate v* at (z*, t<sub>n+1</sub>), and correct:')
e('z<sub>n+1</sub> = z<sub>n</sub> + (h<sub>n</sub>/2)(v<sub>n</sub> + v*)')
p('Preserve and document the reference implementation’s final Euler step so that endpoint handling is not a hidden source of differences.')
p('<b>Adams-Bashforth 2.</b> Start with an Euler update, then reuse the previous velocity. For r<sub>n</sub> = h<sub>n</sub>/h<sub>n−1</sub>, use the variable-step update')
e('z<sub>n+1</sub> = z<sub>n</sub> + h<sub>n</sub>[(1 + r<sub>n</sub>/2)v<sub>n</sub> − (r<sub>n</sub>/2)v<sub>n−1</sub>]')
p('For constant steps this reduces to the standard second-order Adams-Bashforth method. Validate the implementation on simple test ODEs, check startup and variable-step coefficients, and inspect stability under coarse schedules. Higher numerical order does not guarantee better generated images.')
p('<b>Schedules and comparisons.</b> Compare uniform spacing with a small family of parameterized, monotone nonuniform grids. Tune on validation seeds and keep the selected grid fixed across test images. First compare solvers on uniform grids; then compare schedules with the solver fixed; finally compare the best combination with the unmodified JiT reference. Image-specific adaptive stepping is outside scope.')
h('IV. EVALUATION, FEASIBILITY, AND EXPECTED CONTRIBUTION')
p('<b>Fair budgets.</b> Evaluate nominal budgets of 10, 20, 30, 50, and 100 network passes. Count actual transformer evaluations, including both classifier-free guidance branches, solver substages, and startup. Log guided velocity evaluations separately. If an exact budget is infeasible for a method, use the nearest feasible configuration at or below it and report its realized count. Hold labels, initial noise, guidance, and evaluation settings constant across methods.')
p('<b>Image quality.</b> Measure Fréchet Inception Distance (FID) against fixed ImageNet reference statistics. Use small generated sets only to screen configurations, then evaluate finalists with the same larger sample count and preprocessing. Set that count after measuring pilot throughput and report it explicitly. Repeat finalist evaluations where feasible and report variability; small differences in noisy FID estimates are not evidence of superiority. Schedules are never tuned on final evaluation seeds.')
p('<b>Runtime and memory.</b> Measure synchronized wall-clock generation time after warm-up, at fixed precision and batch size, with repeated runs. Report median latency, its variability, actual network passes, and peak GPU memory. Exclude model loading and disk I/O from generation timing and state this boundary. Network-pass reductions are not treated as measured speedups.')
story.append(FrameBreak())
p('<b>Analysis.</b> Plot FID against measured latency and report Pareto-efficient configurations: those for which no alternative is both faster and at least as good in FID. Compare paired images generated from identical noise and labels, and selected intermediate clean-image predictions. These visualizations explain differences but do not replace distribution-level evaluation.')
p('<b>Feasibility.</b> Limit the study to one pretrained model and inference-only changes. Pilot runs establish throughput and the final compute budget before large-scale generation. If resources are insufficient, prioritize solver comparisons and reduce schedule variants and budget levels. Preserve the reference and equal-sample final comparisons; do not assume an unmeasured GPU-hour estimate.')
p('<b>Expected contribution.</b> A reproducible evaluation of existing numerical sampling methods for JiT, a recommended configuration for different computational budgets, and a measured account of the quality-latency trade-off. Deliverables include sampler code, experiment settings, quantitative benchmarks, paired image grids, and a final technical report. Negative results and instability at very low budgets will be reported alongside successful configurations.')
h('V. PROPOSED SCHEDULE')
p('<i>Weeks 1–2:</i> set up JiT, check hardware, reproduce reference samples, and benchmark throughput. <i>3–4:</i> implement and validate Euler, Heun, and multistep sampling. <i>5–6:</i> compare solvers at fixed network-pass budgets. <i>7–8:</i> implement and tune nonuniform schedules on validation seeds. <i>9–10:</i> evaluate selected configurations using larger equal-sized image sets. <i>11:</i> analyze quality-latency frontiers and visual comparisons. <i>12–13:</i> prepare reproducible code, final report, and presentation.')
h('REFERENCES')
for s in ['[1] T. Li and K. He, “Back to Basics: Let Denoising Generative Models Denoise,” arXiv:2511.13720, 2025.','[2] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, “DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models,” arXiv:2211.01095, 2022.','[3] A. Sabour, S. Fidler, and K. Kreis, “Align Your Steps: Optimizing Sampling Schedules in Diffusion Models,” in Proc. ICML, 2024.','[4] T. Li, “JiT: PyTorch implementation,” github.com/LTH14/JiT. Accessed Nov. 5, 2026.']:
 story.append(Paragraph(s,ref))
def cover(c,d):
 c.setFillColor(colors.black)
 c.setFont('Times-Roman',23)
 c.drawCentredString(306,738,'Accelerating Just Image Transformers')
 c.drawCentredString(306,711,'Through Efficient Numerical Sampling')
 c.setFont('Times-Roman',11);c.drawCentredString(306,690,'Training-Free Inference Optimization with Existing Numerical Solvers')
 c.setFont('Times-Roman',10);c.drawCentredString(306,663,'Iason-Daniil Zakynthinos')
 c.setFont('Times-Roman',8.5);c.drawCentredString(306,647,'School of ECE, NTUA · Advanced Topics in Artificial Intelligence · 2026–2027')
first=[Frame(margin,48,col,580,id='l1',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),Frame(margin+col+gap,48,col,580,id='r1',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
rest=[Frame(margin,48,col,696,id='l',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),Frame(margin+col+gap,48,col,696,id='r',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
doc=BaseDocTemplate(out,pagesize=(W,H),title='Accelerating Just Image Transformers Through Efficient Numerical Sampling',author='Iason-Daniil Zakynthinos')
doc.addPageTemplates([PageTemplate(id='First',frames=first,onPage=cover,autoNextPageTemplate='Body'),PageTemplate(id='Body',frames=rest)])
doc.build(story)
print(out)
