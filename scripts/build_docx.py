import re, subprocess, zipfile, shutil, os
s=open("src.md",encoding="utf8").read()
TITLE=s.split("\n",1)[0].lstrip("# ").strip(); s=s.split("\n",1)[1]
s=re.sub(r"Oct \d+, 2026 · @\S+","[Authors: insert names, affiliations and corresponding-author e-mail]",s)
# ---- display equations (native Word math via pandoc)
ALG='''
Table: Algorithm 1. The battery twin: synchronization at each check-up and Monte Carlo fast-forward

| |
|:--|
| **State.** z = [L_SEI, L_pl, φ₁, φ₂, φ₃, b_OT]: SEI-type loss (calendar + cycling), plating-type loss, three per-cell log-multipliers (set to 0 with zero variance for the population-parameter twin), and the offset between operating-temperature and room-temperature capacity. |
| **Input at step k.** Time t_k and throughput Q_k; observations available: room-temperature capacity (σ = 0.008), relative mid-frequency resistance R_mid/R_mid,0 (σ = 0.15), or operating-temperature cycle capacity (σ = 0.012). |
| **1 Forecast.** Increment each route from (t_{k−1}, Q_{k−1}) to (t_k, Q_k) by Eq. 1 with the current multipliers; linearize the increment in φ; propagate the covariance and add process noise (10⁻⁴ per route, 10⁻³ for b_OT). |
| **2 Update.** For each available channel: capacity observes L_SEI + L_pl (+ b_OT at operating temperature); resistance observes 1 + a·L_SEI + b·L_SEI² (linearized); Kalman gain, mean and covariance update. |
| **3 Knee posterior.** Once L_SEI + L_pl ≥ L*, compare the observed increment over the last 90 days with the model increment with and without the acceleration g; update the knee probability p_knee from its temperature-dependent prior by Bayes' rule. |
| **4 Fast-forward.** Draw N = 300 samples of (log-normal rate multiplier, σ = 0.12; knee outcome ~ Bernoulli(p_knee)); roll each forward along the planned (t, Q) with rates × g once past L*; report mean, 5th and 95th percentiles. |
| **5 Calibrate.** Multiply the half-width of the band by the split-conformal factor fitted on training cells at the same history length (6.3, 8.0, 7.3 for 100, 200, 300 days). |
| **Output.** Route states with covariance, calibrated forecast band, knee probability; for the decision layer, the rollouts themselves (Algorithm 2). |

Table: Algorithm 2. Operating map by risk-adjusted Monte Carlo rollout

| |
|:--|
| **Input.** Temperature T; candidate actions a = (charge rate C ∈ {0.33, 1.0, 1.67}, window ΔV ∈ {2.5–4.2, 3.25–4.09, 3.25–4.2 V}); throughput per day for each action from the training cells that ran it; end-of-life threshold 30% loss; horizon 600 days. |
| **1** For each action, run 400 rollouts of Algorithm 1 step 4 from a fresh cell (L = 0, p_knee = prior(T)). |
| **2** For each rollout record Q_EOL, the throughput delivered before 30% loss (or at the horizon). |
| **3** Score the action by CVaR₂₀ = mean of the lowest 20% of Q_EOL (Eq. 4); also record the mean and the plating share L_pl/(L_SEI + L_pl) at the horizon. |
| **4** Recommend the action with the highest CVaR₂₀ at each temperature. |
| **Validation (offline).** Compare predicted mean Q_EOL with the throughput reached by held-out cells under each action (Spearman ρ over actions with an observed end of life), and the recommendation with the empirically best action. |

'''
EQ=[]
def grab_latex(m):
    EQ.append(re.sub(r"\s+"," ",m.group(1)).strip()); return f"@@EQ{len(EQ)-1}@@"
s=re.sub(r"```latex\n(.*?)\n```",grab_latex,s,flags=re.S)
print("equations from document:",len(EQ))
open("eq.md","w").write("\n\n".join(f"$${e}$$" for e in EQ)); subprocess.run(["pandoc","eq.md","-o","eq.docx"],check=True)
omml=re.findall(r"<m:oMathPara>.*?</m:oMathPara>",zipfile.ZipFile("eq.docx").read("word/document.xml").decode("utf8"),flags=re.S); assert len(omml)==len(EQ)
RPR='<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
def big(o):
    o=re.sub(r"<m:r>(?!<w:rPr>)",f"<m:r>{RPR}",o); o=re.sub(r"<m:r><w:rPr>.*?</w:rPr>",f"<m:r>{RPR}",o)
    return o.replace("<m:oMathPara>","<m:oMathPara><m:oMathParaPr><m:jc m:val=\"center\"/></m:oMathParaPr>",1)
NOB='<w:tblBorders><w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/><w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/><w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/><w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/><w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/><w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/></w:tblBorders>'
def eqblock(i):
    cell=lambda w,jc,inner: f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:spacing w:before="160" w:after="160"/><w:jc w:val="{jc}"/></w:pPr>{inner}</w:p></w:tc>'
    return "\n\n```{=openxml}\n"+f'<w:tbl><w:tblPr><w:tblW w:w="9026" w:type="dxa"/>{NOB}<w:tblLayout w:type="fixed"/><w:tblLook w:val="0000"/></w:tblPr><w:tblGrid><w:gridCol w:w="8226"/><w:gridCol w:w="800"/></w:tblGrid><w:tr>'+cell(8226,"center",big(omml[i]))+cell(800,"right",f'<w:r><w:rPr><w:sz w:val="24"/></w:rPr><w:t>({i+1})</w:t></w:r>')+"</w:tr></w:tbl>"+"\n```\n\n"
# ---- inline math
REP=[("**Calendar route L\\_cal**","**Calendar route $L_{\\mathrm{cal}}$**"),("**Cycling SEI route L\\_cyc**","**Cycling SEI route $L_{\\mathrm{cyc}}$**"),("**Plating route L\\_pl**","**Plating route $L_{\\mathrm{pl}}$**"),
 ("Here t is time, Q is charge throughput, C the charge rate, ΔV the voltage window, and T the aging temperature.","Here $t$ is time, $Q$ is charge throughput, $C$ the charge rate, $\\Delta V$ the voltage window, and $T$ the aging temperature."),
 ("the knee threshold L\\* and acceleration g are","the knee threshold $L^{*}$ and acceleration $g$ are"),("Fitting loss = k tⁿ to each","Fitting $\\mathrm{loss} = k\\,t^{n}$ to each"),("gives n = 0.81 ± 0.08","gives $n = 0.81 \\pm 0.08$"),
 ("(R² = 0.70, n = 48)","($R^{2}$ = 0.70, $n$ = 48)"),("E\\_a/RT.","$E_a/RT$."),("per-cell fits of loss = k tⁿ in log space","per-cell fits of $\\mathrm{loss} = k\\,t^{n}$ in log space"),("and of loss = k√t;","and of $\\mathrm{loss} = k\\sqrt{t}$;"),
 ("joint linear model ln k = a + b·SoC − E\\_a/RT.","joint linear model $\\ln k = a + b\\,\\mathrm{SoC} - E_a/RT$."),("follows t⁰·⁸","follows $t^{0.8}$"),("t⁰·⁸ rather than √t","$t^{0.8}$ rather than $\\sqrt{t}$"),("the t⁰·⁸ exponent","the $t^{0.8}$ exponent"),("a t⁰·⁸ law","a $t^{0.8}$ law"),
 ("follow √t","follow $\\sqrt{t}$"),("t⁰·⁸ rather than √t","$t^{0.8}$ rather than $\\sqrt{t}$"),("follow the √t law","follow the $\\sqrt{t}$ law"),("the √t law would give 0.5","the $\\sqrt{t}$ law would give 0.5"),("Fitting a √t rate constant","Fitting a $\\sqrt{t}$ rate constant"),
 ("Extended Kalman filter over \\[L\\_cal + L\\_cyc, L\\_pl, three log-multipliers, operating-temperature offset\\]","Extended Kalman filter over the state $[L_{\\mathrm{cal}}+L_{\\mathrm{cyc}},\\ L_{\\mathrm{pl}},\\ \\varphi_{1..3},\\ b_{\\mathrm{OT}}]$"),
 ("cyc\\_condition","cyc_condition"),("kit\\_twin\\_reproduction.zip","the repository kit-si-graphite-digital-twin")]
miss=[a for a,b in REP if a not in s]; print("unmatched:",miss)
for a,b in REP: s=s.replace(a,b)
s=re.sub(r"ρ = (−?0\.\d+)",lambda m: f"$\\rho = {m.group(1).replace('−','-')}$",s)

# algorithm-table and CVaR-paragraph symbols -> inline math
ALGREP=[("z = \\[L\\_SEI, L\\_pl, φ₁, φ₂, φ₃, b\\_OT\\]","$z = [L_{\\mathrm{SEI}},\\ L_{\\mathrm{pl}},\\ \\varphi_1,\\ \\varphi_2,\\ \\varphi_3,\\ b_{\\mathrm{OT}}]$"),
 ("Time t\\_k and throughput Q\\_k;","Time $t_k$ and throughput $Q_k$;"),("from (t\\_{k−1}, Q\\_{k−1}) to (t\\_k, Q\\_k)","from $(t_{k-1}, Q_{k-1})$ to $(t_k, Q_k)$"),
 ("10⁻³ for b\\_OT)","$10^{-3}$ for $b_{\\mathrm{OT}}$)"),("Capacity observes L\\_SEI + L\\_pl (+ b\\_OT at operating temperature); resistance observes 1 + a·L\\_SEI + b·L\\_SEI²","Capacity observes $L_{\\mathrm{SEI}} + L_{\\mathrm{pl}}$ (+ $b_{\\mathrm{OT}}$ at operating temperature); resistance observes $1 + a L_{\\mathrm{SEI}} + b L_{\\mathrm{SEI}}^{2}$"),
 ("Once L\\_SEI + L\\_pl ≥ L\\*","Once $L_{\\mathrm{SEI}} + L_{\\mathrm{pl}} \\ge L^{*}$"),("update p\\_knee","update $p_{\\mathrm{knee}}$"),("Bernoulli(p\\_knee)","Bernoulli($p_{\\mathrm{knee}}$)"),("once past L\\*","once past $L^{*}$"),
 ("(L = 0, p\\_knee = prior(T))","($L = 0$, $p_{\\mathrm{knee}}$ = prior($T$))"),("record Q\\_EOL,","record $Q_{\\mathrm{EOL}}$,"),("plating share L\\_pl/(L\\_SEI + L\\_pl)","plating share $L_{\\mathrm{pl}}/(L_{\\mathrm{SEI}} + L_{\\mathrm{pl}})$"),
 ("predicted mean Q\\_EOL","predicted mean $Q_{\\mathrm{EOL}}$"),("where Q\\_EOL is the throughput before 30% loss in one rollout and q\\_0.20(a) its 20th percentile under action a.","where $Q_{\\mathrm{EOL}}$ is the throughput before 30% loss in one rollout and $q_{0.20}(a)$ its 20th percentile under action $a$."),
 ("scores an action a by","scores an action $a$ by"),("gives E\\_a = 10.5","gives $E_a$ = 10.5"),("surface gives √t","surface gives $\\sqrt{t}$"),("rather than √t,","rather than $\\sqrt{t}$,"),("not the √t law","not the $\\sqrt{t}$ law")]
for a,b in ALGREP:
    if a not in s: print("ALGREP unmatched:",a[:50])
    s=s.replace(a,b)

print("left-over escaped underscores:",re.findall(r"\S*\\_\S*",s)[:10])
FIG={"Graphical abstract":"graphical_abstract","Fig. 1 The anode, its nanoscale degradation routes, and the fleet":"fig1_anode_fleet","Fig. 2 Four findings":"fig_findings_H1-H4","Fig. 3 Route-level explanations":"fig_xai_materials","Fig. 4 The twin in three dimensions":"fig7_twin_3d","Fig. 5 Prospective forecasts":"fig_twin_real_cells","Fig. 6 Per-cell twin dashboard":"fig6_cell_dashboard","Fig. 7 Operating map":"fig5_operating_map","Supplementary Fig. S1":"figS1_simulation_benchmark"}
s,n=re.subn(r"&#91;image: ([^\]\\]+)\\?\]",lambda m: f"![](figs/{FIG[m.group(1).strip()]}.png){{width=6.7in}}",s); print("figures placed:",n)
# loose lists, table widths
lines=s.split("\n"); out=[]; islist=lambda l: bool(re.match(r"^(- |\d+\. )",l))
for i,l in enumerate(lines):
    out.append(l)
    if islist(l) and i+1<len(lines) and islist(lines[i+1]): out.append("")
lines,out,i=out,[],0
while i<len(lines):
    if lines[i].startswith("|") and i+1<len(lines) and re.match(r"^\|[\s:|-]+\|$",lines[i+1]):
        j=i
        while j<len(lines) and lines[j].startswith("|"): j+=1
        rows=[[c.strip() for c in re.split(r"(?<!\\)\|",r.strip().strip("|"))] for k,r in enumerate(lines[i:j]) if k!=1]; nc=len(rows[0])
        wid=[3+min(42,max(12,max(len(re.sub(r"[$\\{}]","",r[c])) if c<len(r) else 0 for r in rows))) for c in range(nc)]
        out+=[lines[i],"|"+"|".join("-"*w for w in wid)+"|"]+lines[i+2:j]; i=j
    else: out.append(lines[i]); i+=1
s="\n".join(out)
for k in range(len(EQ)): s=s.replace(f"@@EQ{k}@@",eqblock(k))
open("paper.md","w",encoding="utf8").write(s)
subprocess.run(["pandoc","paper.md","-f","markdown+pipe_tables+tex_math_dollars+raw_attribute+table_captions","-o","raw.docx","--metadata",f"title={TITLE}","--resource-path=."],check=True)
shutil.rmtree("x",ignore_errors=True); zipfile.ZipFile("raw.docx").extractall("x")
d=open("x/word/document.xml",encoding="utf8").read()
B='<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/><w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="999999"/></w:tblBorders>'
def tbl(m):
    t=m.group(0)
    if "tblBorders" in t: return t
    if 'w:tblStyle w:val="Table"' not in t: return t
    for anchor in ("<w:tblLayout","<w:tblCellMar","<w:tblLook"):
        if anchor in t: return t.replace(anchor,B+anchor,1)
    return t.replace("</w:tblPr>",B+"</w:tblPr>")
d=re.sub(r"<m:r>(?!<w:rPr>)",f"<m:r>{RPR}",d)
d,nt=re.subn(r"<w:tblPr>.*?</w:tblPr>",tbl,d,flags=re.S)
def algsize(m):
    t=m.group(0)
    if "Operation" not in t or "Step" not in t: return t
    t=re.sub(r"<w:sz w:val=\"\d+\"/>","",t); t=re.sub(r"<w:szCs w:val=\"\d+\"/>","",t)
    t=re.sub(r"<w:r>(?!<w:rPr>)","<w:r><w:rPr><w:sz w:val=\"21\"/><w:szCs w:val=\"21\"/></w:rPr>",t)
    t=re.sub(r"<w:r><w:rPr>","<w:r><w:rPr><w:sz w:val=\"21\"/><w:szCs w:val=\"21\"/>",t)
    return t
d=re.sub(r"<w:tbl>.*?</w:tbl>",algsize,d,flags=re.S)
d=re.sub(r'<w:pgSz [^>]*/>','<w:pgSz w:w="11906" w:h="16838"/>',d); d=re.sub(r'<w:pgMar [^>]*/>','<w:pgMar w:top="1300" w:right="1150" w:bottom="1300" w:left="1150" w:header="708" w:footer="708" w:gutter="0"/>',d)
if "<w:pgSz" not in d: d=d.replace("</w:body>",'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1300" w:right="1150" w:bottom="1300" w:left="1150" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body>')
open("x/word/document.xml","w",encoding="utf8").write(d)
st=open("x/word/styles.xml",encoding="utf8").read()
st=re.sub(r"<w:rFonts [^>]*Theme[^>]*/>",'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>',st)
st=re.sub(r'<w:color [^>]*/>','<w:color w:val="000000"/>',st)
st=re.sub(r'(<w:style [^>]*w:styleId="Compact"[^>]*>.*?</w:pPr>)',r'\1<w:rPr><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>',st,count=1,flags=re.S)
open("x/word/styles.xml","w",encoding="utf8").write(st)
se=open("x/word/settings.xml",encoding="utf8").read(); open("x/word/settings.xml","w",encoding="utf8").write(re.sub(r"<w:zoom [^>]*/>","",se))
OUT="/mnt/user-data/outputs/Si_graphite_digital_twin_KIT_paper_NML.docx"
if os.path.exists(OUT): os.remove(OUT)
subprocess.run(f"cd x && zip -Xqr '{OUT}' .",shell=True,check=True)
print("tables ruled:",nt,"| native equations:",d.count("<m:oMathPara>"),"display,",d.count("<m:oMath>")-d.count("<m:oMathPara>"),"inline | size",os.path.getsize(OUT))
