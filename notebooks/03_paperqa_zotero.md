# Paper QA for Graphene Oxide

We import a Zotero group library containing the following articles:
- 10.1021/tx400385x
- 10.1021/tx400385x
- 10.1016/j.toxlet.2010.11.016
- 10.1038/cdd.2010.11
- 10.1016/j.biomaterials.2012.07.040
- 10.1016/j.biomaterials.2012.02.021
- 10.1021/nl202515a
- 10.1007/s11671-010-9751-6
- 10.1002/smll.201201546
- 10.1021/nn101097v
- 10.1021/nn202699t
- 10.6023/A20060216
- 10.3390/ijms221910578
- 10.1021/am300253c
- 10.1002/smll.201102743
- 10.1021/acsomega.2c03171
- 10.1016/j.envpol.2017.12.034
- 10.1021/la203607w
- 10.1021/nn1007176
- 10.1016/j.biomaterials.2012.11.001
- 10.1021/am300253c
- 10.1016/j.biomaterials.2012.05.064
- 10.1039/C2JM31396K
- 10.1016/j.biomaterials.2013.01.001

and use the `paperqa` library (a wrapper for `langchain` to perform question answering on journal articles).

We define the prompt style to instruct the model to return its answer as a bullet point list, and then ask the following questions:


```python
import os
# Necessary to import OpenAI
with open('assets/openai_api_key', 'r') as f:
    os.environ['OPENAI_API_KEY'] = f.read()
with open('assets/zotero_api_key', 'r') as f:
    os.environ['ZOTERO_API_KEY'] = f.read()
with open('assets/zotero_user_id', 'r') as f:
    os.environ['ZOTERO_USER_ID'] = f.read()

from pathlib import Path
import paperqa 
from langchain.chat_models import ChatOpenAI
import pickle
import nest_asyncio
from IPython.display import display, Markdown, Latex
nest_asyncio.apply()
import json
from paperqa.contrib import ZoteroDB


```


```python
docs_pickle = '../data/paperqa/docs_zotero.pickle'
model_name = 'gpt-4'
```


```python
from paperqa import Docs, Answer, PromptCollection
from langchain.prompts import PromptTemplate
my_qaprompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer the question '{question}' as a bullet point list."
    "For each bullet point, include the citation (the paper title) used to give an answer."
    "Use the context below. "
    "If there is insufficient context, answer \"NA\" "
    "\n\n"
    "Context: {context}\n\n")
prompts=PromptCollection(qa=my_qaprompt)

docs = paperqa.Docs(llm=ChatOpenAI(temperature=0.0, model_name=model_name),
                prompts=prompts)
zotero = ZoteroDB(library_type="group")  # "group" if group library
doc_number = 0
print('Looking for papers with attached PDFs in zotero group library')
for item in zotero.iterate():
    print(f'{doc_number}: adding PDF for {item.title}')
    doc_number+=1
    docs.add(item.pdf, docname=item.title)
with open(docs_pickle, "wb") as f:
    pickle.dump(docs, f)
docs.memory = True
k = 1

```

    Looking for papers with attached PDFs in zotero group library
    0: adding PDF for Understanding the pH-Dependent Behavior of Graphene Oxide Aqueous Solutions: A Comparative Experimental and Molecular Dynamics Simulation Study
    1: adding PDF for The use of a glucose-reduced graphene oxide suspension for photothermal cancer therapy
    2: adding PDF for The role of the lateral dimension of graphene oxide in the regulation of cellular responses
    3: adding PDF for The effects of graphene oxide nanosheets localized on F-actin filaments on cell-cycle alterations
    4: adding PDF for Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface
    5: adding PDF for Size-dependent genotoxicity of graphene nanoplatelets in human stem cells
    6: adding PDF for Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets
    7: adding PDF for Simultaneous induction of autophagy and toll-like receptor signaling pathways by graphene oxide
    8: adding PDF for Nanotoxicity of Graphene and Graphene Oxide
    9: adding PDF for Minimizing Oxidation and Stable Nanoscale Dispersion Improves the Biocompatibility of Graphene in the Lung
    10: adding PDF for Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy
    11: adding PDF for In vivo biodistribution and toxicology of functionalized nano-graphene oxide in mice after oral and intraperitoneal administration
    12: adding PDF for In vitro toxicity evaluation of graphene oxide on A549 cells
    13: adding PDF for Graphene-Based Antibacterial Paper
    14: adding PDF for Graphene Oxide: A Nonspecific Enhancer of Cellular Growth
    15: adding PDF for Graphene Oxide, But Not Fullerenes, Targets Immunoproteasomes and Suppresses Antigen Presentation by Dendritic Cells
    16: adding PDF for Dependence of Graphene Oxide (GO) Toxicity on Oxidation Level, Elemental Composition, and Size
    17: adding PDF for Biocompatibility of Graphene Oxide
    18: adding PDF for Cytotoxicity Effects of Graphene and Single-Wall Carbon Nanotubes in Neural Phaeochromocytoma-Derived PC12 Cells
    19: adding PDF for Cell Death
    20: adding PDF for An Update on Graphene Oxide: Applications and Toxicity
    21: adding PDF for A mechanism study on toxicity of graphene oxide to Daphnia magna: Direct link between bioaccumulation and oxidative stress


## QUESTION 1
> 'Can the cell internalize graphene oxide?'


```python
q = 'Can the cell internalize graphene oxide?'
answer = docs.query(q,k=10,max_sources=10)
display(Markdown(f'**LLM ANSWER**:\n\n {answer.answer}'))

```


**LLM ANSWER**:

 - Cells can internalize protein-coated graphene oxide nanosheets (PCGO), with the mechanism of uptake being size-dependent (Source: "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets").
- Smaller nanosheets are primarily taken up through clathrin-mediated endocytosis, while larger nanosheets are more likely to be internalized through phagocytosis (Source: "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets").
- The internalization of Au-GO (gold-graphene oxide) into Ca Ski cells has been evidenced, with the process increasing significantly at an incubation time of 4 hours, and reaching a maximum at 6 hours (Source: "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy").
- The cellular entry of Au-GO is primarily through energy-dependent, clathrin-mediated endocytosis (Source: "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy").
- PCGO adheres to the cell surface and is then internalized, based on observations of PCGO attaching to the surface of a model cell line, C2C12 (Source: "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets").
- Graphene oxide (GO) has been reported to be an efficient intracellular transporter for drug and gene delivery, indicating its ability to enter cells (Source: "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets").


## QUESTION 2
> 'How does size impact graphene uptake by the cell? On which cells?'


```python
q2 = 'How does size impact graphene uptake by the cell? On which cells?'
answer2 = docs.query(q2,k=10,max_sources=10)
display(Markdown(f'**LLM ANSWER**:\n\n {answer2.answer}'))

```


**LLM ANSWER**:

 - The size of graphene oxide (GO) sheets does not significantly affect the amount internalized by macrophages, but it does influence the intracellular location and the inflammatory response induced. Only two types of phagocytes were found capable of internalizing GO. ("The role of the lateral dimension of graphene oxide in the regulation of cellular responses pages 1-1")
- The size of protein-coated graphene oxide nanosheets (PCGO) influences their uptake by cells. Smaller nanosheets are primarily taken up through a process called clathrin-mediated endocytosis, while larger nanosheets are more likely to be taken up through phagocytosis. ("Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets pages 1-1")
- The uptake of graphene by cells is influenced by its size, with different cellular responses observed based on the lateral dimensions of graphene oxide. In macrophages, particle size affects the cellular response. ("The role of the lateral dimension of graphene oxide in the regulation of cellular responses pages 8-9")
- The effects of graphene oxide (GO) lateral dimension, from nano to micro, on cellular responses such as cellular uptake, internalization mechanisms, intracellular trafficking, and inflammation response were investigated in two types of macrophages and four non-phagocytic cells. ("The role of the lateral dimension of graphene oxide in the regulation of cellular responses pages 2-2")
- The size of reduced graphene oxide nanoplatelets (rGONPs) significantly impacts their cytotoxic and genotoxic effects on human mesenchymal stem cells (hMSCs) isolated from umbilical cord blood. ("Size-dependent genotoxicity of graphene nanoplatelets in human stem cells pages 1-1")
- The size of graphene oxide nanosheets (PCGOs) impacts their uptake by C2C12 cells, a cell line known to possess phagocytic activity. Larger nanosheets are predominantly taken up through phagocytosis, while smaller nanosheets primarily enter cells through clathrin-mediated endocytosis. ("Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets pages 6-7")


## QUESTION 3
> 'How does surface functionalization affect graphene oxide interaction with the cell?'


```python
q3 = 'How does surface functionalization affect graphene oxide interaction with the cell?'
answer3 = docs.query(q3,k=10,max_sources=10)
display(Markdown(f'**LLM ANSWER**:\n\n {answer3.answer}'))

```


**LLM ANSWER**:

 - Surface functionalization of graphene oxide (GO) significantly impacts its interaction with cells. Unmodified GO in serum absorbs a large amount of protein molecules, which can promote macrophage phagocytosis. However, surface-modified GO (GO-NH 2, GO-PAA, GO-PEG) significantly reduces protein adsorption and interaction with macrophages. The surface properties of GO can also affect its interaction with the cell membrane. Surface modifications can increase or decrease the interaction of GO with the cell membrane. For instance, PEG and PAA modifications can effectively reduce membrane abnormalities, membrane integrity damage, and membrane potential depolarization caused by GO. ("Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface" pages 7-8)
  
- The primary mechanism of Au-GO (gold-graphene oxide) uptake is clathrin-mediated endocytosis, which is energy-dependent. The same experiment was conducted with GO (graphene oxide) alone, and the results were consistent, indicating that the presence of gold nanoparticles (Au NPs) did not alter the cellular uptake mechanism. ("Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy" pages 6-6)

- The surface of GO can be chemically modified to regulate its properties and design specific functionalities. The oxygen-containing functional groups on the surface of GO can be used to control its surface chemistry. These modifications can improve its dispersion, colloidal stability, and biocompatibility under physiological conditions. ("Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface" pages 1-2)

- The size of GO sheets affected their internalization by different cell types, with only two types of phagocytes capable of internalizing GO. The study also found that GO sheets of different sizes (2mm and 350 nm) were equally taken up by macrophages, suggesting that size does not affect uptake. The study also found that micro-sized GO induced stronger inflammation responses than nano-sized GO. ("The role of the lateral dimension of graphene oxide in the regulation of cellular responses" pages 1-1)

- Protein-coated graphene oxide (PCGO) adheres to the cell surface and is then internalized. This adhesion is hypothesized to be due to several factors, including the similar curvature between the nanosheets and plasma membrane, multiple binding forces such as electrostatic and hydrophobic interactions, and potential specific ligand-receptor interactions. ("Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets" pages 3-4)

- Smaller nanosheets primarily enter cells through clathrin-mediated endocytosis, while larger ones are more likely to be taken up through phagocytosis. ("Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets" pages 1-1)


## QUESTION 4

> 'Which functional groups in the surface of graphene oxide lead to increased uptake by the cell?'


```python
q4 = 'Which functional groups in the surface of graphene oxide lead to increased uptake by the cell?'
answer4 = docs.query(q4,k=10,max_sources=10)
display(Markdown(f'**LLM ANSWER**:\n\n {answer4.answer}'))

```


**LLM ANSWER**:

 Based on the provided context, the functional groups in the surface of graphene oxide that lead to increased uptake by the cell include:

- Oxygen-containing functional groups, which facilitate interaction with cell membranes ("Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface" pages 7-8).
- Carbon free radicals, present in highly hydroxylated GO, which can induce stronger lipid peroxidation and cell membrane damage ("Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface" pages 7-8).
- Carboxyl (C-COOH) and carbon-carbon groups (C-C and C=C), which contribute to higher molecular toxicity related to DNA damage stress, protein stress, and chemical stress ("Dependence of Graphene Oxide (GO) Toxicity on Oxidation Level, Elemental Composition, and Size" pages 11-12).

The context does not provide sufficient information to identify additional functional groups on the surface of graphene oxide that could increase cellular uptake.


## CONTEXTS

### Question 1


```python
display(Markdown("\n\n__________\n\n".join([":\n\n".join([answer.dict()['contexts'][i]['text']['doc']['citation'], answer.dict()['contexts'][i]['text']['text'], ]) for i in range(len(answer.dict()['contexts']))])))
```


Mu, Qingxin, et al. "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets." Department of Chemical Biology & Therapeutics, St. Jude Children’s Research Hospital, 2023.:

Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide
Nanosheets
Qingxin Mu,†Gaoxing Su,†,‡Liwen Li,†,‡Ben O. Gilbertson,§Lam H. Yu,§Qiu Zhang,‡Ya-Ping Sun,⊥
and Bing Yan *,†,‡
†Department of Chemical Biology & Therapeutics, St. Jude Children ’s Research Hospital, Memphis, Tennessee, 38105, United States
‡School of Chemistry and Chemical Engineering, Shandong University, Jinan, China, 250100
§Department of Physics, University of Memphis, Memphis, Tennessee, 38152, United States
⊥Department of Chemistry and Laboratory for Emerging Materials and Technology Hunter Hall, Clemson University, Clemson,
South Carolina, 29634-0973, United States
*SSupporting Information
ABSTRACT: As an emerging applied material, graphene has shown
tremendous application potential in many fields, including bio-medicine. However, the biological behavior of these nanosheets,especially their interactions with cells, is not well understood. Here,we report our findings about the cell surface adhesion, subcellularlocations, and size-dependent uptake mechanisms of protein-coatedgraphene oxide nanosheets (PCGO). Small nanosheets enter cellsmainly through clathrin-mediated endocytosis, and the increase ofgraphene size enhances phagocytotic uptake of the nanosheets. Thesefindings will facilitate biomedical and toxicologic studies of graphenesand provide fundamental understanding of interactions at theinterface of two-dimensional nanostructures and biological systems.
KEYWORDS: graphene oxide nanosheets, protein binding, cell uptake, clathrin-mediated endocytosis, phagocytosis, size dependence
■INTRODUCTION
Graphene, a hexagonal carbon nanostructure similar to carbon
nanotubes and fullerene, has unique electronic, thermal, andmechanical properties, showing tremendous application
potential in fields such as electronics and biomedicine.
1,2
Graphene oxide (GO), which is oxidized graphite withenhanced aqueous solubility, has been proven to be an efficient
biosensor,
3drug carrier,4,5and photothermal cancer-killing
agent.6,7GO nanosheets are able to enter cells which renders
them to become promising candidates for intracellular deliveryof drugs and cellular imaging. However, the mechanisms of how
the emerging nanostrucutures interface with biological systems
are still largely unknown. In particular, a fundamental
understanding of its ability to penetrate cell membranes and
other biological barriers is still lacking. For instance, whether
the nanosheets parallelly attach onto cell surface or vertically
insert into cell membrane? By what manner they enter cells?
Such cellular uptake properties of nanoparticles may affect cell
signaling, proliferation, differentiation, and nanoparticle ex-cretion.
8−10Cellular uptake of nanoparticles with other shapes
has been studied.11We and other researchers previously
discovered endosomal leakage and nuclear translocation of
multiwalled carbon nanotubes.9,12However, the behavior of
sheet-shaped nanostructures has not been reported. Further-
more, 

__________

Huang, Jie, et al. "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy." Wiley Online Library, vol. 2577, 2012, www.wileyonlinelibrary.com. Accessed 14 July 2023.:

Au-GO inside the cell from different spots of the dark-ﬁ  eld microscopic image. Two prominent peaks at 1330 
and 1600 cm 
 − 1  appear, which can be assigned undoubtedly to the D and G bands of graphene, [  8  ]  respectively, thereby 
indicating the internalization of Au-GO into Ca Ski cells. It is interesting to note that the SERS intensity of GO varies dramatically at different chosen spots inside the cell, which implies inhomogeneous distribution of GO inside the cell. This observation highlights the good spatial resolution of the SERS technique for study of intracellular events, which is similar to the widely used ﬂ  uorescence method. 
[  28  ]  This 
can be conﬁ  rmed by checking the SERS spectra of many 
cells incubated with Au-GO. To compare the Raman spectra of GO (Figure  3 ) and Au-GO (Figure  4 ) inside cells, the weight concentrations of GO and Au-GO were adjusted to be the same in terms of GO (1.6  μ g mL 
 − 1 ). Clearly, the 
presence of Au NPs on the GO sheet plays an important role in the observation of enhanced Raman spectra of GO in live cells.
         Figure  3 .     a) Bright and b) dark-ﬁ  eld microscopic images of Ca Ski cells incubated with GO. c) Raman spectra of the different spots marked in (b) in 
the Ca Ski cell.  
(a) 
(b)500 1000 1500 2000 2500 3000
Raman shift (cm-1)1
2
3
4
550 CPSRaman Intensity
(c) 
     Figure  4 .     a) Bright and dark-ﬁ  eld microscopic images of Ca Ski cells incubated with Au-GO for 4 h. c) Raman spectra of  the different spots marked 
in (b) in the Ca Ski cell.  
(a) 
500 1000 1500 2000 2500 30005431Intensity (a.u.)
Raman shift (cm-1)100 cps
1330
1600
2
(c) (b)
12
3
45
small  2012, 8, No. 16, 2577–2584
 16136829, 2012, 16, Downloaded from https://onlinelibrary.wiley.com/doi/10.1002/smll.201102743 by University Of Maastricht, Wiley Online Library on [11/07/2023]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
J. Huang et al.
2580 www.small-journal.com © 2012 Wiley-VCH Verlag GmbH & Co. KGaA, Weinheimfull papers
  2.3. Time Course of Cell Entry of Au-GO 
 To understand the cell internalization process of the Au-GO, 
we measured the SERS spectra of Au-GO incubated with Ca Ski cells for 1, 2, 4, 6, 8, and 12 h. As shown in  Figure     5  , incu-
bation for 1 and 2 h yields no detectable SERS signal, which suggests that incubation for 1 or 2 h was too short for a sig-niﬁ cant amount of Au-GO to enter into cells. Strong SERS 
signals of GO were detected when the Au-GO was incubated with Ca Ski cells for 4 h, and the signal reached its strongest after 6 h of incubation. After that, the SERS weakened, and was barely above the noise level at 12 h of incubation. From the above spectral data we speculate that the amount of Au-GO taken up by Ca Ski cells increased signiﬁ  cantly at an 
incubation time of 4 h, and reached a maximum at 6 h. After that the 

__________

Huang, Jie, et al. "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy." Wiley Online Library, vol. 2577, 2012, www.wileyonlinelibrary.com. Accessed 14 July 2023.:

herjee  ,   R. N.   Ghosh  ,   F. R.   Maxﬁ  eld  ,  Physiol. Rev.    1997  ,  77 , 
 759 .  
    [ 32 ]     S. C.   Silverstein  ,   R. M.   Steinman  ,   Z. A.   Cohn  ,  Annu. Rev. Biochem.   
 1977  ,  46 ,  669 .  
    [ 33 ]     S. L.   Schmid  ,   L. L.   Carter  ,  J. Cell Biol.    1990  ,  111 ,  2307 .  
    [ 34 ]     Y.   Xiao  ,   S. P .   Forry  ,   X. G.   Gao  ,   R. D.   Holbrook  ,   W. G.   Telford  ,   A.   Tona  , 
 J. Nanobiotechnol.    2010  ,  8 ,  13 .  
    [ 35 ]     U. S.   Huth  ,   R.   Schubert  ,   R.   Peschka-Süss  ,  J. Controlled Release   
 2006  ,  110 ,  490 .  
    [ 36 ]     N. E.   Bishop  ,  Rev. Med. Virol.    1997  ,  7 ,  199 .  
    [ 37 ]     C. Y.   Yang  ,   M. F.   Tai  ,   C. P .   Lin  ,   C. W.   Lu  ,   J. L.   Wang  ,   J. K.   Hsiao  , 
  H. M.   Liu  ,  PLoS One    2011  ,  6 ,  25524 .  
    [ 38 ]     S.   Huth  ,   J.   Lausier  ,   S. W.   Gersting  ,   C.   Rudolph  ,   C.   Plank  ,   U.   Welsch  , 
  J.   Rosenecker  ,  J. Gene Med.    2004  ,  6 ,  923 .  
    [ 39 ]     B. D.   Chithrani  ,   W. C. W.   Chan  ,  Nano Lett.    2007  ,  7 ,  1542 .  
    [ 40 ]     Z. Q.   Chu  ,   Y. J.   Huang  ,   Q.   Tao  ,   Q.   Li  ,  Nanoscale    2011  ,  3 , 
 3291 .  
    [ 41 ]     N. W. S.   Kam  ,   H. J.   Dai  ,  J. Am. Chem. Soc.    2005  ,  127 , 
 6021 .  
    [ 42 ]     G.   Frens  ,  Nat. Phys. Sci.    1973  ,  241 ,  20 .  
    [ 43 ]     W. S.   Hummers  ,   R. E.   Offeman  ,  J. Am. Chem. Soc.    1958  ,  80 , 
 1339 .  
    [ 44 ]     N. I.   Kovtyukhova  ,   P . J.   Ollivier  ,   B. R.   Martin  ,   T. E.   Mallouk  , 
  S. A.   Chizhik  ,   E. V.   Buzaneva  ,   A. D.   Gorchinskiy  ,  Chem. Mater.   
 1999  ,  11 ,  771 .  
    [ 45 ]     S.   Stankovich  ,   D. A.   Dikin  ,   G. H. B.   Dommett  ,   K. M.   Kohlhaas  , 
  E. J.   Zimney  ,   E. A.   Stach  ,   R. D.   Piner  ,   S. T.   Nguyen  ,   R. S.   Ruoff  ,  Nature    2006  ,  442 ,  282 .  
 
  Received: December 28, 2011 
 Revised: March 19, 2012Published online: May 29, 2012        [ 1 ]     Z.   Liu  ,   J. T.   Robinson  ,   X. M.   Sun  ,   H. J.   Dai  ,  J. Am. Chem. Soc.    2008  , 
 130 ,  10876 .  
     [ 2 ]     X. M.   Sun  ,   Z.   Liu  ,   K.   Welsher  ,   J. T.   Robinson  ,   A.   Goodwin  ,   S.   Zaric  , 
  H. J.   Dai  ,  Nano Res.    2008  ,  1 ,  203 .  
     [ 3 ]     X. Y.   Yang  ,   X. Y.   Zhang  ,   Z.   Liu  ,   Y. F.   Ma  ,   Y.   Huang  ,   Y. S.   Chen  ,  J. 
Phys. Chem. C     2008  ,  112 ,  17554 .  
     [ 4 ]     X. Y.   Yang  ,   X. Y.   Zhang  ,   Y. F.   Ma  ,   Y.   Huang  ,   Y. S.   Wang  ,   Y. S.   Chen  ,  J. 
Mater. Chem.    2009  ,  19 ,  2710 .  
     [ 5 ]     W. J.   Hong  ,   H.   Bai  ,   Y. X.   Xu  ,   Z. Y.   Yao  ,   Z. Z.   Gu  ,   G. Q.   Shi  ,  J. Phys. 
Chem. C    2010  ,  114 ,  1822 .  
     [ 6 ]     Y.   Wang  ,   Z. H.   Li  ,   J.   Wang  ,   J. H.   Li  ,   Y. H.   Lin  ,  Trends Biotechnol.   
 2011  ,  29 ,  205 .  
     [ 7 ]     L. Z.

__________

Huang, Jie, et al. "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy." Wiley Online Library, vol. 2577, 2012, www.wileyonlinelibrary.com. Accessed 14 July 2023.:

2577 © 2012 Wiley-VCH Verlag GmbH & Co. KGaA, Weinheim wileyonlinelibrary.com
  1. Introduction 
 In recent years, an increasing interest in the biological and 
medical applications of graphene oxide (GO), such as drug/gene delivery, cancer therapy, biosensing, and cellular imaging, has emerged owing to its unique structure and intrinsic prop-erties. 
[  1–5  ]  Although much progress has been made on appli-
cations of GO in the biomedical ﬁ  eld, little is known about 
the mechanism of cellular uptake and intracellular pathway of GO. 
[  6  ,  7  ]  
 To this end, we have designed and prepared a conjugate 
(Au-GO) of GO and Au nanoparticles (NPs) and studied the cellular uptake of Au-GO by means of the surface-enhanced Raman scattering (SERS) technique. Here, the Au NPs served as SERS-active substrate and GO as a vehicle for loading and delivery of Au NPs into cells. The intrinsic Mechanism of Cellular Uptake of Graphene Oxide 
Studied by Surface-Enhanced Raman Spectroscopy
  Jie   Huang  ,     Cheng   Zong  ,     He   Shen  ,     Min   Liu  ,     Biao   Chen  ,     Bin   Ren  ,   *      and   Zhiju n   Zhang   *   
Raman signals of GO inside cells were examined to reveal 
the cellular uptake behavior of Au-GO. In a previous paper, [  8  ]  
we prepared Au-GO by assembly of 2-mercaptopyridine-modiﬁ  ed Au NPs onto the GO surface, and observed two 
strong peaks at 1330 and 1600 cm 
 − 1 , characteristic of D and G 
bands of graphene, [  9  ]  in the SERS spectrum of GO. In addi-
tion, we showed that introduction of Au NPs onto the GO surface led to a signiﬁ  cantly enhanced Raman signal of probe 
molecules, compared to that for isolated Au NPs. 
[  8  ]  In this 
work, we investigated the cellular uptake of Au-GO by moni-toring the intrinsic Raman signal of GO in live cells. To eluci-date the entry mechanism, we examined the entry of Au-GO into cells pretreated with several types of endocytic inhibi-tors which selectively block speciﬁ  c uptake pathways. 
[  10  ,  11  ]  
We demonstrate that the SERS technique is very useful for studying the cellular uptake behavior of GO. By means of the SERS technique combined with ﬂ  uorescence microscopy 
and transmission electron microscopy (TEM), we conclude that cell entry of the Au-GO is mainly via energy-dependent, clathrin-mediated endocytosis. 
   2. Results and Discussion 
  2.1. Synthesis of Au-GO 
 In our previous paper, we formed Au-GO via assembly of 
2-mercaptopyridine-modiﬁ  ed Au NPs onto GO sheet via 
 π – π  stacking, and found that Au-GO exhibited a signiﬁ  cantly  DOI: 10.1002/smll.201102743  The last few years have witnessed rapid development of biological and medical 
applications of graphene oxide (GO), such as drug/gene delivery, biosensing, and bioimaging. However, little is known about the cellular uptake mechanism and pathway of GO. In this work, surface-enhanced Raman scattering (SERS) spectroscopy is employed to investigate the cellular internalization of GO loaded with Au nanopar

__________

Mu, Qingxin, et al. "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets." Department of Chemical Biology & Therapeutics, St. Jude Children’s Research Hospital, 2023.:

.; Wan, J.; Zhang, S.; Tian, B.; Zhang, Y.; Liu, Z.
Biomaterials 2012 ,33, 2206 −2214.■NOTE ADDED AFTER ASAP PUBLICATION
This paper was published on the Web on March 23, 2012.
Additional minor text corrections were added, and thecorrected version was reposted on March 27, 2012.ACS Applied Materials & Interfaces Research Article
dx.doi.org/10.1021/am300253c |ACS Appl. Mater. Interfaces 2012, 4, 2259 −2266 2266


__________

Mu, Qingxin, et al. "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets." Department of Chemical Biology & Therapeutics, St. Jude Children’s Research Hospital, 2023.:

35 1.1 ±0.2 9.1 ±7.1 3.9
PCGO1 0.86 ±0.37 1.2 ±0.6 9.6 ±7.2 5.2
PCGO2 0.42 ±0.26 1.1 ±0.3 5.2 ±3.2 3.2
aThe equivalent disk diameter data were skewed, and the Box-Cox
transformation was applied to yield a more normal distribution.24The
standard deviations of the transformed data were retransformed back
to the original data scale to obtain the reported standard deviationvalues.bAverage height was measured across the surface area of all
nanosheets.ACS Applied Materials & Interfaces Research Article
dx.doi.org/10.1021/am300253c |ACS Appl. Mater. Interfaces 2012, 4, 2259 −2266 2261
FITC-BSA was released from GO for at least 24 h (Figure S2 in
Supporting Information).
Cell Surface Adhesion of PCGO. Cellular uptake
mechanisms of nanoparticles having various shapes have beenreported. For instance, spherical nanoparticles enter cellsthrough CME, caveolae-mediated endocytosis, phagocytosis,or macropinocytosis, which all require energy.
14−16Tubular
nanoparticles enter cells through endocytosis or energy-independent direct penetration.17,18All these processes require
that nanoparticles attach to the cellular membrane beforeengulfment or insertion.19Unlike spherical or tubular nano-
particles, GO has large flat surfaces with irregular shapes.Additionally, the flexibility and folding properties of GO ’s thin
layers make them act as gauzelike shapes in biological medium.
GO has been reported to be an efficient intracellular
transporter for drug and gene delivery, indicating that it canefficiently enter cells.
2On the basis of these observations, we
hypothesized that GO adheres to the cell surface and is theninternalized.
Driven by our preliminary hypothesis, we first investigated
whether PCGO could attach to the surface of cells and in whatorientation this occurred. A model cell line C2C12 (mousemesenchymal progenitor) was selected in this study. UponSEM examinations, large PCGO pieces were frequentlyobserved adhering face to face onto the cell surface (Figure1B,C). We never observed any PCGO and cells bindingperpendicularly. On the basis of SEM observations, previousreport on nanoparticle −cell interactions, and properties of
PCGO, we speculate that the adhesion is a result of severalfactors. First, the similar curvature between the nanosheets andplasma membrane would facilitate their holding together.Second, there are multiple binding forces between them,including electrostatic and hydrophobic interactions betweennanosheets and phospholipid bilayers. Third, there could bespecific ligand −receptor interactions between proteins bound
to PCGO and membrane receptors. This factor might inducereceptor-mediated endocytosis of PCGO. On the basis of∼10 000 BSA molecules per square micrometer area of GO
(AFM studies) and the cross section of BSA ∼14×4×4 nm,
we estimate that the average BSA coverage on the grapheneoxide surface to be 43%. Therefore, although the density of
protein molecules on GO surface is high, there is still space on
GO surface to facilitate the di


### Question 2


```python
display(Markdown("\n\n__________\n\n".join([":\n\n".join([answer2.dict()['contexts'][i]['text']['doc']['citation'], answer2.dict()['contexts'][i]['text']['text'], ]) for i in range(len(answer2.dict()['contexts']))])))
```


Yue, Hua, et al. "The Role of the Lateral Dimension of Graphene Oxide in the Regulation of Cellular Responses." National Key Laboratory of Biochemical Engineering, Institute of Process Engineering, Chinese Academy of Sciences, 2012.:

The role of the lateral dimension of graphene oxide in the regulation of cellular
responses
Hua Yuea,b,1, Wei Weia,1, Zhanguo Yuea,b, Bin Wanga, Nana Luoa,b, Yongjun Gaoc, Ding Mac,
Guanghui Maa,*, Zhiguo Sua,*
aNational Key Laboratory of Biochemical Engineering, Institute of Process Engineering, Chinese Academy of Sciences, P.O. Box 353, Beijing 100190, PR China
bGraduate University of the Chinese Academy of Sciences, Beijing 100049, PR China
cCollege of Chemistry and Molecular Engineering, Peking University, Beijing 100871, PR China
article info
Article history:
Received 29 November 2011Accepted 7 February 2012Available online 28 February 2012
Keywords:
CarbonCytokineCytotoxicityDrug delivery
Inﬂammationabstract
The nanomaterial graphene oxide (GO) has attracted explosive interests in various areas. However, its
performance in biological environments is still largely unknown, particularly with regard to cellularresponse to GO. Here we separated the GO sheets in different size and systematically investigated size
effect of the GO in response to different types of cells. In terms of abilities to internalize GO, enormous
discrepancies were observed in the six cell types, with only two phagocytes were found to be capable ofinternalizing GO. The 2
mm and 350 nm GO greatly differed in lateral dimensions, but equally contributed
to the uptake amount in macrophages. Similar amounts of antibody opsonization and active Fc g
receptor-mediated phagocytosis were demonstrated the cause of this behavior. In comparison with thenanosized GO, the GO in micro-size showed divergent intracellular locations and induced much strongerinﬂammation responses. Present study provided insight into selective internalization, size-independent
uptake, and several other biological behaviors undergone by GO. These ﬁndings might help build
necessary knowledge for potential incorporation of the unique two-dimensional nanomaterial asa biomedical tool, and for avoiding potential hazards.
/C2112012 Elsevier Ltd. All rights reserved.
1. Introduction
Understanding the performance of engineered micro/nano
materials in a biological context is an important issue for guiding
their biomedical applications. Typically, zero-dimensional (0D)
fullerenes and one-dimensional (1D) carbon nanotubes (CNTs)
initiated two surges, and the evaluation of their interaction with
living matter strongly voted great potentials in cancer therapy,
molecular imaging, and drug delivery [1e3]. Following fullerene and
CNTs, ultrathin but very strong two-dimensional (2D) graphenessoon draw much more attentions [4e6]and have merited the 2010
Nobel Prize in physics. Apart from the tremendous interest in elec-trical applications, graphene-based material is also an exciting
candidate for exploration in the biological context. The unique 2D
high surface area structure can potentially act as a template for cargo
molecules ( e.g.proteins, nucleic acids, and drug entities) [7]. Water-
insoluble anti-cancer drugs ( e.g. hydr

__________

Mu, Qingxin, et al. "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets." Department of Chemical Biology & Therapeutics, St. Jude Children’s Research Hospital, 2023.:

Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide
Nanosheets
Qingxin Mu,†Gaoxing Su,†,‡Liwen Li,†,‡Ben O. Gilbertson,§Lam H. Yu,§Qiu Zhang,‡Ya-Ping Sun,⊥
and Bing Yan *,†,‡
†Department of Chemical Biology & Therapeutics, St. Jude Children ’s Research Hospital, Memphis, Tennessee, 38105, United States
‡School of Chemistry and Chemical Engineering, Shandong University, Jinan, China, 250100
§Department of Physics, University of Memphis, Memphis, Tennessee, 38152, United States
⊥Department of Chemistry and Laboratory for Emerging Materials and Technology Hunter Hall, Clemson University, Clemson,
South Carolina, 29634-0973, United States
*SSupporting Information
ABSTRACT: As an emerging applied material, graphene has shown
tremendous application potential in many fields, including bio-medicine. However, the biological behavior of these nanosheets,especially their interactions with cells, is not well understood. Here,we report our findings about the cell surface adhesion, subcellularlocations, and size-dependent uptake mechanisms of protein-coatedgraphene oxide nanosheets (PCGO). Small nanosheets enter cellsmainly through clathrin-mediated endocytosis, and the increase ofgraphene size enhances phagocytotic uptake of the nanosheets. Thesefindings will facilitate biomedical and toxicologic studies of graphenesand provide fundamental understanding of interactions at theinterface of two-dimensional nanostructures and biological systems.
KEYWORDS: graphene oxide nanosheets, protein binding, cell uptake, clathrin-mediated endocytosis, phagocytosis, size dependence
■INTRODUCTION
Graphene, a hexagonal carbon nanostructure similar to carbon
nanotubes and fullerene, has unique electronic, thermal, andmechanical properties, showing tremendous application
potential in fields such as electronics and biomedicine.
1,2
Graphene oxide (GO), which is oxidized graphite withenhanced aqueous solubility, has been proven to be an efficient
biosensor,
3drug carrier,4,5and photothermal cancer-killing
agent.6,7GO nanosheets are able to enter cells which renders
them to become promising candidates for intracellular deliveryof drugs and cellular imaging. However, the mechanisms of how
the emerging nanostrucutures interface with biological systems
are still largely unknown. In particular, a fundamental
understanding of its ability to penetrate cell membranes and
other biological barriers is still lacking. For instance, whether
the nanosheets parallelly attach onto cell surface or vertically
insert into cell membrane? By what manner they enter cells?
Such cellular uptake properties of nanoparticles may affect cell
signaling, proliferation, differentiation, and nanoparticle ex-cretion.
8−10Cellular uptake of nanoparticles with other shapes
has been studied.11We and other researchers previously
discovered endosomal leakage and nuclear translocation of
multiwalled carbon nanotubes.9,12However, the behavior of
sheet-shaped nanostructures has not been reported. Further-
more, 

__________

Yue, Hua, et al. "The Role of the Lateral Dimension of Graphene Oxide in the Regulation of Cellular Responses." National Key Laboratory of Biochemical Engineering, Institute of Process Engineering, Chinese Academy of Sciences, 2012.:

 Goldstein LS. Cruising along microtubule highways: how membranes
move through the secretory pathway. J Cell Biol 1998;140:1277 e80.
[43] De Koker S, De Geest BG, Singh SK, De Rycke R, Naessens T, Van Kooyk Y, et al.
Polyelectrolyte microcapsules as antigen delivery vehicles to dendritic cells:
uptake, processing, and cross-presentation of encapsulated antigens. Angew
Chem Int Ed 2009;48:8485 e9.
[44] Mantovani A. B cells and macrophages in cancer: yin and yang. Nat Med 2011;
17:285 e6.
[45] Stow JL, Low PC, Offenhauser C,Sangermani D.Cytokinesecretion in macrophages
and other cells: pathways and mediators. Immunobiology 2009;214:601 e12.H. Yue et al. / Biomaterials 33 (2012) 4013 e4021 4021

__________

Yue, Hua, et al. "The Role of the Lateral Dimension of Graphene Oxide in the Regulation of Cellular Responses." National Key Laboratory of Biochemical Engineering, Institute of Process Engineering, Chinese Academy of Sciences, 2012.:

styrene nanoparticles by human macrophagesand a monocytic cell line. Acs Nano 2011;5:1657 e69.
[26] Dobrovolskaia MA, McNeil SE. Immunological properties of engineered
nanomaterials. Nat Nanotechnol 2007;2:469 e78.
[27] Jiang W, Kim BY, Rutka JT, Chan WC. Nanoparticle-mediated cellular response
is size-dependent. Nat Nanotechnol 2008;3:145 e50.
[28] Lundqvist M, Stigler J, Elia G, Lynch I, Cedervall T, Dawson KA. Nanoparticle size
and surface properties determine the protein corona with possible implicationsfor biological impacts. Proc Natl Acad Sci USA 2008;105:14265 e70.
[29] Stankovich S, Dikin DA, Dommett GHB, Kohlhaas KM, Zimney EJ, Stach EA,
et al. Graphene-based composite materials. Nature 2006;442:282 e6.
[30] Stone V, Johnston H, Schins RPF. Development of in vitro systems for nano-
toxicology: methodological considerations. Crit Rev Toxicol 2009;39:613 e26.
[31] Zhang Y, Ali SF, Dervishi E, Xu Y, Li Z, Casciano D, et al. Cytotoxicity effects of
graphene and single-wall carbon nanotubes in neural phaeochromocytoma-derived PC12 cells. Acs Nano 2010;4:3181 e6.
[32] Ge C, Du J, Zhao L, Wang L, Liu Y, Li D, et al. Binding of blood proteins to carbon
nanotubes reduces cytotoxicity. Proc Natl Acad Sci USA 2011;108:16968 e73.
[33] Sadik OA, Zhou AL, Kikandi S, Du N, Wang Q, Varner K. Sensors as tools for
quantitation, nanotoxicity and nanomonitoring assessment of engineerednanomaterials. J Environ Monitor 2009;11:1782 e800.
[34] Xie LM, Ling X, Fang Y, Zhang J, Liu ZF. Graphene as a substrate to suppress
ﬂuorescence in resonance Raman spectroscopy. J Am Chem Soc 2009;131:
9890e1.
[35] Singh SK, Singh MK, Nayak MK, Kumari S, Gracio JJA, Dash D. Size distribution
analysis and physical/ﬂ uorescence characterization of graphene oxide sheets
byﬂow cytometry. Carbon 2011;49:684 e92.H. Yue et al. / Biomaterials 33 (2012) 4013 e4021 4020[36] Yue ZG, Wei W, Lv PP, Yue H, Wang LY, Su ZG, et al. Surface charge affects
cellular uptake and intracellular traf ﬁcking of chitosan-based nanoparticles.
Biomacromolecules 2011;12:2440 e6.
[37] Yue H, Wei W, Yue ZG, Lv PP, Wang LY, Ma GH, et al. Particle size affects the
cellular response in macrophages. Eur J Pharm Sci 2010;41:650 e7.
[38] Shanbhag AS, Jacobs JJ, Black J, Galante JO, Glant TT. Macrophage/particle
interactions: effect of size, composition and surface area. J Biomed Mater Res
1994;28:81 e90.
[39] Gratton SE, Ropp PA, Pohlhaus PD, Luft JC, Madden VJ, Napier ME, et al. The
effect of particle design on cellular internalization pathways. Proc Natl Acad
Sci USA 2008;105:11613 e8.
[40] Petros RA, DeSimone JM. Strategies in the design of nanoparticles for thera-
peutic applications. Nat Rev Drug Discov 2010;9:615 e27.[41] Geng Y, Dalhaimer P, Cai S, Tsai R, Tewari M, Minko T, et al. Shape effects of
ﬁlaments versus spherical particles in ﬂow and drug delivery. Nat Nano-
technol 2007;2:249 e55.
[42] Bloom GS, Goldstein LS. Cruising along microtubule highways: how membranes
move through the secretory pathway

__________

Yue, Hua, et al. "The Role of the Lateral Dimension of Graphene Oxide in the Regulation of Cellular Responses." National Key Laboratory of Biochemical Engineering, Institute of Process Engineering, Chinese Academy of Sciences, 2012.:

 nanometer scale thickness) affects
cellular response is poorly understood, which needs to be addressed
urgently. In order to ﬁll this knowledge gap, we systematically
investigated the effects of GO lateral dimension, from nano to micro,on a series of cellular responses including the cellular uptake,
internalization mechanisms, intracellular trafﬁ cking, and in ﬂam-
mation response. Two macrophages (peritoneal macrophage PMØand murine macrophage J774A.1 cell line) and four non-phagocyticcells (murine Lewis lung carcinoma LLC, human breast cancer
MCF-7, human hepatocarcinoma cells HepG2, and human umbilical
vein endothelial cells HUVEC) were exposed to GO with different
lateral dimensions, and the cellular responses were testi ﬁed by
exploring the intrinsic properties of this nanomaterial.
2. Materials and methods
2.1. Materials
Ethylenediaminetetraacetic acid (EDTA), nocodazole, and latrunculin B were
purchased from Merck. Glutaradehyde was from Sigma eAldrich Inc. Penicillin and
streptomycin, Gibco Dulbecco ’s Phosphate-Buffered Saline (D-PBS), Hank ’s solution,
Gibco Dulbecco ’s Modi ﬁed Eagle ’s Medium (DMEM), Rhodamine-phalloidin, 4,
6-diamidino-2-phenylindole (DAPI), Lyso Tracker Red DND-99, and LIVE/DEAD Cell
Viability Kit were all bought from Invitrogen. Fetal bovine serum (FBS) was fromHyClone. Cell-Counting Kit-8 (CCK8) kit was from the Dojindo Laboratories. BD/C212Cytometric Bead Array (CBA) Mouse In ﬂammation Kit was obtained from
BD Biosciences. Mouse Immunology G (Ig G), anti-Fc
gRI (anti-CD64) antibody (Ab),
anti-Fc gRIII (anti-CD16) Ab, and anti-mannose receptor (anti-CD206) Ab were
ordered from Biolegend. Bovine IgG and horseradish peroxidase (HRP)-labeled GoatAnti-bovine IgG were from KPL Inc. All other reagents were of analytic grade.
2.2. Synthesis and characterization of GO
2.2.1. GO preparation
Preparation of uniform-sized sheets was started from the primary GO that made
by a modi ﬁed Hummers method. After suf ﬁcient sonication and washing, the GO
sheets were separated by making use of speci ﬁc sedimentation rates of graphene
in different size. The centrifugal forces were selected as 100 e200 g and
10,000 e30,000 g to obtain the 2
mm GO and 350 nm sheets, respectively. For
preparation of Mn-free GO, 3% H 2O2solution was used to reduce residual KMnO 4
and MnO 2. The solid product was separated by ﬁltration, washed repeatedly with 5%
HCl solution until the sulfate could not be detected with BaCl 2, and ﬁnally washed
with deionized water to neutrality.
2.2.2. Atomic force microscopy (AFM) analysis of GO
In virtue of a typical absorption peak at 230 nm induced by the pep* transition
of GO [21], concentrations were determined using an Ultrospec 2100 pro UV/Visible
spectrophotometer. GO characterization was performed on a BioScope Catalyst AFM
(Veeco), operating in tapping mode in air at room temperature.
2.2.3. GO stability and dispersion capacity
To examine the stability and dispersion capacity of GO with the two sizes, 100 mg
GO

__________

Akhavan, Omid, Ghaderi, Elham, and Akhavan, Alireza. "Size-dependent genotoxicity of graphene nanoplatelets in human stem cells." Department of Physics, Sharif University of Technology, 2012.:

Size-dependent genotoxicity of graphene nanoplatelets in human stem cells
Omid Akhavana,b,*, Elham Ghaderia, Alireza Akhavana
aDepartment of Physics, Sharif University of Technology, P.O. Box 11155e 9161, Tehran, Iran
bInstitute for Nanoscience and Nanotechnology, Sharif University of Technology, P.O. Box 14588 e89694, Tehran, Iran
article info
Article history:
Received 22 May 2012Accepted 20 July 2012
Available online 3 August 2012
Keywords:
Reduced graphene Oxide
Nanoscale lateral dimension
Size effectCytotoxicityGenotoxicityStem cellsabstract
Reduced graphene oxide nanoplatelets (rGONPs) were synthesized by sonication of covalently PEGylated
GO sheets followed by a chemical reduction using hydrazine and bovine serum albumin. Human
mesenchymal stem cells (hMSCs), as a fundamental factor in tissue engineering, were isolated fromumbilical cord blood (as a recently proposed source for extracting fresh hMSCs) to investigate, for theﬁrst time, the size-dependent cyto- and geno-toxic effects of the rGONPs on the cells. The cell viability
test showed signi ﬁcant cell destructions by 1.0
mg/mL rGONPs with average lateral dimensions (ALDs) of
11/C64 nm, while the rGO sheets with ALDs of 3.8 /C60.4mm could exhibit a signi ﬁcant cytotoxic effect only
at highconcentration of 100 mg/mL after 1 h exposure time. Although oxidative stress and cell wall
membrane damage were determined as the main mechanism involved in the cytotoxicity of the rGOsheets, neither of them could completely describe the cell destructions induced by the rGONPs, espe-cially at the concentrations /C201.0
mg/mL. In fact, the rGONPs showed genotoxic effects on the stem cells
through DNA fragmentations and chromosomal aberrations, even at low concentration of 0.1 mg/mL. Our
results present essential knowledge for more ef ﬁcient and innocuous applications of graphene sheets
and particularly nanoplatelets in upcoming nanotechnology-based tissue engineering as, e.g., drug
transporter and scaffolds.
/C2112012 Elsevier Ltd. All rights reserved.
1. Introduction
Graphene, as a single-atom-thick sheet of sp2-bonded carbon
atoms with a two-dimensional honeycomb lattice structure, has
received increasing attentions in recent years due to its exciting
promises for fundamental research and progressive nanotech-
nology-based applications. The unique physicochemical proper-
ties of graphene (including high surface area [ 1], extraordinary
electrical [ 2] and thermal [ 3] conductivities, strong mechanical
strength, capability of bio-functionalization and mass production
[4]) have drawn the attention of scienti ﬁc community towards
numerous potential applications in, e.g., biotechnology such as
biosensing [ 4,5] disease diagnosis [ 6], bacterial inhibition [ 7-11 ],
antiviral materials [ 12], cancer targeting [ 13,14] and photothermal
therapy [ 15e17], drug delivery [ 18e20], electrical stimulation of
cells [ 21] and tissue engineering [ 22-24 ]. This means that simul-
taneous detailed investigations abou

__________

Mu, Qingxin, et al. "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets." Department of Chemical Biology & Therapeutics, St. Jude Children’s Research Hospital, 2023.:

.; Wan, J.; Zhang, S.; Tian, B.; Zhang, Y.; Liu, Z.
Biomaterials 2012 ,33, 2206 −2214.■NOTE ADDED AFTER ASAP PUBLICATION
This paper was published on the Web on March 23, 2012.
Additional minor text corrections were added, and thecorrected version was reposted on March 27, 2012.ACS Applied Materials & Interfaces Research Article
dx.doi.org/10.1021/am300253c |ACS Appl. Mater. Interfaces 2012, 4, 2259 −2266 2266


__________

Mu, Qingxin, et al. "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets." Department of Chemical Biology & Therapeutics, St. Jude Children’s Research Hospital, 2023.:

with the uptake of pooled PCGO at 4 °C
(Figure 1G,H). Cytochalasin D (Cyto D) inhibits actinpolymerization and actin microfilaments formation. It inhibitsphagocytosis in cells that possess partial or full phagocytoticfunction.
29C2C12 cells are known to possess phagocytic
activity.30Cyto D inhibited cell uptake in a size-dependent
manner (Figure 5). It inhibited the uptake of PCGO1 muchstronger than it did that of PCGO2, indicating that largernanosheets entered cells predominantly through phagocytosis.Chlorpromazine, a Rho GTPase and CME inhibitor,
31reduced
Figure 3. Ultrastructural examination of C2C12 cells incubated with
GNP-labeled PCGO1 (50 μg/mL, 30 min). (A) PCGO1 adhered on
the cell surface, (B) in cell invaginations, and (C) in intracellular
vesicles. Red arrows indicate GNP-labeled PCGO1. The scale bars
represent 100 nm.
Figure 4. Ultrastructural examination of C2C12 cells incubated with
GNP-labeled PCGO2 (50 μg/mL, 30 min). (A) PCGO2 adhered on
the cell surface, (B) in cell invaginations, and (C) in intracellular
vesicles. Red arrows indicate GNP-labeled PCGO2. The scale barsrepresent 100 nm in A and B and 500 nm in C.
Figure 5. Inhibition of cellular uptake by various inhibitors. The
percentage of inhibition was generated from the ratios between mean
fluorescence intensities of cells incubated with PCGOs (50 μg/mL)
for 1 h with and without various inhibitors. All results are expressed as
mean ±SD;∗,p< 0.05.ACS Applied Materials & Interfaces Research Article
dx.doi.org/10.1021/am300253c |ACS Appl. Mater. Interfaces 2012, 4, 2259 −2266 2264
the uptake of all PCGO nanosheets, yet the inhibition of
PCGO2 was much stronger than that of PCGO1, indicatingthat smaller PCGO nanosheets entered cells primarily throughCME. Our TEM observations, flow cytometric quantification,and inhibition assay results collectively showed that protein-coated small nanosheets enter cells through CME, while largenanosheets use both CME and phagocytosis. Because there isstill no detailed description of cellular uptake mechanisms oftwo-dimensional nanosheets, we propose a working model forthe cellular uptake of large and small PCGOs (Figure 6). Forlarge PCGO, the nanosheets first attach onto cell surfacefollowed by membrane invagination and extending ofpseudopodia and are subsequently engulfed into phagosome.For small PCGO, the nanosheets attach onto cell surfacefollowed by formation of clathrin-coated pits and are
subsequently engulfed into endosome. All nanosheets enter
lysosomes for excretion. The cellular uptake properties ofdifferent sizes of PCGO provide insight into the bioactivity ofthese nanostructures that can be used in designing biomedicaldevices. Large nanosheets may preferentially translocate intothe reticuloendothelial system, and small nanosheets arecapable of being distributed in various organs. The in vitroand in vivo activity of GO nanosheets may be regulatable bycontrolling their size. Such hypothesis is consistent with recentstudies.
32,33In an

__________

Akhavan, Omid, Ghaderi, Elham, and Akhavan, Alireza. "Size-dependent genotoxicity of graphene nanoplatelets in human stem cells." Department of Physics, Sharif University of Technology, 2012.:

depends
on the lateral size of the sheets.
Acknowledgements
O. Akhavan would like to thank the Research Council of Sharif
University of Technology and also Iran Nanotechnology Initiative
Council for ﬁnancial support of the work.
References
[1] Li D, Müller MB, Gilje S, Kaner RB, Wallace GG. Processable aqueous disper-
sions of graphene nanosheets. Nat Nanotechnol 2008;3:101 e5.
[2] Geim AK, Novoselov KS. The rise of graphene. Nat Mater 2007;6:183 e91.
[3] Koh YK, Bae M, Cahill DG, Pop E. Heat conduction across monolayer and few-
layer graphenes. Nano Lett 2010;10:4363 e8.
[4] Shao YY, Wang J, Wu H, Liu J, Aksay IA, Lin YH. Graphene based electro-
chemical sensors and biosensors: A review. Electroanal 2010;22:1027 e36.
[5] Akhavan O, Ghaderi E, Rahighi R. Toward single-DNA electrochemical bio-
sensing by graphene nanowalls. ACS Nano 2012;6:2904 e16.
[6] Mohanty N, Berry V. Graphene-based single-bacterium resolution biodevice
and DNA transistor: Interfacing graphene derivatives with nanoscale and
microscale biocomponents. Nano Lett 2008;8:4469 e76.
[7] Akhavan O, Ghaderi E. Toxicity of graphene and graphene oxide nanowalls
against bacteria. ACS Nano 2010;4:5731 e6.
[8] Hu W, Peng C, Luo W, Lv M, Li X, Li D, et al. Graphene-based antibacterial
paper. ACS Nano 2010;4:4317 e23.
[9] Akhavan O, Ghaderi E. Escherichia coli bacteria reduce graphene oxide to
bactericidal graphene in a self-limiting manner. Carbon 2012;50:1853 e60.
[10] Ma J, Zhang J, Xiong Z, Yong Y, Zhao XS. Preparation, characterization and
antibacterial properties of silver-modi ﬁed graphene oxide. J Mater Chem
2011;21:3350 e2.
[11] Akhavan O, Ghaderi E. Photocatalytic reduction of graphene oxide nanosheets
on TiO 2thinﬁlm for photoinactivation of bacteria in solar light irradiation.
J Phys Chem C 2009;113:20214 e20.
[12] Akhavan O, Choobtashani M, Ghaderi E. Protein degradation and RNA ef ﬂux of
viruses photocatalyzed by graphene-tungsten oxide composite under visiblelight irradiation. J Phys Chem C 2012;116:9653 e9.O. Akhavan et al. / Biomaterials 33 (2012) 8017 e8025 8024[13] Yang K, Zhang S, Zhang G, Sun X, Lee ST, Liu Z. Graphene in mice: Ultrahigh in vivo
tumor uptake and ef ﬁcient photothermal therapy. Nano Lett 2010;10:3318 e23.
[14] Robinson JT, Tabakman SM, Liang Y, Wang H, Sanchez Casalongue H, Vinh D,
et al. Ultrasmall reduced graphene oxide with high near-infrared absorbance
for photothermal therapy. J Am Chem Soc 2011;133:6825 e31.
[15] Yang K, Wan J, Zhang S, Tian B, Zhang Y, Liu Z. The in ﬂuence of surface
chemistry and size of nanoscale graphene oxide on photothermal therapy ofcancer using ultra-low laser power. Biomaterials 2011;33:2206 e14.
[16] Zhang W, Guo Z, Huang D, Liu Z, Guo X, Zhong H. Synergistic effect of chemo-
photothermal therapy using PEGylated graphene oxide. Biomaterials 2011;
32:8555 e61.
[17] Akhavan O, Ghaderi E, Aghayee S, Fereydooni Y, Talebi A. The use of a glucose-
reduced graphene oxide suspension for photothermal cancer therapy. J MaterChem 201

__________

Huang, Jie, et al. "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy." Wiley Online Library, vol. 2577, 2012, www.wileyonlinelibrary.com. Accessed 14 July 2023.:

Au-GO inside the cell from different spots of the dark-ﬁ  eld microscopic image. Two prominent peaks at 1330 
and 1600 cm 
 − 1  appear, which can be assigned undoubtedly to the D and G bands of graphene, [  8  ]  respectively, thereby 
indicating the internalization of Au-GO into Ca Ski cells. It is interesting to note that the SERS intensity of GO varies dramatically at different chosen spots inside the cell, which implies inhomogeneous distribution of GO inside the cell. This observation highlights the good spatial resolution of the SERS technique for study of intracellular events, which is similar to the widely used ﬂ  uorescence method. 
[  28  ]  This 
can be conﬁ  rmed by checking the SERS spectra of many 
cells incubated with Au-GO. To compare the Raman spectra of GO (Figure  3 ) and Au-GO (Figure  4 ) inside cells, the weight concentrations of GO and Au-GO were adjusted to be the same in terms of GO (1.6  μ g mL 
 − 1 ). Clearly, the 
presence of Au NPs on the GO sheet plays an important role in the observation of enhanced Raman spectra of GO in live cells.
         Figure  3 .     a) Bright and b) dark-ﬁ  eld microscopic images of Ca Ski cells incubated with GO. c) Raman spectra of the different spots marked in (b) in 
the Ca Ski cell.  
(a) 
(b)500 1000 1500 2000 2500 3000
Raman shift (cm-1)1
2
3
4
550 CPSRaman Intensity
(c) 
     Figure  4 .     a) Bright and dark-ﬁ  eld microscopic images of Ca Ski cells incubated with Au-GO for 4 h. c) Raman spectra of  the different spots marked 
in (b) in the Ca Ski cell.  
(a) 
500 1000 1500 2000 2500 30005431Intensity (a.u.)
Raman shift (cm-1)100 cps
1330
1600
2
(c) (b)
12
3
45
small  2012, 8, No. 16, 2577–2584
 16136829, 2012, 16, Downloaded from https://onlinelibrary.wiley.com/doi/10.1002/smll.201102743 by University Of Maastricht, Wiley Online Library on [11/07/2023]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License
J. Huang et al.
2580 www.small-journal.com © 2012 Wiley-VCH Verlag GmbH & Co. KGaA, Weinheimfull papers
  2.3. Time Course of Cell Entry of Au-GO 
 To understand the cell internalization process of the Au-GO, 
we measured the SERS spectra of Au-GO incubated with Ca Ski cells for 1, 2, 4, 6, 8, and 12 h. As shown in  Figure     5  , incu-
bation for 1 and 2 h yields no detectable SERS signal, which suggests that incubation for 1 or 2 h was too short for a sig-niﬁ cant amount of Au-GO to enter into cells. Strong SERS 
signals of GO were detected when the Au-GO was incubated with Ca Ski cells for 4 h, and the signal reached its strongest after 6 h of incubation. After that, the SERS weakened, and was barely above the noise level at 12 h of incubation. From the above spectral data we speculate that the amount of Au-GO taken up by Ca Ski cells increased signiﬁ  cantly at an 
incubation time of 4 h, and reached a maximum at 6 h. After that the 


### Question 3


```python
display(Markdown("\n\n__________\n\n".join([":\n\n".join([answer3.dict()['contexts'][i]['text']['doc']['citation'], answer3.dict()['contexts'][i]['text']['text'], ]) for i in range(len(answer3.dict()['contexts']))])))
```


"Ma, Minghao, Xu, Ming, and Liu, Sijin. 'Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface.' Acta Chimica Sinica, vol. 78, 2020, pp. 877-887.":

综述 
Review  
 
 * E-mail: mingxu@rcees.ac.cn 
Received June 8, 2020; published August 3, 2020. 
Project supported by the National Natural Science Foundation of China (Nos. 21922611, 21637004, 21920102007) and the Youth Inno vation Promotion Asso-
ciation CAS (No. 2019042). 
  项目受国家自然科学基金 (Nos. 21922611, 21637004, 21920102007) 和中国科学院青年创新促进会 (No. 2019042) 资助 . 
  
Acta Chim. Sinica 2020 , 78, 877—887 © 2020 Shanghai Institute of Organic Chemistry, Chinese Academy of Sciences http://sioc-journal.cn    877 化 学 学 报 
 ACTA CHIMICA SINICA 
 
氧化石墨烯的表面化学修饰及纳米 −生物界面作用机理  
马明昊a,b    徐明 *,a,b,c    刘思金a,b 
(a中国科学院生态环境研究中心   环境化学与生态毒理学国家重点实验室   北京  100085) 
(b中国科学院大学   北京  100049) 
(c国科大杭州高等研究院   杭州  310024) 
摘要   由于具备独特的物理化学性质 , 氧化石墨烯已被广泛地应用于生命科学与人体健康等相关领域 . 然而 , 如何最
大化地发挥氧化石墨烯的优势与特点 , 并克服其自身固有性质导致的生物不良效应 , 依然是当前面临的难题 . 为更好
地了解该领域的研究现状 , 本文主要综述了近年来氧化石墨烯的表面化学调控和生物作用机理方面的最新研究进展 . 
首先 , 简要介绍了氧化石墨烯的物理化学特性、典型的表面化学调控策略 (氧化还原、羧基化、氨基化、有机小分子修
饰、聚合物修饰、多肽 /蛋白修饰、核酸修饰和纳米颗粒修饰 ), 以及不同表面修饰引起的生物效应 . 继而 , 重点总结了
氧化石墨烯表面修饰影响其生物效应的主要界面作用机理 , 包括蛋白冠形成、细胞膜损伤、膜受体作用与氧化应激损
伤. 最后 , 针对氧化石墨烯表面化学调控和生物效应与机理相关研究所面临的科学问题与挑战进行了展望 . 
关键词   氧化石墨烯 ; 表面化学修饰 ; 纳米 -生物界面 ; 作用机理 ; 生物效应  
Surface Chemical Modifications of Graphene Oxide and Interactio n 
Mechanisms at the Nano-Bio Interface 
Ma, Minghaoa,b    X u ,  M i n g *,a,b,c    Liu, Sijina,b 
(a State Key Laboratory of Envir onmental Chemistry and Ecotoxicology , Research Center for Eco-Environmental Sciences , 
Chinese Academy of Sciences , Beijing 100085 , China ) 
(b University of Chinese Academy of Sciences , Beijing 100049 , China ) 
(c School of Environment , Hangzhou Institute for Advanced Study , University of Chinese Academy of Sciences ,  
Hangzhou 310024 , China ) 
Abstract   Due to the unique physicochemical pr operties, graphene oxide has been wide ly applied in material chemistry, 
biomedical science and life science. However, here is still a great challenge to maximize the advantages of graphene oxide 
and overcome the deleterious eff ects caused by its inherent properties. For a better understanding of current status in this 
research field, recent progress in surface chemical modifications of graphene oxide and interaction mechanisms at the 
nano-bio interface has been comprehensivel y reviewed. First, the physicochemical pr operties of graphene oxide and the rep-
resentative strategies of surfa ce chemical modificati ons will be briefly in troduced, including oxidation and reduction, carbox-
ylation, amination, small organic molecu le modification, polymer m odification, peptide/protein modification, nucleic acid 
modification and nanoparticle modi fication, as well as their potential roles in mediating the graphene oxide-resulted biologi-
cal effects. Following, we will present the primary interaction mechanisms of pristine and surface-modified graphene oxide at 
the nano-bio interface, including the formation of protein coro na, cell membrane damage, membra ne receptor interaction and 
oxidative stress. Finally, the knowledge

__________

Mu, Qingxin, et al. "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets." Department of Chemical Biology & Therapeutics, St. Jude Children’s Research Hospital, 2023.:

35 1.1 ±0.2 9.1 ±7.1 3.9
PCGO1 0.86 ±0.37 1.2 ±0.6 9.6 ±7.2 5.2
PCGO2 0.42 ±0.26 1.1 ±0.3 5.2 ±3.2 3.2
aThe equivalent disk diameter data were skewed, and the Box-Cox
transformation was applied to yield a more normal distribution.24The
standard deviations of the transformed data were retransformed back
to the original data scale to obtain the reported standard deviationvalues.bAverage height was measured across the surface area of all
nanosheets.ACS Applied Materials & Interfaces Research Article
dx.doi.org/10.1021/am300253c |ACS Appl. Mater. Interfaces 2012, 4, 2259 −2266 2261
FITC-BSA was released from GO for at least 24 h (Figure S2 in
Supporting Information).
Cell Surface Adhesion of PCGO. Cellular uptake
mechanisms of nanoparticles having various shapes have beenreported. For instance, spherical nanoparticles enter cellsthrough CME, caveolae-mediated endocytosis, phagocytosis,or macropinocytosis, which all require energy.
14−16Tubular
nanoparticles enter cells through endocytosis or energy-independent direct penetration.17,18All these processes require
that nanoparticles attach to the cellular membrane beforeengulfment or insertion.19Unlike spherical or tubular nano-
particles, GO has large flat surfaces with irregular shapes.Additionally, the flexibility and folding properties of GO ’s thin
layers make them act as gauzelike shapes in biological medium.
GO has been reported to be an efficient intracellular
transporter for drug and gene delivery, indicating that it canefficiently enter cells.
2On the basis of these observations, we
hypothesized that GO adheres to the cell surface and is theninternalized.
Driven by our preliminary hypothesis, we first investigated
whether PCGO could attach to the surface of cells and in whatorientation this occurred. A model cell line C2C12 (mousemesenchymal progenitor) was selected in this study. UponSEM examinations, large PCGO pieces were frequentlyobserved adhering face to face onto the cell surface (Figure1B,C). We never observed any PCGO and cells bindingperpendicularly. On the basis of SEM observations, previousreport on nanoparticle −cell interactions, and properties of
PCGO, we speculate that the adhesion is a result of severalfactors. First, the similar curvature between the nanosheets andplasma membrane would facilitate their holding together.Second, there are multiple binding forces between them,including electrostatic and hydrophobic interactions betweennanosheets and phospholipid bilayers. Third, there could bespecific ligand −receptor interactions between proteins bound
to PCGO and membrane receptors. This factor might inducereceptor-mediated endocytosis of PCGO. On the basis of∼10 000 BSA molecules per square micrometer area of GO
(AFM studies) and the cross section of BSA ∼14×4×4 nm,
we estimate that the average BSA coverage on the grapheneoxide surface to be 43%. Therefore, although the density of
protein molecules on GO surface is high, there is still space on
GO surface to facilitate the di

__________

"Ma, Minghao, Xu, Ming, and Liu, Sijin. 'Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface.' Acta Chimica Sinica, vol. 78, 2020, pp. 877-887.":

1]. 我们近期的研究发现 , 
在血清中 , 原始态 GO会吸附大量蛋白分子 , 而经过表
面修饰后的 GO (GO-NH 2, GO-PAA, GO-PEG) 对蛋白的 
化 学 学 报 综述  
  
 
Acta Chim. Sinica 2020 , 78, 877—887 © 2020 Shanghai Institute of Organic Chemistry, Chinese Academy of Sciences http://sioc-journal.cn     883 吸附显著降低 . 并且 , 作为 GO表面蛋白冠的主要成分
之一 , 免疫球蛋白 G(IgG)会增加原始态 GO与巨噬细胞
的作用 , 促进巨噬细胞的吞噬 , 而PAA与PEG表面修饰
可以显著地降低 GO与巨噬细胞的作用[77].  
 
图7  GO及其与胎牛血清 (FBS)或BSA作用后的 AFM表征图[109] 
Figure 7  AFM images of GO, FBS- and BSA-coated GO. Reprinted 
with permission from ref. [109]. C opyright 2015 Royal Society of Chem-
istry. 
4.2  细胞膜损伤  
作为第一道防线 , 细胞膜可以阻止细胞外物质自由
进入细胞 , 保证细胞内环境的相对稳定 . 细胞膜主要由
磷脂双分子层、膜蛋白和糖类等物质组成 . 当细胞暴露
GO后, GO首先会与细胞膜发生接触 , 进而被细胞粘附
和吞噬、停留在细胞膜上或进入细胞内部 (图8)[112]. 在
上述作用过程中 , GO可能会通过物理吸附或膜表面受
体的选择性识别与细胞膜发生作用 , 进而对细胞膜的结
构和功能产生影响 . GO与细胞膜的作用机理主要包括 :  
(1)“纳米刀 (nanoknives) ”效应 . GO与细胞膜接触时 , 在
近膜区域 , 其二维片层结构的边缘会引起细胞膜局部的
穿孔 , 造成细胞膜结构的破损[113]; (2)“细胞包裹 (cell 
wrapping) ”效应 . GO可以覆盖在细胞膜表面 , 阻碍细胞
膜正常的生物功能 , 如抑制物质的跨膜传递[114,115]; (3)
“磷脂抽提 (phospholipids extraction) ”效应 . GO的sp2杂
化结构会导致其与磷脂分子产生强烈作用 , 磷脂分子会
发生重新排布 , 被GO从细胞膜中抽提出来 , 造成细胞
膜损伤[116,117]; (4)细胞膜受体作用 . GO可以通过与细胞
膜表面的受体分子发生作用 , 引起相关生物过程的紊乱
或分子表达异常 , 进而导致细胞膜的破坏[118]; (5)氧化
应激损伤 . GO可以诱导细胞产生 ROS, 引起脂质过氧
化, 并造成细胞膜损伤[119-121].  
 
图8  石墨烯类材料与细胞作用的可能机理图 . (a)细胞膜粘附 ; (b)与
磷脂双分子层结合 ; (c)跨膜作用 ; (d)细胞内化 ; (e)网格蛋白介导的内
吞作用 ; (f)内体或吞噬体的内化作用 ; (g)进入溶酶体或其它核周室 ; 
(h)进入外泌体[112] 
Figure 8  Possible interactions between gr aphene-related materials with 
cells. (a) Adhesion onto the outer surface of the cell membrane. (b) In-corporation in between the monolayer s of the plasma membrane lipid 
bilayer. (c) Translocation of membra ne. (d) Cytoplasmic internalization. 
(e) Clathrin-mediated endocytosis. (f) Endosomal or phagosomal inter-
nalization. (g) Lysosomal or other pe rinuclear compartment localization. 
(h) Exosomal localization. Reprinte d with permission from ref. [112]. 
Copyright 2014 American Association for the Advancement of Science. 
GO的表面性质会显著地影响其与细胞膜作用的过
程. 通过表面修饰调控 , 则可以增加或降低 GO与细胞
膜的作用 . 相关研究发现 , GO表面的含氧化学基团有
助于 GO与细胞膜的相互作用[122]. 相较于 rGO, 高度羟
基化的 GO(hGO) 表面的碳自由基密度更高 , 能够诱导
更强的脂质过氧化和细胞膜损伤[45]. 与之类似 , 相较于
rGO, 含有更多缺陷位点的 GO更易于从磷脂双分子层
中抽提磷脂分子 , 破坏细胞膜完整性[123]. 我们的研究
则发现 , PEG和PAA修饰可以有效地降低 GO引起的膜
形态异常、膜完整性破坏、膜流动性降低与膜电势去极化
[77]. 生物大分子修饰也可以降低 GO与细胞膜的相互
作用[109,124]. 例如 , BSA修饰可以减少  GO的有效表面
积、增加空间位阻 , 进而阻碍 GO与磷脂双分子层之间
的作用 , 降低其对磷脂双分子层的破坏[109].  
4.3  膜受体作用  
除了磷脂双分子层 , GO还可以作用于细胞膜表面
的受体分子 , 进而活化或抑制其介导的下游信号通路 , 
引起多种生物效应 , 如细胞死亡、炎症反应、激素分泌
等. 相关研究表明 , GO可以作用于巨噬细胞膜表面的
Toll样受体 (如TLR-4、TLR-9), 并激活其下游的 NF-κB
信号通路 , 引起巨噬细胞的 M1极化 , 分泌炎症因子 (如
TNF-α、IL-6)[125-127]. 此外 , 库普弗细胞 (kupffer cell) 也可
以通过上述机理被 GO诱导发生 M1极化 , 分泌 IL-1β
和TNF- α等细胞因子 , 并进一步通过 NF-κB信号通路促
进肝实质细胞分泌炎症因子 IL-6[128]. 相关动物实验证
实, 暴露 GO后, 小鼠体内被活化的巨噬细胞可以原位
招募单核 -巨噬细胞和中性粒细胞等炎性细胞 , 引发急 
化 学 学 报 综述  
 
884   http://sioc-journal.cn © 2020 Shanghai Institute of Organic Chemistry, Chinese Academy of Sciences Acta Chim. Sinica 2020 , 78, 877—887
 性炎症反应[126]. 另有研究发现 , 还原氧化石墨烯量子
点(rGOQDs) 可以通过干扰芳香烃受体 (AhR)产生生殖
毒性 , 进而导致斑马鱼胚胎出现心囊水肿、脊柱弯

__________

Huang, Jie, et al. "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy." Wiley Online Library, vol. 2577, 2012, www.wileyonlinelibrary.com. Accessed 14 July 2023.:

r confocal microscopy was used to examine the uptake of Au-GO ( Figure     7  ). When the Ca Ski 
cells pretreated with chlorpromazine were incubated with Au-GO-RBITC, the ﬂ  uorescence intensity reduced to 28.2% 
of that of the cells without chlorpromazine pretreatment. For incubation of the cells pretreated with amiloride then with Au-GO-RBITC, the ﬂ  uorescence intensity reduced less 
signiﬁ  cantly, being 85.1% that of the cells in the absence of 
amiloride. The Ca Ski cells pretreated with M 
β CD then with 
Au-GO-RBITC did not signiﬁ  cantly alter their ﬂ  uorescence 
intensity, being 88.9% of that for the cells in the absence of M 
β CD. Furthermore, cells pretreated with NaN 3  showed a 
remarkable decrease in ﬂ  uorescence intensity. All these ﬂ  u-
orescence imaging results strongly suggest that the principal mechanism of cellular uptake of the Au-GO is clathrin-mediated endocytosis, and is energy dependent. This is in good consistency with the conclusion drawn from the SERS experiments. 
  
 To understand how the Au NPs affect the cellular uptake 
behavior of Au-GO, we examined the cellular uptake of GO alone in the presence of inhibitors. In the experiment, GO was ﬁ  rst labeled with RBITC. Next, the cells were either 
untreated or treated with the endocytotic inhibitors for 1 h, then incubated with GO-RBITC. Laser confocal micro-scopy was used to examine the uptake of GO (Figure S6, Supporting Information). When the cells preincubated with chlorpromazine or NaN 
3  were treated with GO-RBITC, the 
ﬂ uorescence became signiﬁ  cantly weaker than that of the 
cells not treated with inhibitors. When the cells pretreated with amiloride or M 
β CD were incubated with GO-RBITC, the ﬂ  uorescence intensity had no obvious change compared 
to that of the cells not treated with inhibitors. The results sug-gest that the principal mechanism of cellular uptake of GO is a clathrin-mediated endocytosis, and is energy dependent, like Au-GO. Clearly, the Au NPs attached to GO in our work did not affect the cell entry mechanism. It is well known that clathrin-mediated endocytosis occurs for foreign materials with sizes ranging from 100 to 200 nm, 
[  35  ,  36  ]  caveolae occur 
for materials of 50–80 nm, and macropinocytosis occurs for larger materials (500–800 nm). The major route for endo-cytosis is clathrin-dependent endocytosis, which is found in virtually all cells. 
[  37  ]  The average size of GO used in our 
experiments is 100–200 nm, as evidenced by the AFM image (Figure S1, Supporting Information), which ﬁ  ts the require-
ment for clathrin-mediated cell entry. However, the cell entry of GO through caveolae or both clathrin and caveolae mechanisms could not be completely excluded. An inhibi-tion effect was observed, although small, for cells pretreated by amiloride in the SERS and ﬂ  uorescence imaging results. 
At the present stage, it is very hard to clarify whether the difference in the SERS and ﬂ  uorescence imaging results 
between th

__________

Yue, Hua, et al. "The Role of the Lateral Dimension of Graphene Oxide in the Regulation of Cellular Responses." National Key Laboratory of Biochemical Engineering, Institute of Process Engineering, Chinese Academy of Sciences, 2012.:

 e.g.proteins, nucleic acids, and drug entities) [7]. Water-
insoluble anti-cancer drugs ( e.g. hydroxycamptothecin andpaclitaxel) are readily adsorbed viastrong hydrophobic and
p-
stacking interactions [8], providing a novel strategy for efﬁ cient
delivery. Small biological or chemical molecules can potentially beinserted between graphene sheets [9], which will further expand its
range of uses. Albeit promising, native graphene is subject to poorsolubility and high aggregation, which has hampered its biological
application. The functionalization of graphene has therefore been
exploited, and graphene oxide (GO) is becoming a favored form
[10,11] , which can be adequately dispersed in water, and allows
further functionalization because of its carboxylic groups.
With the immense potential of GO in a very broad future, it is
essential to investigate its interaction with cell types that are popu-lous in the body ( e.g.blood cells) and likely to interact with foreign
materials [12e14]. Generally, phagocytes ( e.g.macrophages) and
non-phagocytic cells ( e.g.endothelial and tumor cells) are the two
major cell types involved in biological response to exogenous GO.While macrophages are playing a key role in the non-speci ﬁc defense
(viaactive phagocytosis or cytokines release), non-phagocytic cells
are often correlated with tissue impairments and cancerous diseases.
Regarding the potential of GO, a number of groups devoted their
efforts to GO-based materials in drug delivery (to tumor tissues),
photo-thermal therapy, and gene delivery [8,15e17]. These pilot
*Corresponding authors. Tel./fax: þ86 10 82627072.
E-mail addresses: ghma@home.ipe.ac.cn (G. Ma), zgsu@home.ipe.ac.cn (Z. Su).
1Both authors contributed equally to this work.
Contents lists available at SciVerse ScienceDirect
Biomaterials
journal homepage: www.elsevi er.com/locate/biomaterials
0142-9612/$ esee front matter /C2112012 Elsevier Ltd. All rights reserved.
doi:10.1016/j.biomaterials.2012.02.021Biomaterials 33 (2012) 4013 e4021studies provided impetus on targeted therapeutic as well as diag-
nostic platforms. However, detailed information about cellular
responses to the exogenous GO is still unavailable, which not only
hinders the fabrication of graphene-based nano-devices but also
delays biological approaches for mechanistic studies.
It has been accepted that the physicochemical properties, espe-
cially with aspect to size, can regulate cellular responses to materials,
and relevant information is invaluable for their design in biomedical
area [18e20]. Material size is known to in ﬂuence the cellular inter-
nalization, which in turn dictates the microenvironments thatnanomaterials experience. Therefore, tuning the size of engineered
biomaterials is accessible to achieve high cells, tissues, or even
subcellular organelles targeting. Unfortunately, how the size of GO
with novel 2D structure (with nanometer scale thickness) affects
cellular response is poorly understood, which needs to be addres

__________

Huang, Jie, et al. "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy." Wiley Online Library, vol. 2577, 2012, www.wileyonlinelibrary.com. Accessed 14 July 2023.:

2577 © 2012 Wiley-VCH Verlag GmbH & Co. KGaA, Weinheim wileyonlinelibrary.com
  1. Introduction 
 In recent years, an increasing interest in the biological and 
medical applications of graphene oxide (GO), such as drug/gene delivery, cancer therapy, biosensing, and cellular imaging, has emerged owing to its unique structure and intrinsic prop-erties. 
[  1–5  ]  Although much progress has been made on appli-
cations of GO in the biomedical ﬁ  eld, little is known about 
the mechanism of cellular uptake and intracellular pathway of GO. 
[  6  ,  7  ]  
 To this end, we have designed and prepared a conjugate 
(Au-GO) of GO and Au nanoparticles (NPs) and studied the cellular uptake of Au-GO by means of the surface-enhanced Raman scattering (SERS) technique. Here, the Au NPs served as SERS-active substrate and GO as a vehicle for loading and delivery of Au NPs into cells. The intrinsic Mechanism of Cellular Uptake of Graphene Oxide 
Studied by Surface-Enhanced Raman Spectroscopy
  Jie   Huang  ,     Cheng   Zong  ,     He   Shen  ,     Min   Liu  ,     Biao   Chen  ,     Bin   Ren  ,   *      and   Zhiju n   Zhang   *   
Raman signals of GO inside cells were examined to reveal 
the cellular uptake behavior of Au-GO. In a previous paper, [  8  ]  
we prepared Au-GO by assembly of 2-mercaptopyridine-modiﬁ  ed Au NPs onto the GO surface, and observed two 
strong peaks at 1330 and 1600 cm 
 − 1 , characteristic of D and G 
bands of graphene, [  9  ]  in the SERS spectrum of GO. In addi-
tion, we showed that introduction of Au NPs onto the GO surface led to a signiﬁ  cantly enhanced Raman signal of probe 
molecules, compared to that for isolated Au NPs. 
[  8  ]  In this 
work, we investigated the cellular uptake of Au-GO by moni-toring the intrinsic Raman signal of GO in live cells. To eluci-date the entry mechanism, we examined the entry of Au-GO into cells pretreated with several types of endocytic inhibi-tors which selectively block speciﬁ  c uptake pathways. 
[  10  ,  11  ]  
We demonstrate that the SERS technique is very useful for studying the cellular uptake behavior of GO. By means of the SERS technique combined with ﬂ  uorescence microscopy 
and transmission electron microscopy (TEM), we conclude that cell entry of the Au-GO is mainly via energy-dependent, clathrin-mediated endocytosis. 
   2. Results and Discussion 
  2.1. Synthesis of Au-GO 
 In our previous paper, we formed Au-GO via assembly of 
2-mercaptopyridine-modiﬁ  ed Au NPs onto GO sheet via 
 π – π  stacking, and found that Au-GO exhibited a signiﬁ  cantly  DOI: 10.1002/smll.201102743  The last few years have witnessed rapid development of biological and medical 
applications of graphene oxide (GO), such as drug/gene delivery, biosensing, and bioimaging. However, little is known about the cellular uptake mechanism and pathway of GO. In this work, surface-enhanced Raman scattering (SERS) spectroscopy is employed to investigate the cellular internalization of GO loaded with Au nanopar

__________

"Ma, Minghao, Xu, Ming, and Liu, Sijin. 'Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface.' Acta Chimica Sinica, vol. 78, 2020, pp. 877-887.":

, cell membrane damage, membra ne receptor interaction and 
oxidative stress. Finally, the knowledge gaps  and future challenges in this research field will be detailedly discussed. 
Keywords   graphene oxide; surface chemical modification; nano-bio interface; interaction mechanism; biological effect  
   
1  引言 
2004年, Geim等[1]首次通过机械剥离法制得单层石
墨烯 . 石墨烯 (graphene) 是一种二维纳米材料 , 具有类
似蜂巢的六元环平面网状结构 , 厚度仅为 0.3354 nm[1]. 
在石墨烯的纳米片层结构中 , 碳原子发生了 sp2杂化 , 
每个碳原子通过键长为 142 nm的σ键与其它三个碳原
子连接 , 并在与片层垂直的方向形成离域大 π键[2,3]. 这些结构特征赋予了石墨烯优异的机械强度、化学稳定
性、高比表面积、导电和导热性能[4-7]. 因此 , 石墨烯在
能源、材料、健康、环境等方面都展现了广阔的应用前景.  
石墨烯拥有多种衍生物 , 包括氧化石墨烯 (graphene 
oxide, GO) 、还原氧化石墨烯 (reduced graphene oxide, 
rGO)、氟化石墨烯 (fluorographene) 等. 不同类型的石墨DOI: 10.6023/A20060216  
化 学 学 报 综述  
 
878   http://sioc-journal.cn © 2020 Shanghai Institute of Organic Chemistry, Chinese Academy of Sciences Acta Chim. Sinica 2020 , 78, 877—887
 烯衍生物具备各自独特的物理化学性质 , 并可应用于生
物传感器、成像探针、药物载体、抗菌材料等方面 . 例
如, 经过表面功能化的 GO可用于捕获循环肿瘤细胞[8]、
递送抗肿瘤药物[9]、杀伤细菌与病毒[10-12]. 氟化石墨烯
则可作为磁共振成像和光声成像的造影剂[13].  
作为最重要的一类石墨烯衍生物 , GO与石墨烯具
有类似的二维纳米片层结构 , 并可通过石墨烯的氧化反
应制得 . 虽然氧化过程会破坏石墨烯原有的高度共轭的
化学结构 , 但也引入了大量的化学活性位点 . 因此 , GO
纳米片层边缘与表面具有丰富的含氧官能团 , 可以用于
调控其表面化学性质 , 并进行特定的功能化设计 . 同时 , 
由于表面性质的改变 , 相较于石墨烯 , GO在生理条件
下具有更好的分散性、胶体稳定性和生物相容性 . 这些
特性使得 GO在环境健康与疾病治疗领域受到比其它石
墨烯类材料更广泛的关注[14,15]. 尽管如此 , 针对 GO的
表面化学调控与生物作用机理的研究依然有限 , 存在诸
多重要的科学问题需要解答 . 例如 , 如何通过合适的化
学调控手段对 GO进行表面修饰与功能化的同时 , 也改
善其生物相容性 , 降低人体健康风险[16].  
为了更好地了解当前的研究现状 , 就有必要对 GO
表面化学修饰与纳米 -生物界面作用机理的研究进展进
行系统地回顾与总结 . 基于上述目的 , 本文综述了 GO
的物理化学特性、表面化学修饰的主要策略及可能引起
的生物效应 , 以及 GO表面性质影响纳米 -生物界面作用
的关键机理 . 最后 , 我们将对该研究领域存在的主要问
题与未来挑战进行讨论与展望 .  
2  氧化石墨烯的结构与性质  
常用的 GO化学合成方法包括 Hummers 法和
Hofmann 法. 利用硝酸、氯酸钾或高锰酸钾等强氧化剂
处理石墨后 , 可以在石墨的瑕疵点位上引入含氧化学基
团[17]. 随后 , 由于含氧化学基团的亲水性 , 水分子可以
插入片层之间 , 将氧化石墨烯分散为单体[2]. 在不同的
合成方法与反应条件下 , 制备的 GO可能具有不同的横
向尺寸 (lateral size) 、碳氧比率 (C/O ratio) 、片层数等特
点(图1)[18].  
与石墨烯不同 , 含氧化学基团会影响 GO的二维纳
米片层结构与性质 . 在GO纳米片上 , 含氧化学基团结
合的 sp3杂化碳原子不规则的分布会导致二维片层发生
褶皱 . 因此 , 相较于石墨烯的片层厚度 (0.3354 nm)[1], 
GO单片层的厚度显著提高 (0.7～1.3 nm)[19-22]. 此外 , sp3
杂化碳原子还破坏了石墨烯原本的共轭结构 , 导致 GO
表面出现缺陷[23]. 目前 , 最广为接受的 GO表面化学结
构模型为 Lerf-Klinowski 模型 . 在该模型中 , 羟基
(−OH)、环氧基常出现在 GO片层内部 , 而羧基
(−COOH)、内酯 (−COO −)、羰基 (−C＝O)则出现在片层
边缘[17,24]. 由于这些含氧化学基团的存在 , 在水溶液中 , GO表面通常带有大量负电荷 . 因此 , 纳米片之间的静
电力排斥作用增加了 GO的胶体稳定性[25,26]. 例如 , 经
过超声处理并静置 3周后 , GO在水溶液、乙二醇、 N,N-
二甲基甲酰胺 (DMF)、N-甲基 -2-吡咯烷酮 (NMP)、四氢
呋喃 (THF)中仍呈现出良好的分散状态[27]. 然而 , 在高
盐或高蛋白溶液中 , 未经表面修饰的 GO则容易发生聚
沉[28,29]. 为了提高 GO在生理条件下的稳定性与生物兼
容性 , 可以通过表面化学修饰的方式对 GO进行改性 , 
代表性的方式包括氧化还原、羧基化、氨基化、有机小
分子修饰、聚合物修饰、生物分子修饰、纳米颗粒修饰等. 尽管 GO的表面化学调控方式五花八门 , 但大部分
表面修饰对 GO生物效应与作用机理的影响还缺少系统
的考察与评价 , 亟待开展更深入的科学研究 .  
 
图1  依据层数、横向尺寸、碳氧比率对石墨烯类材料进行分类的网
格图[18]  
Figure 1   Classification grid for the cate gorization of different graphene 
types according to three fundamental gr aphene based mate rials properties: 
number of graphene layers, average lateral dimension, and atomic car-
bon/oxygen ratio. Reprinted with perm ission from ref. [18]. Copyright 
2020 Wiley-VCH. 
3  氧化石墨烯的表面化学调控方法与生物效应  
GO表面含氧官能团的种类、

__________

Mu, Qingxin, et al. "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets." Department of Chemical Biology & Therapeutics, St. Jude Children’s Research Hospital, 2023.:

.; Wan, J.; Zhang, S.; Tian, B.; Zhang, Y.; Liu, Z.
Biomaterials 2012 ,33, 2206 −2214.■NOTE ADDED AFTER ASAP PUBLICATION
This paper was published on the Web on March 23, 2012.
Additional minor text corrections were added, and thecorrected version was reposted on March 27, 2012.ACS Applied Materials & Interfaces Research Article
dx.doi.org/10.1021/am300253c |ACS Appl. Mater. Interfaces 2012, 4, 2259 −2266 2266


__________

Mu, Qingxin, et al. "Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide Nanosheets." Department of Chemical Biology & Therapeutics, St. Jude Children’s Research Hospital, 2023.:

Size-Dependent Cell Uptake of Protein-Coated Graphene Oxide
Nanosheets
Qingxin Mu,†Gaoxing Su,†,‡Liwen Li,†,‡Ben O. Gilbertson,§Lam H. Yu,§Qiu Zhang,‡Ya-Ping Sun,⊥
and Bing Yan *,†,‡
†Department of Chemical Biology & Therapeutics, St. Jude Children ’s Research Hospital, Memphis, Tennessee, 38105, United States
‡School of Chemistry and Chemical Engineering, Shandong University, Jinan, China, 250100
§Department of Physics, University of Memphis, Memphis, Tennessee, 38152, United States
⊥Department of Chemistry and Laboratory for Emerging Materials and Technology Hunter Hall, Clemson University, Clemson,
South Carolina, 29634-0973, United States
*SSupporting Information
ABSTRACT: As an emerging applied material, graphene has shown
tremendous application potential in many fields, including bio-medicine. However, the biological behavior of these nanosheets,especially their interactions with cells, is not well understood. Here,we report our findings about the cell surface adhesion, subcellularlocations, and size-dependent uptake mechanisms of protein-coatedgraphene oxide nanosheets (PCGO). Small nanosheets enter cellsmainly through clathrin-mediated endocytosis, and the increase ofgraphene size enhances phagocytotic uptake of the nanosheets. Thesefindings will facilitate biomedical and toxicologic studies of graphenesand provide fundamental understanding of interactions at theinterface of two-dimensional nanostructures and biological systems.
KEYWORDS: graphene oxide nanosheets, protein binding, cell uptake, clathrin-mediated endocytosis, phagocytosis, size dependence
■INTRODUCTION
Graphene, a hexagonal carbon nanostructure similar to carbon
nanotubes and fullerene, has unique electronic, thermal, andmechanical properties, showing tremendous application
potential in fields such as electronics and biomedicine.
1,2
Graphene oxide (GO), which is oxidized graphite withenhanced aqueous solubility, has been proven to be an efficient
biosensor,
3drug carrier,4,5and photothermal cancer-killing
agent.6,7GO nanosheets are able to enter cells which renders
them to become promising candidates for intracellular deliveryof drugs and cellular imaging. However, the mechanisms of how
the emerging nanostrucutures interface with biological systems
are still largely unknown. In particular, a fundamental
understanding of its ability to penetrate cell membranes and
other biological barriers is still lacking. For instance, whether
the nanosheets parallelly attach onto cell surface or vertically
insert into cell membrane? By what manner they enter cells?
Such cellular uptake properties of nanoparticles may affect cell
signaling, proliferation, differentiation, and nanoparticle ex-cretion.
8−10Cellular uptake of nanoparticles with other shapes
has been studied.11We and other researchers previously
discovered endosomal leakage and nuclear translocation of
multiwalled carbon nanotubes.9,12However, the behavior of
sheet-shaped nanostructures has not been reported. Further-
more, 

__________

Yue, Hua, et al. "The Role of the Lateral Dimension of Graphene Oxide in the Regulation of Cellular Responses." National Key Laboratory of Biochemical Engineering, Institute of Process Engineering, Chinese Academy of Sciences, 2012.:

The role of the lateral dimension of graphene oxide in the regulation of cellular
responses
Hua Yuea,b,1, Wei Weia,1, Zhanguo Yuea,b, Bin Wanga, Nana Luoa,b, Yongjun Gaoc, Ding Mac,
Guanghui Maa,*, Zhiguo Sua,*
aNational Key Laboratory of Biochemical Engineering, Institute of Process Engineering, Chinese Academy of Sciences, P.O. Box 353, Beijing 100190, PR China
bGraduate University of the Chinese Academy of Sciences, Beijing 100049, PR China
cCollege of Chemistry and Molecular Engineering, Peking University, Beijing 100871, PR China
article info
Article history:
Received 29 November 2011Accepted 7 February 2012Available online 28 February 2012
Keywords:
CarbonCytokineCytotoxicityDrug delivery
Inﬂammationabstract
The nanomaterial graphene oxide (GO) has attracted explosive interests in various areas. However, its
performance in biological environments is still largely unknown, particularly with regard to cellularresponse to GO. Here we separated the GO sheets in different size and systematically investigated size
effect of the GO in response to different types of cells. In terms of abilities to internalize GO, enormous
discrepancies were observed in the six cell types, with only two phagocytes were found to be capable ofinternalizing GO. The 2
mm and 350 nm GO greatly differed in lateral dimensions, but equally contributed
to the uptake amount in macrophages. Similar amounts of antibody opsonization and active Fc g
receptor-mediated phagocytosis were demonstrated the cause of this behavior. In comparison with thenanosized GO, the GO in micro-size showed divergent intracellular locations and induced much strongerinﬂammation responses. Present study provided insight into selective internalization, size-independent
uptake, and several other biological behaviors undergone by GO. These ﬁndings might help build
necessary knowledge for potential incorporation of the unique two-dimensional nanomaterial asa biomedical tool, and for avoiding potential hazards.
/C2112012 Elsevier Ltd. All rights reserved.
1. Introduction
Understanding the performance of engineered micro/nano
materials in a biological context is an important issue for guiding
their biomedical applications. Typically, zero-dimensional (0D)
fullerenes and one-dimensional (1D) carbon nanotubes (CNTs)
initiated two surges, and the evaluation of their interaction with
living matter strongly voted great potentials in cancer therapy,
molecular imaging, and drug delivery [1e3]. Following fullerene and
CNTs, ultrathin but very strong two-dimensional (2D) graphenessoon draw much more attentions [4e6]and have merited the 2010
Nobel Prize in physics. Apart from the tremendous interest in elec-trical applications, graphene-based material is also an exciting
candidate for exploration in the biological context. The unique 2D
high surface area structure can potentially act as a template for cargo
molecules ( e.g.proteins, nucleic acids, and drug entities) [7]. Water-
insoluble anti-cancer drugs ( e.g. hydr


### Question 4


```python
display(Markdown("\n\n__________\n\n".join([":\n\n".join([answer4.dict()['contexts'][i]['text']['doc']['citation'], answer4.dict()['contexts'][i]['text']['text'], ]) for i in range(len(answer4.dict()['contexts']))])))
```


Huang, Jie, et al. "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy." Wiley Online Library, vol. 2577, 2012, www.wileyonlinelibrary.com. Accessed 14 July 2023.:

r confocal microscopy was used to examine the uptake of Au-GO ( Figure     7  ). When the Ca Ski 
cells pretreated with chlorpromazine were incubated with Au-GO-RBITC, the ﬂ  uorescence intensity reduced to 28.2% 
of that of the cells without chlorpromazine pretreatment. For incubation of the cells pretreated with amiloride then with Au-GO-RBITC, the ﬂ  uorescence intensity reduced less 
signiﬁ  cantly, being 85.1% that of the cells in the absence of 
amiloride. The Ca Ski cells pretreated with M 
β CD then with 
Au-GO-RBITC did not signiﬁ  cantly alter their ﬂ  uorescence 
intensity, being 88.9% of that for the cells in the absence of M 
β CD. Furthermore, cells pretreated with NaN 3  showed a 
remarkable decrease in ﬂ  uorescence intensity. All these ﬂ  u-
orescence imaging results strongly suggest that the principal mechanism of cellular uptake of the Au-GO is clathrin-mediated endocytosis, and is energy dependent. This is in good consistency with the conclusion drawn from the SERS experiments. 
  
 To understand how the Au NPs affect the cellular uptake 
behavior of Au-GO, we examined the cellular uptake of GO alone in the presence of inhibitors. In the experiment, GO was ﬁ  rst labeled with RBITC. Next, the cells were either 
untreated or treated with the endocytotic inhibitors for 1 h, then incubated with GO-RBITC. Laser confocal micro-scopy was used to examine the uptake of GO (Figure S6, Supporting Information). When the cells preincubated with chlorpromazine or NaN 
3  were treated with GO-RBITC, the 
ﬂ uorescence became signiﬁ  cantly weaker than that of the 
cells not treated with inhibitors. When the cells pretreated with amiloride or M 
β CD were incubated with GO-RBITC, the ﬂ  uorescence intensity had no obvious change compared 
to that of the cells not treated with inhibitors. The results sug-gest that the principal mechanism of cellular uptake of GO is a clathrin-mediated endocytosis, and is energy dependent, like Au-GO. Clearly, the Au NPs attached to GO in our work did not affect the cell entry mechanism. It is well known that clathrin-mediated endocytosis occurs for foreign materials with sizes ranging from 100 to 200 nm, 
[  35  ,  36  ]  caveolae occur 
for materials of 50–80 nm, and macropinocytosis occurs for larger materials (500–800 nm). The major route for endo-cytosis is clathrin-dependent endocytosis, which is found in virtually all cells. 
[  37  ]  The average size of GO used in our 
experiments is 100–200 nm, as evidenced by the AFM image (Figure S1, Supporting Information), which ﬁ  ts the require-
ment for clathrin-mediated cell entry. However, the cell entry of GO through caveolae or both clathrin and caveolae mechanisms could not be completely excluded. An inhibi-tion effect was observed, although small, for cells pretreated by amiloride in the SERS and ﬂ  uorescence imaging results. 
At the present stage, it is very hard to clarify whether the difference in the SERS and ﬂ  uorescence imaging results 
between th

__________

"Ma, Minghao, Xu, Ming, and Liu, Sijin. 'Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface.' Acta Chimica Sinica, vol. 78, 2020, pp. 877-887.":

1]. 我们近期的研究发现 , 
在血清中 , 原始态 GO会吸附大量蛋白分子 , 而经过表
面修饰后的 GO (GO-NH 2, GO-PAA, GO-PEG) 对蛋白的 
化 学 学 报 综述  
  
 
Acta Chim. Sinica 2020 , 78, 877—887 © 2020 Shanghai Institute of Organic Chemistry, Chinese Academy of Sciences http://sioc-journal.cn     883 吸附显著降低 . 并且 , 作为 GO表面蛋白冠的主要成分
之一 , 免疫球蛋白 G(IgG)会增加原始态 GO与巨噬细胞
的作用 , 促进巨噬细胞的吞噬 , 而PAA与PEG表面修饰
可以显著地降低 GO与巨噬细胞的作用[77].  
 
图7  GO及其与胎牛血清 (FBS)或BSA作用后的 AFM表征图[109] 
Figure 7  AFM images of GO, FBS- and BSA-coated GO. Reprinted 
with permission from ref. [109]. C opyright 2015 Royal Society of Chem-
istry. 
4.2  细胞膜损伤  
作为第一道防线 , 细胞膜可以阻止细胞外物质自由
进入细胞 , 保证细胞内环境的相对稳定 . 细胞膜主要由
磷脂双分子层、膜蛋白和糖类等物质组成 . 当细胞暴露
GO后, GO首先会与细胞膜发生接触 , 进而被细胞粘附
和吞噬、停留在细胞膜上或进入细胞内部 (图8)[112]. 在
上述作用过程中 , GO可能会通过物理吸附或膜表面受
体的选择性识别与细胞膜发生作用 , 进而对细胞膜的结
构和功能产生影响 . GO与细胞膜的作用机理主要包括 :  
(1)“纳米刀 (nanoknives) ”效应 . GO与细胞膜接触时 , 在
近膜区域 , 其二维片层结构的边缘会引起细胞膜局部的
穿孔 , 造成细胞膜结构的破损[113]; (2)“细胞包裹 (cell 
wrapping) ”效应 . GO可以覆盖在细胞膜表面 , 阻碍细胞
膜正常的生物功能 , 如抑制物质的跨膜传递[114,115]; (3)
“磷脂抽提 (phospholipids extraction) ”效应 . GO的sp2杂
化结构会导致其与磷脂分子产生强烈作用 , 磷脂分子会
发生重新排布 , 被GO从细胞膜中抽提出来 , 造成细胞
膜损伤[116,117]; (4)细胞膜受体作用 . GO可以通过与细胞
膜表面的受体分子发生作用 , 引起相关生物过程的紊乱
或分子表达异常 , 进而导致细胞膜的破坏[118]; (5)氧化
应激损伤 . GO可以诱导细胞产生 ROS, 引起脂质过氧
化, 并造成细胞膜损伤[119-121].  
 
图8  石墨烯类材料与细胞作用的可能机理图 . (a)细胞膜粘附 ; (b)与
磷脂双分子层结合 ; (c)跨膜作用 ; (d)细胞内化 ; (e)网格蛋白介导的内
吞作用 ; (f)内体或吞噬体的内化作用 ; (g)进入溶酶体或其它核周室 ; 
(h)进入外泌体[112] 
Figure 8  Possible interactions between gr aphene-related materials with 
cells. (a) Adhesion onto the outer surface of the cell membrane. (b) In-corporation in between the monolayer s of the plasma membrane lipid 
bilayer. (c) Translocation of membra ne. (d) Cytoplasmic internalization. 
(e) Clathrin-mediated endocytosis. (f) Endosomal or phagosomal inter-
nalization. (g) Lysosomal or other pe rinuclear compartment localization. 
(h) Exosomal localization. Reprinte d with permission from ref. [112]. 
Copyright 2014 American Association for the Advancement of Science. 
GO的表面性质会显著地影响其与细胞膜作用的过
程. 通过表面修饰调控 , 则可以增加或降低 GO与细胞
膜的作用 . 相关研究发现 , GO表面的含氧化学基团有
助于 GO与细胞膜的相互作用[122]. 相较于 rGO, 高度羟
基化的 GO(hGO) 表面的碳自由基密度更高 , 能够诱导
更强的脂质过氧化和细胞膜损伤[45]. 与之类似 , 相较于
rGO, 含有更多缺陷位点的 GO更易于从磷脂双分子层
中抽提磷脂分子 , 破坏细胞膜完整性[123]. 我们的研究
则发现 , PEG和PAA修饰可以有效地降低 GO引起的膜
形态异常、膜完整性破坏、膜流动性降低与膜电势去极化
[77]. 生物大分子修饰也可以降低 GO与细胞膜的相互
作用[109,124]. 例如 , BSA修饰可以减少  GO的有效表面
积、增加空间位阻 , 进而阻碍 GO与磷脂双分子层之间
的作用 , 降低其对磷脂双分子层的破坏[109].  
4.3  膜受体作用  
除了磷脂双分子层 , GO还可以作用于细胞膜表面
的受体分子 , 进而活化或抑制其介导的下游信号通路 , 
引起多种生物效应 , 如细胞死亡、炎症反应、激素分泌
等. 相关研究表明 , GO可以作用于巨噬细胞膜表面的
Toll样受体 (如TLR-4、TLR-9), 并激活其下游的 NF-κB
信号通路 , 引起巨噬细胞的 M1极化 , 分泌炎症因子 (如
TNF-α、IL-6)[125-127]. 此外 , 库普弗细胞 (kupffer cell) 也可
以通过上述机理被 GO诱导发生 M1极化 , 分泌 IL-1β
和TNF- α等细胞因子 , 并进一步通过 NF-κB信号通路促
进肝实质细胞分泌炎症因子 IL-6[128]. 相关动物实验证
实, 暴露 GO后, 小鼠体内被活化的巨噬细胞可以原位
招募单核 -巨噬细胞和中性粒细胞等炎性细胞 , 引发急 
化 学 学 报 综述  
 
884   http://sioc-journal.cn © 2020 Shanghai Institute of Organic Chemistry, Chinese Academy of Sciences Acta Chim. Sinica 2020 , 78, 877—887
 性炎症反应[126]. 另有研究发现 , 还原氧化石墨烯量子
点(rGOQDs) 可以通过干扰芳香烃受体 (AhR)产生生殖
毒性 , 进而导致斑马鱼胚胎出现心囊水肿、脊柱弯

__________

Jiang, Tao, et al. "Dependence of Graphene Oxide (GO) Toxicity on Oxidation Level, Elemental Composition, and Size." International Journal of Molecular Sciences, vol. 22, 2021, p. 10578.:

sions, suggesting possible effects
from the contact, or the possibly internalization of GO sheets by yeast cells. Furthermore,
some vital ions, such as iron, could be transported outside the cells because of the strong
binding of iron to oxygen-functional groups on the GO surface, leading to iron deﬁciency
and inhibitory metabolism in eukaryotes (e.g., S. cerevisiae ,C. albicans , and K. pastoris ) [59].
3.2. Effect of UV Treatment and Thermal Reduction on Toxicity of GOs
It has been proven that UV treatment could induce the surface activation of GO by
the photodesorption of adsorbed molecules (e.g., O 2and H 2O) on GO [ 60]. In our study,
compared to the untreated GO, the UV-treated GO possessed higher carboxyl (C-COOH)
and carbon-carbon groups (C-C and C=C), lower epoxy (C-O-C) and hydroxyl (C-OH)
functional groups, and similar oxygen content (Table 1). These features might contribute to
the signiﬁcantly higher molecular toxicity related to DNA damage stress, protein stress,
and chemical stress. The correlation analysis between PELI1.5 totalvalues (Table 2) with
epoxy and hydroxyl groups (Table 1) indicates that they have a positive linear relation
(r= 0.65), i.e., the GO with lower epoxy and hydroxyl groups tended to have a lower
PELI1.5 totaland, thus, a higher toxicity.
The toxicity of reduced GO has also been investigated in eukaryotes. It has been
demonstrated that the reduced GO induces severe and long-lasting injury in the cells of
humans and animals [ 61,62]. Du et al. studied reduced GO toxicity on algal cells, and
summarized the modes of action that were similar to GO [ 63]. Initially, the reduced GO
enveloped the algal cells by adhering to the cell surfaces. Then, it induced a perturbationInt. J. Mol. Sci. 2021 ,22, 10578 12 of 18
of the cell wall and membrane integrity, as well as nuclear chromatin condensation. Lastly,
it increased ROS and malondialdehyde (MAD) production and inhibited antioxidant
systems, consequently inducing oxidative stress in algal cells. Despite the fact that the
toxicity mechanisms of reduced GO in algal cells have been proposed as similar to GO,
their extents of toxicity may be varied [63].
Numerous previous studies reveal contradictory conclusions regarding the compari-
son of toxicity between the reduced GO and the untreated control GO [ 30,58,63–66]. For
example, Katsumiti et al. found that reduced GO functionalized with polyvinylpyrroli-
done (PVP) (rGO-PVP) was more toxic than GO and GO-PVP in mussel due to the higher
degree of internalization and ROS generation for rGO-PVP [ 58]. In contrast, Kang et al.
demonstrated that GO had a more potent toxicological effect than reduced GO in neural
pheochromocytoma-derived pc12 cell lines, with apoptosis and cell cycle arrest as the
main toxicity pathways [ 64]. In fact, the reduction usually resulted in the smaller size of
GO because the decomposition of oxygen-containing groups also removed carbon atoms
from the carbon plane and split the GO sheets into small

__________

Jiang, Tao, et al. "Dependence of Graphene Oxide (GO) Toxicity on Oxidation Level, Elemental Composition, and Size." International Journal of Molecular Sciences, vol. 22, 2021, p. 10578.:

 determine the overrepresented (signiﬁcantly
enriched, p< 0.05) biological categories, i.e., biological processes, cellular components,
and molecular functions. Gene ontology was analyzed for the treatments with the most
activated proteins (ORFs) (PELI ORF> 1.5) for each GO (i.e., at 32 mg/L), using the whole
74 biomarkers library as the reference, and activated ORFs as the test set [46].
5. Conclusions
In conclusion, the toxicity of graphene oxides (GOs), and its dependence on oxidation
level, elemental composition, and size, were comprehensively and systematically evalu-
ated with ﬁve GOs, i.e., untreated control GO, UV-treated GO with different elemental
compositions, thermally reduced GO with a lower oxidation level, and two sonicated
GOs with smaller sizes. The results show that elemental composition and size do indeed
exert impacts on GO toxicity, while the oxidation level exhibited no signiﬁcant effects.
The UV-treated GO, with signiﬁcantly higher carbon-carbon groups and carboxyl groups
(C-COOH), showed a higher toxicity level, especially in the protein and chemical stress
categories. With the decrease in size, the toxicity level of sonicated GOs tended to increase.
We proposed that the covering and subsequent internalization of GO sheets might be the
main mode of action to yeast cells.
The comprehensive and systematic evaluation on the toxicity proﬁling and mecha-
nisms at the molecular level of untreated and treated GOs ﬁlls the knowledge gap on GO
molecular toxicity and its dependence on various physicochemical characteristics. The
derived high-resolution molecular ﬁngerprint can also serve as a screening tool to feasibly
guide GO preparation, treatment, and risk management. Furthermore, the generated data
can direct the development of the prototypic quantitative structure-activity relationship
(QSAR) model with hierarchic structures to predict GO toxicity, which integrates the cur-
rent QSAR framework with bioassay data by correlating GO descriptors with toxicity
endpoints at both the molecular and phenotypic levels.
Supplementary Materials: The following are available online at https://www.mdpi.com/article/10
.3390/ijms221910578/s1.
Author Contributions: Conceptualization, T.J., C.D.V . and A.Z.G.; Methodology, T.J., C.A.A., Y.L.,
N.G., J.L., C.D.V . and A.Z.G.; Software, T.J. and S.M.R.; Validation, T.J., Y.L., N.G., J.L., C.D.V . and
A.Z.G.; Formal Analysis, T.J., C.A.A., Y.L., N.G., J.L., C.D.V . and A.Z.G.; Data Curation, T.J., C.A.A.,
Y.L. and N.G.; Writing—Original Draft Preparation, T.J.; Writing—Review & Editing, T.J., C.D.V .,
and A.Z.G.; Visualization, T.J. and S.M.R.; Supervision, C.D.V . and A.Z.G.; Project Administration,
C.D.V . and A.Z.G.; Funding Acquisition, C.D.V . and A.Z.G. All authors have read and agreed to the
published version of the manuscript.
Funding: The authors acknowledge the ﬁnancial support from the United States National Science
Foundation (NSF, CBET-1437257, CBET-1810769, IIS-1546428) and National In

__________

Huang, Jie, et al. "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy." Wiley Online Library, vol. 2577, 2012, www.wileyonlinelibrary.com. Accessed 14 July 2023.:

herjee  ,   R. N.   Ghosh  ,   F. R.   Maxﬁ  eld  ,  Physiol. Rev.    1997  ,  77 , 
 759 .  
    [ 32 ]     S. C.   Silverstein  ,   R. M.   Steinman  ,   Z. A.   Cohn  ,  Annu. Rev. Biochem.   
 1977  ,  46 ,  669 .  
    [ 33 ]     S. L.   Schmid  ,   L. L.   Carter  ,  J. Cell Biol.    1990  ,  111 ,  2307 .  
    [ 34 ]     Y.   Xiao  ,   S. P .   Forry  ,   X. G.   Gao  ,   R. D.   Holbrook  ,   W. G.   Telford  ,   A.   Tona  , 
 J. Nanobiotechnol.    2010  ,  8 ,  13 .  
    [ 35 ]     U. S.   Huth  ,   R.   Schubert  ,   R.   Peschka-Süss  ,  J. Controlled Release   
 2006  ,  110 ,  490 .  
    [ 36 ]     N. E.   Bishop  ,  Rev. Med. Virol.    1997  ,  7 ,  199 .  
    [ 37 ]     C. Y.   Yang  ,   M. F.   Tai  ,   C. P .   Lin  ,   C. W.   Lu  ,   J. L.   Wang  ,   J. K.   Hsiao  , 
  H. M.   Liu  ,  PLoS One    2011  ,  6 ,  25524 .  
    [ 38 ]     S.   Huth  ,   J.   Lausier  ,   S. W.   Gersting  ,   C.   Rudolph  ,   C.   Plank  ,   U.   Welsch  , 
  J.   Rosenecker  ,  J. Gene Med.    2004  ,  6 ,  923 .  
    [ 39 ]     B. D.   Chithrani  ,   W. C. W.   Chan  ,  Nano Lett.    2007  ,  7 ,  1542 .  
    [ 40 ]     Z. Q.   Chu  ,   Y. J.   Huang  ,   Q.   Tao  ,   Q.   Li  ,  Nanoscale    2011  ,  3 , 
 3291 .  
    [ 41 ]     N. W. S.   Kam  ,   H. J.   Dai  ,  J. Am. Chem. Soc.    2005  ,  127 , 
 6021 .  
    [ 42 ]     G.   Frens  ,  Nat. Phys. Sci.    1973  ,  241 ,  20 .  
    [ 43 ]     W. S.   Hummers  ,   R. E.   Offeman  ,  J. Am. Chem. Soc.    1958  ,  80 , 
 1339 .  
    [ 44 ]     N. I.   Kovtyukhova  ,   P . J.   Ollivier  ,   B. R.   Martin  ,   T. E.   Mallouk  , 
  S. A.   Chizhik  ,   E. V.   Buzaneva  ,   A. D.   Gorchinskiy  ,  Chem. Mater.   
 1999  ,  11 ,  771 .  
    [ 45 ]     S.   Stankovich  ,   D. A.   Dikin  ,   G. H. B.   Dommett  ,   K. M.   Kohlhaas  , 
  E. J.   Zimney  ,   E. A.   Stach  ,   R. D.   Piner  ,   S. T.   Nguyen  ,   R. S.   Ruoff  ,  Nature    2006  ,  442 ,  282 .  
 
  Received: December 28, 2011 
 Revised: March 19, 2012Published online: May 29, 2012        [ 1 ]     Z.   Liu  ,   J. T.   Robinson  ,   X. M.   Sun  ,   H. J.   Dai  ,  J. Am. Chem. Soc.    2008  , 
 130 ,  10876 .  
     [ 2 ]     X. M.   Sun  ,   Z.   Liu  ,   K.   Welsher  ,   J. T.   Robinson  ,   A.   Goodwin  ,   S.   Zaric  , 
  H. J.   Dai  ,  Nano Res.    2008  ,  1 ,  203 .  
     [ 3 ]     X. Y.   Yang  ,   X. Y.   Zhang  ,   Z.   Liu  ,   Y. F.   Ma  ,   Y.   Huang  ,   Y. S.   Chen  ,  J. 
Phys. Chem. C     2008  ,  112 ,  17554 .  
     [ 4 ]     X. Y.   Yang  ,   X. Y.   Zhang  ,   Y. F.   Ma  ,   Y.   Huang  ,   Y. S.   Wang  ,   Y. S.   Chen  ,  J. 
Mater. Chem.    2009  ,  19 ,  2710 .  
     [ 5 ]     W. J.   Hong  ,   H.   Bai  ,   Y. X.   Xu  ,   Z. Y.   Yao  ,   Z. Z.   Gu  ,   G. Q.   Shi  ,  J. Phys. 
Chem. C    2010  ,  114 ,  1822 .  
     [ 6 ]     Y.   Wang  ,   Z. H.   Li  ,   J.   Wang  ,   J. H.   Li  ,   Y. H.   Lin  ,  Trends Biotechnol.   
 2011  ,  29 ,  205 .  
     [ 7 ]     L. Z.

__________

"Ma, Minghao, Xu, Ming, and Liu, Sijin. 'Surface Chemical Modifications of Graphene Oxide and Interaction Mechanisms at the Nano-Bio Interface.' Acta Chimica Sinica, vol. 78, 2020, pp. 877-887.":

综述 
Review  
 
 * E-mail: mingxu@rcees.ac.cn 
Received June 8, 2020; published August 3, 2020. 
Project supported by the National Natural Science Foundation of China (Nos. 21922611, 21637004, 21920102007) and the Youth Inno vation Promotion Asso-
ciation CAS (No. 2019042). 
  项目受国家自然科学基金 (Nos. 21922611, 21637004, 21920102007) 和中国科学院青年创新促进会 (No. 2019042) 资助 . 
  
Acta Chim. Sinica 2020 , 78, 877—887 © 2020 Shanghai Institute of Organic Chemistry, Chinese Academy of Sciences http://sioc-journal.cn    877 化 学 学 报 
 ACTA CHIMICA SINICA 
 
氧化石墨烯的表面化学修饰及纳米 −生物界面作用机理  
马明昊a,b    徐明 *,a,b,c    刘思金a,b 
(a中国科学院生态环境研究中心   环境化学与生态毒理学国家重点实验室   北京  100085) 
(b中国科学院大学   北京  100049) 
(c国科大杭州高等研究院   杭州  310024) 
摘要   由于具备独特的物理化学性质 , 氧化石墨烯已被广泛地应用于生命科学与人体健康等相关领域 . 然而 , 如何最
大化地发挥氧化石墨烯的优势与特点 , 并克服其自身固有性质导致的生物不良效应 , 依然是当前面临的难题 . 为更好
地了解该领域的研究现状 , 本文主要综述了近年来氧化石墨烯的表面化学调控和生物作用机理方面的最新研究进展 . 
首先 , 简要介绍了氧化石墨烯的物理化学特性、典型的表面化学调控策略 (氧化还原、羧基化、氨基化、有机小分子修
饰、聚合物修饰、多肽 /蛋白修饰、核酸修饰和纳米颗粒修饰 ), 以及不同表面修饰引起的生物效应 . 继而 , 重点总结了
氧化石墨烯表面修饰影响其生物效应的主要界面作用机理 , 包括蛋白冠形成、细胞膜损伤、膜受体作用与氧化应激损
伤. 最后 , 针对氧化石墨烯表面化学调控和生物效应与机理相关研究所面临的科学问题与挑战进行了展望 . 
关键词   氧化石墨烯 ; 表面化学修饰 ; 纳米 -生物界面 ; 作用机理 ; 生物效应  
Surface Chemical Modifications of Graphene Oxide and Interactio n 
Mechanisms at the Nano-Bio Interface 
Ma, Minghaoa,b    X u ,  M i n g *,a,b,c    Liu, Sijina,b 
(a State Key Laboratory of Envir onmental Chemistry and Ecotoxicology , Research Center for Eco-Environmental Sciences , 
Chinese Academy of Sciences , Beijing 100085 , China ) 
(b University of Chinese Academy of Sciences , Beijing 100049 , China ) 
(c School of Environment , Hangzhou Institute for Advanced Study , University of Chinese Academy of Sciences ,  
Hangzhou 310024 , China ) 
Abstract   Due to the unique physicochemical pr operties, graphene oxide has been wide ly applied in material chemistry, 
biomedical science and life science. However, here is still a great challenge to maximize the advantages of graphene oxide 
and overcome the deleterious eff ects caused by its inherent properties. For a better understanding of current status in this 
research field, recent progress in surface chemical modifications of graphene oxide and interaction mechanisms at the 
nano-bio interface has been comprehensivel y reviewed. First, the physicochemical pr operties of graphene oxide and the rep-
resentative strategies of surfa ce chemical modificati ons will be briefly in troduced, including oxidation and reduction, carbox-
ylation, amination, small organic molecu le modification, polymer m odification, peptide/protein modification, nucleic acid 
modification and nanoparticle modi fication, as well as their potential roles in mediating the graphene oxide-resulted biologi-
cal effects. Following, we will present the primary interaction mechanisms of pristine and surface-modified graphene oxide at 
the nano-bio interface, including the formation of protein coro na, cell membrane damage, membra ne receptor interaction and 
oxidative stress. Finally, the knowledge

__________

Jiang, Tao, et al. "Dependence of Graphene Oxide (GO) Toxicity on Oxidation Level, Elemental Composition, and Size." International Journal of Molecular Sciences, vol. 22, 2021, p. 10578.:

Fortner, J.D.; Biswas, P . Graphene Oxides in Water: Correlating Morphology and Surface Chemistry with
Aggregation Behavior. Environ. Sci. Technol. 2016 ,50, 6964–6973. [CrossRef] [PubMed]
67. Schniepp, H.C.; Li, J.L.; McAllister, M.J.; Sai, H.; Herrera-Alonso, M.; Adamson, D.H.; Prud’homme, R.K.; Car, R.; Saville, D.A.;
Aksay, I.A. Functionalized single graphene sheets derived from splitting graphite oxide. J. Phys. Chem. B 2006 ,110, 8535–8539.
[CrossRef] [PubMed]
68. Hou, W.C.; Lee, P .L.; Chou, Y.C.; Wang, Y.S. Antibacterial property of graphene oxide: The role of phototransformation. Environ.
Sci.-Nano 2017 ,4, 647–657. [CrossRef]
69. Mao, J.; Guo, R.H.; Yan, L.T. Simulation and analysis of cellular internalization pathways and membrane perturbation for
graphene nanosheets. Biomaterials 2014 ,35, 6069–6077. [CrossRef]
70. Liu, W.T.; Bien, M.Y.; Chuang, K.J.; Chang, T.Y.; Jones, T.; BeruBe, K.; Lalev, G.; Tsai, D.H.; Chuang, H.C.; Cheng, T.J.; et al.
Physicochemical and biological characterization of single-walled and double-walled carbon nanotubes in biological media.
J. Hazard. Mater. 2014 ,280, 216–225. [CrossRef]
71. Wang, R.H.; Mikoryak, C.; Li, S.Y.; Bushdiecker, D.; Musselman, I.H.; Pantano, P .; Draper, R.K. Cytotoxicity Screening of
Single-Walled Carbon Nanotubes: Detection and Removal of Cytotoxic Contaminants from Carboxylated Carbon Nanotubes.
Mol. Pharm. 2011 ,8, 1351–1361. [CrossRef]
72. Nikitin, A.; Ogasawara, H.; Mann, D.; Denecke, R.; Zhang, Z.; Dai, H.; Cho, K.; Nilsson, A. Hydrogenation of single-walled
carbon nanotubes. Phys. Rev. Lett. 2005 ,95, 225507. [CrossRef] [PubMed]
73. Hohmann, S.M.; Willem, H. Yeast Stress Responses ; Springer: Berlin/Heidelberg, Germany, 2003.
74. Gasch, A.P .; Spellman, P .T.; Kao, C.M.; Carmel-Harel, O.; Eisen, M.B.; Storz, G.; Botstein, D.; Brown, P .O. Genomic expression
programs in the response of yeast cells to environmental changes. Mol. Biol. Cell 2000 ,11, 4241–4257. [CrossRef]
75. Lucau-Danila, A.; Lelandais, G.; Kozovska, Z.; Tanty, V .; Delaveau, T.; Devaux, F.; Jacq, C. Early expression of yeast genes affected
by chemical stress. Mol. Cell Biol. 2005 ,25, 1860–1868. [CrossRef]
76. Huh, W.K.; Falvo, J.V .; Gerke, L.C.; Carroll, A.S.; Howson, R.W.; Weissman, J.S.; O’Shea, E.K. Global analysis of protein localization
in budding yeast. Nature 2003 ,425, 686–691. [CrossRef]
77. Salamone, M.; Heddle, J.; Stuart, E.; Katz, M. Towards an improved micronucleus test: Studies on 3 model agents, mitomycin C,
cyclophosphamide and dimethylbenzanthracene. Mutat. Res. 1980 ,74, 347–356. [CrossRef]
78. Godon, C.; Lagniel, G.; Lee, J.; Buhler, J.M.; Kieffer, S.; Perrot, M.; Boucherie, H.; Toledano, M.B.; Labarre, J. The H 2O2stimulon in
Saccharomyces cerevisiae. J. Biol. Chem. 1998 ,273, 22480–22489. [CrossRef] [PubMed]
79. Ling, L.U.; Tan, K.B.; Lin, H.; Chiu, G.N. The role of reactive oxygen species and autophagy in saﬁngol-induced cell death. Cell
Death Dis. 2011 ,2, e129. [CrossRef]
80. Wang, H.;

__________

Huang, Jie, et al. "Mechanism of Cellular Uptake of Graphene Oxide Studied by Surface-Enhanced Raman Spectroscopy." Wiley Online Library, vol. 2577, 2012, www.wileyonlinelibrary.com. Accessed 14 July 2023.:

2577 © 2012 Wiley-VCH Verlag GmbH & Co. KGaA, Weinheim wileyonlinelibrary.com
  1. Introduction 
 In recent years, an increasing interest in the biological and 
medical applications of graphene oxide (GO), such as drug/gene delivery, cancer therapy, biosensing, and cellular imaging, has emerged owing to its unique structure and intrinsic prop-erties. 
[  1–5  ]  Although much progress has been made on appli-
cations of GO in the biomedical ﬁ  eld, little is known about 
the mechanism of cellular uptake and intracellular pathway of GO. 
[  6  ,  7  ]  
 To this end, we have designed and prepared a conjugate 
(Au-GO) of GO and Au nanoparticles (NPs) and studied the cellular uptake of Au-GO by means of the surface-enhanced Raman scattering (SERS) technique. Here, the Au NPs served as SERS-active substrate and GO as a vehicle for loading and delivery of Au NPs into cells. The intrinsic Mechanism of Cellular Uptake of Graphene Oxide 
Studied by Surface-Enhanced Raman Spectroscopy
  Jie   Huang  ,     Cheng   Zong  ,     He   Shen  ,     Min   Liu  ,     Biao   Chen  ,     Bin   Ren  ,   *      and   Zhiju n   Zhang   *   
Raman signals of GO inside cells were examined to reveal 
the cellular uptake behavior of Au-GO. In a previous paper, [  8  ]  
we prepared Au-GO by assembly of 2-mercaptopyridine-modiﬁ  ed Au NPs onto the GO surface, and observed two 
strong peaks at 1330 and 1600 cm 
 − 1 , characteristic of D and G 
bands of graphene, [  9  ]  in the SERS spectrum of GO. In addi-
tion, we showed that introduction of Au NPs onto the GO surface led to a signiﬁ  cantly enhanced Raman signal of probe 
molecules, compared to that for isolated Au NPs. 
[  8  ]  In this 
work, we investigated the cellular uptake of Au-GO by moni-toring the intrinsic Raman signal of GO in live cells. To eluci-date the entry mechanism, we examined the entry of Au-GO into cells pretreated with several types of endocytic inhibi-tors which selectively block speciﬁ  c uptake pathways. 
[  10  ,  11  ]  
We demonstrate that the SERS technique is very useful for studying the cellular uptake behavior of GO. By means of the SERS technique combined with ﬂ  uorescence microscopy 
and transmission electron microscopy (TEM), we conclude that cell entry of the Au-GO is mainly via energy-dependent, clathrin-mediated endocytosis. 
   2. Results and Discussion 
  2.1. Synthesis of Au-GO 
 In our previous paper, we formed Au-GO via assembly of 
2-mercaptopyridine-modiﬁ  ed Au NPs onto GO sheet via 
 π – π  stacking, and found that Au-GO exhibited a signiﬁ  cantly  DOI: 10.1002/smll.201102743  The last few years have witnessed rapid development of biological and medical 
applications of graphene oxide (GO), such as drug/gene delivery, biosensing, and bioimaging. However, little is known about the cellular uptake mechanism and pathway of GO. In this work, surface-enhanced Raman scattering (SERS) spectroscopy is employed to investigate the cellular internalization of GO loaded with Au nanopar

__________

Jiang, Tao, et al. "Dependence of Graphene Oxide (GO) Toxicity on Oxidation Level, Elemental Composition, and Size." International Journal of Molecular Sciences, vol. 22, 2021, p. 10578.:

; Duan, W.; Zhu, Y.M. Role of surface charge and oxidative stress
in cytotoxicity and genotoxicity of graphene oxide towards human lung ﬁbroblast cells. J. Appl. Toxicol. 2013 ,33, 1156–1164.
[CrossRef] [PubMed]
55. Qiao, Y.; An, J.C.; Ma, L.Y. Single Cell Array Based Assay for in Vitro Genotoxicity Study of Nanomaterials. Anal. Chem. 2013 ,85,
4107–4112. [CrossRef] [PubMed]
56. Lin, X.W.; Chen, L.Y.; Hu, X.; Feng, S.C.; Huang, L.; Quan, G.P .; Wei, X.; Yang, S.T. Toxicity of graphene oxide to white moss
Leucobryum glaucum. RSC Adv. 2017 ,7, 50287–50293. [CrossRef]
57. Akhavan, O.; Ghaderi, E.; Akhavan, A. Size-dependent genotoxicity of graphene nanoplatelets in human stem cells. Biomaterials
2012 ,33, 8017–8025. [CrossRef]
58. Katsumiti, A.; Tomovska, R.; Cajaraville, M.P . Intracellular localization and toxicity of graphene oxide and reduced graphene
oxide nanoplatelets to mussel hemocytes in vitro. Aquat. Toxicol. 2017 ,188, 138–147. [CrossRef]
59. Yu, Q.L.; Zhang, B.; Li, J.R.; Du, T.T.; Yi, X.; Li, M.C.; Chen, W.; Alvarez, P .J.J. Graphene oxide signiﬁcantly inhibits cell growth at
sublethal concentrations by causing extracellular iron deﬁciency. Nanotoxicology 2017 ,11, 1102–1114. [CrossRef]
60. Cheng, C.E.; Tsai, C.W.; Pei, Z.W.; Lin, T.W.; Chang, C.S.; Chien, F.S.S. UV-treated graphene oxide as anode interfacial layers for
P3HT: PCBM solar cells. J. Phys. D Appl. Phys. 2015 ,48, 255103. [CrossRef]Int. J. Mol. Sci. 2021 ,22, 10578 18 of 18
61. Zhang, Y.B.; Ali, S.F.; Dervishi, E.; Xu, Y.; Li, Z.R.; Casciano, D.; Biris, A.S. Cytotoxicity Effects of Graphene and Single-Wall
Carbon Nanotubes in Neural Phaeochromocytoma-Derived PC12 Cells. ACS Nano 2010 ,4, 3181–3186. [CrossRef]
62. Duch, M.C.; Budinger, G.R.S.; Liang, Y.T.; Soberanes, S.; Urich, D.; Chiarella, S.E.; Campochiaro, L.A.; Gonzalez, A.; Chandel, N.S. ;
Hersam, M.C.; et al. Minimizing Oxidation and Stable Nanoscale Dispersion Improves the Biocompatibility of Graphene in the
Lung. Nano Lett. 2011 ,11, 5201–5207. [CrossRef]
63. Du, S.T.; Zhang, P .; Zhang, R.R.; Lu, Q.; Liu, L.; Bao, X.W.; Liu, H.J. Reduced graphene oxide induces cytotoxicity and inhibits
photosynthetic performance of the green alga Scenedesmus obliquus. Chemosphere 2016 ,164, 499–507. [CrossRef]
64. Kang, Y.Y.; Liu, J.; Wu, J.R.; Yin, Q.; Liang, H.M.; Chen, A.J.; Shao, L.Q. Graphene oxide and reduced graphene oxide induced
neural pheochromocytoma-derived PC12 cell lines apoptosis and cell cycle alterations via the ERK signaling pathways. Int. J.
Nanomed. 2017 ,12, 5501–5510. [CrossRef] [PubMed]
65. Contreras-Torres, F.F.; Rodriguez-Galvan, A.; Guerrero-Beltran, C.E.; Martinez-Loran, E.; Vazquez-Garza, E.; Ornelas-Soto, N.;
Garcia-Rivas, G. Differential cytotoxicity and internalization of graphene family nanomaterials in myocardial cells. Mat. Sci. Eng.
C-Mater. 2017 ,73, 633–642. [CrossRef]
66. Jiang, Y.; Raliya, R.; Fortner, J.D.; Biswas, P . Graphene Oxides in Water: Correlating Morphology and Surface Chemistry wi

