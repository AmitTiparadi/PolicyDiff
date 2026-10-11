"""Build publication figures and a PDF report from measured results, never invented scores.

Run with requirements-report.txt installed, or the bundled document Python runtime.
"""
import csv
from collections import Counter
import json
import os
from pathlib import Path
from html import escape
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Flowable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission'
FIG = OUT / 'figures'
RESULT = OUT / 'results'
BLUE, GREEN, GRAY = '#325d82', '#1c7562', '#647382'


class ScreenshotDetail(Flowable):
    """Crop a screenshot within PDF layout; preserve the original full JPEG."""
    def __init__(self, path, region, scale):
        super().__init__()
        self.path, self.region, self.scale = path, region, scale
        self.width, self.height = region[2] * scale, region[3] * scale
        with PILImage.open(path) as im:
            self.source_width, self.source_height = im.size
        self.hAlign = 'CENTER'
    def draw(self):
        x,y,w,h = self.region
        self.canv.saveState()
        bounds=self.canv.beginPath(); bounds.rect(0,0,self.width,self.height)
        self.canv.clipPath(bounds,stroke=0)
        self.canv.drawImage(str(self.path), -x*self.scale, -(self.source_height-y-h)*self.scale,
                           width=self.source_width*self.scale, height=self.source_height*self.scale)
        self.canv.restoreState()


def read(name):
    return json.loads((RESULT / name).read_text(encoding='utf-8'))


def save(fig, name):
    fig.savefig(FIG / name, dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def figures(bench, stress, mt):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    names = list(bench['methods'])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(names)); width = .33
    f1 = [bench['methods'][n]['aggregate']['test']['macro_f1_active'] for n in names]
    modified = [bench['methods'][n]['aggregate']['test']['classification']['modified']['f1-score'] for n in names]
    for offset, values, label, color in [(-width/2, f1, 'Macro F1 (active labels)', BLUE), (width/2, modified, 'Modified-clause F1', GREEN)]:
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel='F1 score', title='Held-out publishers: reviewed reference agreement')
    ax.legend(loc='upper left', fontsize=8); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    save(fig, 'comparison.png')
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    times = [bench['methods'][n]['mean_warm_core_seconds'] for n in names]
    bars = ax.barh(names[::-1], times[::-1], color=[GREEN, BLUE, GRAY, GRAY])
    ax.bar_label(bars, fmt='%.3f s', padding=4)
    ax.set(xlim=(0, max(times)*1.3), xlabel='Mean of per-pair medians, three warm repeats', title='Core scoring and alignment runtime')
    ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    save(fig, 'runtime.png')
    matrix = np.array(bench['methods']['PolicyDiff']['aggregate']['test']['confusion_matrix'])
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True)!=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    labels = bench['methods']['PolicyDiff']['aggregate']['test']['labels']
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel='Predicted label', ylabel='Reference label', title='PolicyDiff confusion matrix (counts)')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=9, color='white' if normalized[i,j]>.5 else '#17232c')
    fig.colorbar(image, ax=ax, label='Fraction within reference row', shrink=.8)
    save(fig, 'confusion_matrix.png')
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    xs = np.arange(len(stress['thresholds'])); width=.32
    for shift, key, label, color in [(-width/2,'without_checks','SBERT alone',BLUE), (width/2,'with_checks','With change checks',GREEN)]:
        bars = ax.bar(xs+shift,[r['correct'][key] for r in stress['thresholds']],width,label=label,color=color)
        ax.bar_label(bars, fmt='%d / 9', padding=3)
    ax.set(xticks=xs, xticklabels=[f"Reword threshold {r['reword_threshold']:.2f}" for r in stress['thresholds']],
           ylim=(0,10), ylabel='Correct cases', title='Separate synthetic diagnostic: small consequential edits')
    ax.legend(fontsize=8); save(fig,'stress_ablation.png')
    with (RESULT / 'corpus_profile.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1,2,figsize=(7.2,2.8))
    words=[float(r['words']) for r in rows]
    axes[0].hist(words, bins=np.geomspace(min(words),max(words)*1.01,9), color=BLUE, edgecolor='white')
    axes[0].set_xscale('log')
    axes[0].set(title='Length: 25 PrivaSeer policies', xlabel='Words per policy (log scale)', ylabel='Documents')
    axes[1].hist([float(r['flesch_kincaid_grade']) for r in rows], bins=8, color=GREEN, edgecolor='white')
    axes[1].set(title='Readability distribution', xlabel='Flesch-Kincaid grade', ylabel='Documents')
    from matplotlib.ticker import MaxNLocator
    for axis in axes: axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(); save(fig,'corpus_profile.png')
    fig, ax = plt.subplots(figsize=(7.2,2.6))
    values=[mt['bleu'],mt['chrf']]
    bars=ax.bar(['SacreBLEU (intl)','chrF (character n-grams)'],values,color=[BLUE,GREEN],width=.5)
    ax.bar_label(bars,fmt='%.2f',padding=4)
    ax.set(ylim=(0,100),ylabel='Metric score (0-100)',title='Pretrained English-to-Hindi: 15 published paragraph references')
    ax.text(.5,.91,'Different metrics; neither score measures legal adequacy',transform=ax.transAxes,ha='center',fontsize=8)
    save(fig,'translation_scores.png')
    fig, ax = plt.subplots(figsize=(7.2,4.0)); ax.axis('off')
    boxes=[(.04,.72,'1. Earlier + later policy\nTXT / HTML / Markdown'),(.54,.72,'2. Clean and segment\nKeep numbers and negation'),
           (.54,.43,'3. Exact anchors + SBERT\nCosine similarity'),(.04,.43,'4. Global alignment\nOne-to-one + unmatched nodes'),
           (.04,.14,'5. Change checks and review\nSix clause categories'),(.54,.14,'6. TextRank + readability\nHindi translation + exports')]
    for x,y,label in boxes:
        ax.text(x+.21,y+.075,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.9',facecolor='#f0f4f6',edgecolor=GRAY),fontsize=10)
    for start,end in [((.44,.795),(.53,.795)),((.75,.68),(.75,.575)),((.54,.505),(.44,.505)),((.25,.39),(.25,.285)),((.44,.215),(.54,.215))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.7))
    ax.set(xlim=(0,1),ylim=(0,1)); save(fig,'workflow.png')


def report(bench, stress, mt):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyPD',fontName='Helvetica',fontSize=10.3,leading=14,spaceAfter=8,textColor=colors.HexColor('#17232c')))
    styles.add(ParagraphStyle(name='SmallPD',parent=styles['BodyPD'],fontSize=8.3,leading=11,spaceAfter=5))
    styles.add(ParagraphStyle(name='CaptionPD',parent=styles['SmallPD'],alignment=TA_CENTER,textColor=colors.HexColor(GRAY)))
    styles['Title'].fontName='Helvetica-Bold'; styles['Title'].fontSize=29; styles['Title'].leading=33; styles['Title'].textColor=colors.black
    for name in ['Heading1','Heading2']:
        styles[name].textColor=colors.black
    story, markdown=[],['# PolicyDiff\n\nIntelligent Privacy Policy Comparison and Semantic Change Analysis using NLP\n']
    def p(text,small=False):
        story.append(Paragraph(escape(text),styles['SmallPD' if small else 'BodyPD'])); markdown.append(text+'\n')
    def title(text,new=True):
        if new: story.append(PageBreak())
        story.append(Paragraph(escape(text),styles['Heading1'])); markdown.append('\n## '+text+'\n')
    def table(headers,rows,widths=None):
        matrix=[[Paragraph(escape(str(v)),styles['SmallPD']) for v in row] for row in [headers]+rows]
        t=Table(matrix,colWidths=widths or [480/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf1f3')),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#bec7cd')),('LEFTPADDING',(0,0),(-1,-1),7),
                              ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,10)])
        markdown.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        markdown.extend('| '+' | '.join(str(v).replace('|','/') for v in r)+' |' for r in rows); markdown.append('\n')
    def image(name,caption,max_height=3.1,region=None):
        path=FIG/name
        if not path.exists(): raise FileNotFoundError(path)
        with PILImage.open(path) as im: w,h=im.size
        if region:
            scale=min(480/region[2],max_height*inch/region[3])
            story.append(ScreenshotDetail(path,region,scale))
        else:
            scale=min(480/w,max_height*inch/h)
            story.append(Image(str(path),width=w*scale,height=h*scale,hAlign='CENTER'))
        story.append(Paragraph(escape(caption),styles['CaptionPD']))
        story.append(Spacer(1,5)); markdown.append(f'![{caption}](figures/{name})\n')

    story.append(Spacer(1,25)); story.append(Paragraph('PolicyDiff',styles['Title']))
    p('Intelligent Privacy Policy Comparison and Semantic Change Analysis using NLP')
    p('Project report | Implementation, reproducible experiments, comparative analysis and accessibility')
    story.append(Spacer(1,12))
    table(['Identity field','For manual entry'],[[f,'____________________________'] for f in ['Subject','TCODE','Name','PRN','Branch','Date','Faculty Name']],[130,350])
    p('Project group: Shivam Bhartiya and Amit Tiparadi. Identity fields above remain blank for submission entry.')
    story.append(Paragraph('Abstract',styles['Heading2']))
    p('PolicyDiff compares two versions of a privacy policy using exact text anchors, sentence embeddings, cosine similarity and global clause alignment. Conservative checks preserve numbers, negation and modal wording. An extractive overview, readability scores and pretrained Hindi translation make detected changes accessible through a local upload interface. Experiments compare semantic and lexical methods on seven sourced historical pairs. Results are agreements with an AI-reviewed reference set, with independent human validation explicitly pending.')
    p('Scope: engineering research prototype. The contribution integrates existing pretrained models and algorithms; it does not introduce a newly trained language model.',True)

    title('1. Methodology and system workflow')
    image('workflow.png','Figure 1. Implemented end-to-end workflow.',3.2)
    p('The user supplies earlier and later versions of the same English policy. Loading removes presentation markup and document metadata while retaining sentence content and section context. Sentence-BERT encodes intact clauses; stemming and stopword removal are confined to a separate preprocessing experiment.')
    p('Exact normalized clauses are matched as a multiset, preserving repeated and reordered clauses. Residual clauses receive a similarity matrix. Hungarian assignment maximizes the total matching reward with dummy unmatched nodes. A 0.015 section-name bonus breaks close ties; scores below the match threshold remain unmatched.')
    p('For matched clauses, competing alternatives within a 0.04 margin trigger review. Otherwise critical wording changes trigger modified, high similarity triggers likely reworded, and remaining matches trigger modified. Unmatched earlier clauses are removed; unmatched later clauses are added. These are change candidates, not determinations of legal effect.')
    p('The summary ranks changed source clauses using a TF-IDF similarity graph with PageRank-style TextRank and explicit-change priority. It extracts at most five evidence items. Flesch scores describe prose difficulty. Marian English-to-Hindi translation runs on demand for both sides of changed clauses.')

    title('2. Datasets, preprocessing and models')
    counts=bench['manifest']['pairs']
    table(['Dataset / publisher','Role','Size'],[
        ['PrivaSeer local demo','Preprocessing/readability corpus','4,000 TXT and 4,000 HTML snapshots'],
        ['PrivaSeer deterministic sample','Included reproducible corpus profile','25 loadable English policies'],
        ['Google archive','Development: threshold selection','3 version pairs'],
        ['Mozilla policy Git history','Held-out evaluation publisher','2 version pairs'],
        ['Automattic policy Git history','Held-out evaluation publisher','2 version pairs'],
        ['Google published Hindi policy','Separate MT reference evaluation','15 aligned paragraphs']],[155,205,120])
    p('PrivaSeer contains individual policy snapshots, so the demo is not treated as a historical version-pair benchmark. Google pairs use publisher effective dates. Repository pairs use the last policy commit before each year-end in 2022, 2023 and 2024; Git author timestamps are not necessarily legal effective dates. Raw files, exact commit URLs, dates and SHA-256 hashes are retained. Princeton WebTAP and model fine-tuning are optional future extensions, not implemented results.')
    table(['Technique / model','Implementation'],[
        ['Cleaning / sentence boundaries','BeautifulSoup; Unicode quote/whitespace normalization; deterministic NLTK Punkt with abbreviations.'],
        ['Preprocessing experiment','POS tagging on intact tokens before filtering; preserve numbers/negation; compare Porter stems and POS-aware WordNet lemmas.'],
        ['Semantic representation','all-MiniLM-L6-v2, cosine similarity; bounded token chunks and weighted normalized pooling for long clauses.'],
        ['Baselines and alignment','difflib SequenceMatcher, TF-IDF word/bigram cosine; shared global assignment and classification rules.'],
        ['Summary / readability','Native TextRank implementation; textstat Flesch Reading Ease and Flesch-Kincaid grade.'],
        ['Hindi / interface','Helsinki-NLP/opus-mt-en-hi (pretrained Marian); Streamlit; JSON/CSV/HTML exports.']],[155,325])
    p('Model revisions are pinned in settings.py. Runtime inference reads cached weights only; setup downloads are explicit. Token-chunk pooling trades full-document context for bounded, complete clause coverage.',True)

    title('3. Evaluation design and reproducibility')
    table(['Publisher / date pair','Partition','Earlier / later clauses'],[[r['id'],r['partition'],f"{r['old_clauses']} / {r['new_clauses']}"] for r in counts],[260,85,135])
    totals=Counter()
    for r in counts: totals.update(r['labels'])
    excluded_old=sum(len(r['excluded_old']) for r in counts); excluded_new=sum(len(r['excluded_new']) for r in counts)
    p(f"Reference coverage: {sum(totals.values())} scored correspondences, including {totals['unchanged']} exact anchors and {sum(totals.values())-totals['unchanged']} AI-reviewed non-exact rows. Labels: "+', '.join(f'{k}={v}' for k,v in sorted(totals.items()))+'.')
    p(f'{excluded_old} earlier and {excluded_new} later clauses are excluded from primary one-to-one metrics in five explicitly documented groups: merged sentences, split legal-basis lists, controller tables, translation lists and an italic heading. Their source IDs and reasons remain in the manifest. Predictions touching those IDs are excluded; mistakes outside the groups are still penalized. These exclusions narrow the claimed scope.')
    p('Non-exact correspondences were reviewed against source text with rationale, using lexical candidate retrieval rather than semantic predictions. Exact anchors are automatic. Human-label columns are blank; these scores are not independently validated ground-truth accuracy. Before a stronger research claim, two people should annotate changed clauses, resolve disagreements and record agreement. The human-reviewed mode rejects incomplete review and preserves entered labels.')
    p('Threshold search uses only Google development pairs: match ∈ {0.40, 0.50, 0.55, 0.60, 0.70}, reword ∈ {0.80, 0.88, 0.94}, with ambiguity margin 0.04. Each method receives the same grid and shared alignment/classification. Baseline methods disable critical wording checks. SBERT alone isolates the representation; PolicyDiff adds checks. A second ablation fixes both semantic methods to the PolicyDiff thresholds.')
    p('Macro F1 averages active labels, including no_pair for missed/spurious correspondences. Precision, recall and F1 are also provided per label. Alignment precision/recall score the correspondence itself. Change precision/recall/F1 score non-unchanged correspondences, separate from correct semantic categorization. Confusion counts use the union of reference and predicted keys.')
    p('Runtime covers residual scoring and alignment only: three sequential warm repeats per held-out pair, median per pair, then mean across pairs. Model load, UI, readability and translation are excluded. No statistical significance claim is made.',True)
    if (RESULT/'environment.json').exists():
        env=read('environment.json')
        p(f"Execution environment: {env['system']} {env['machine']}, {env['logical_cores']} logical cores, {env['embedding_device']} embedding device; Python {env['python']}, PyTorch {env['packages']['torch']}. Exact package versions are in the dependency lock.",True)

    title('4. Results and comparative analysis')
    table(['Method','Macro F1','Modified F1','Alignment recall','Warm core (s)'],[
        [name,f"{r['aggregate']['test']['macro_f1_active']:.3f}",f"{r['aggregate']['test']['classification']['modified']['f1-score']:.3f}",
         f"{r['aggregate']['test']['alignment_recall']:.3f}",f"{r['mean_warm_core_seconds']:.3f}"] for name,r in bench['methods'].items()],[120,80,90,100,90])
    image('comparison.png','Figure 2. Test-set label agreement and modified-clause F1.',2.65)
    image('runtime.png','Figure 3. Measured warm runtime; exact-anchor matching reduces residual inference.',1.85)
    proposed=bench['methods']['PolicyDiff']['aggregate']['test']; baseline=bench['methods']['difflib']['aggregate']['test']
    p(f"PolicyDiff macro F1 is {proposed['macro_f1_active']:.3f}, versus {baseline['macro_f1_active']:.3f} for difflib. Its modified-clause precision is {proposed['classification']['modified']['precision']:.3f} and recall {proposed['classification']['modified']['recall']:.3f}. Semantic representations improve reference agreement in this small sample, while many material edits remain missed or ambiguous.")
    p('PolicyDiff and SBERT alone have identical historical macro F1 at the selected thresholds. The fixed-threshold ablation likewise shows no incremental F1 gain from the checks on this sample. That result supports retaining the checks as explainable diagnostics, without claiming proven historical accuracy improvement.',True)

    title('5. Error analysis and controlled ablation')
    image('confusion_matrix.png','Figure 4. Row-normalized shading with raw counts. no_pair records correspondence errors.',3.5)
    p('Success example: Mozilla retention changes from four years to ten years and the vendor list expands; the system flags a modified clause. Failure example: Acoustic-to-Braze replacement in the later Mozilla pair has high semantic similarity and can be classified as reworded, despite a changed named processor. Proper-noun changes require better entity-sensitive checks.')
    p('Further failure cases are exported with source IDs, reference labels and predictions. Equivalent clause splits, merged sentences and Markdown tables can generate false additions/removals or review items. Multi-clause alignment and a richer document parser are clear next steps. Exact anchors dominate the corpus, so ordinary accuracy would obscure performance on changes.')
    image('stress_ablation.png','Figure 5. Nine authored cases, reported separately from historical evidence.',1.8)
    p('The synthetic diagnostic isolates numeric periods, ages, duration units, negation, modality, data types, paraphrase, punctuation and exact text. At the selected reword threshold, checks improve correct cases from '+str(stress['thresholds'][-1]['correct']['without_checks'])+' to '+str(stress['thresholds'][-1]['correct']['with_checks'])+' out of nine. A permissive synonym change can also cause a false positive. These constructed cases demonstrate mechanisms, not generalization.',True)

    title('6. Hindi accessibility and translation evaluation')
    table(['Evaluation item','Measured value'],[
        ['Model','Helsinki-NLP/opus-mt-en-hi; pretrained only'],
        ['Reference set','15 publisher-written Hindi paragraphs from Google, March 28, 2024'],
        ['Corpus SacreBLEU (intl)',f"{mt['bleu']:.2f}"],
        ['Corpus chrF',f"{mt['chrf']:.2f}"],
        ['Translation time including load',f"{mt['total_seconds_including_model_load']:.2f} seconds"],
        ['Reference authority','Publisher-written Hindi; alignment reviewed by AI; independent bilingual review pending']],[180,300])
    image('translation_scores.png','Figure 6. Corpus reference-overlap metrics; these are not adequacy percentages.',2.1)
    p('Paragraph alignment was inspected before translation. Evaluation settings and signatures, model revision, source hash and all reference/output pairs are exported. Hindi prose may have multiple valid translations; low overlap is not automatically an incorrect translation. BLEU/chrF also do not establish preservation of legal negation, quantities or rights.')
    p('Changed English clauses remain visible beside Hindi. Both earlier and later versions are translated. Inputs are processed in bounded chunks and output-limit failures are surfaced. The app uses the pretrained model; a separate IIT Bombay fine-tuning experiment is reported in Section 6a.')
    p('Observed demo failure: the retention translation omits or poorly renders the account-closure condition, despite preserving the number of days. This shows why numeric preservation and reference-overlap scores alone cannot establish meaning preservation. Keep the English evidence and obtain bilingual review.',True)
    image('hindi_interface.jpg','Figure 7. Retention detail from the verified local Hindi view; full screenshot supplied separately.',2.5,region=(380,85,810,380))

    title('6a. IIT Bombay fine-tuning experiment')
    ft=read('iitb_3000_seed42/comparison.json')
    cfg=read('iitb_3000_seed42/run_config.json')
    verification=read('iitb_3000_seed42/verification.json')
    p(f"The pinned English-Hindi encoder-decoder was fine-tuned on {cfg['train_size']:,} IIT Bombay pairs for {cfg['epochs']} epochs on a {cfg['device']}. Validation used {cfg['validation_size']} pairs and both models were scored on the identical {cfg['test_size']}-pair test set. This is fine-tuning of pretrained weights, not training from random initialization.")
    table(['Model','SacreBLEU (intl)','chrF'],[
        ['Pretrained',f"{ft['pretrained']['bleu']:.2f}",f"{ft['pretrained']['chrf']:.2f}"],
        ['Fine-tuned',f"{ft['fine_tuned']['bleu']:.2f}",f"{ft['fine_tuned']['chrf']:.2f}"],
        ['Difference',f"{ft['bleu_delta']:+.2f}",f"{ft['fine_tuned']['chrf']-ft['pretrained']['chrf']:+.2f}"]],[180,150,150])
    p(f"Minimum validation loss selected epoch {verification['selected_epoch']:.0f} (checkpoint 188). Initial loss was {ft['initial_validation_loss']:.4f}; epoch losses were " + ', '.join(f"{r['eval_loss']:.4f}" for r in verification['validation_epochs']) + f". Training took {ft['training_seconds']:.1f} seconds. Both decoders used four beams and a 128-token output cap; no missing-EOS outputs were recorded.")
    image('../results/iitb_3000_seed42/loss_curve.png','Figure 7a. Recorded training and validation losses from the Colab run.',2.15)
    p('Fine-tuning did not improve held-out BLEU or chrF in this run. Validation loss improved initially and rose slightly at epoch 2. Loss and decoded translation overlap measure different objectives; a lower validation loss does not guarantee better BLEU. The epoch-2 increase is consistent with possible overfitting but does not establish the cause of the test-score decrease.')
    p('Official splits, duplicate/overlap filters, a 10,000-row bounded shuffle buffer, and a 128-token length filter define this small convenience sample. Model and data revisions, selected rows, loss history, decoding outputs, and hashes are retained. BLEU/chrF were independently recomputed from the exported predictions and matched; checkpoint selection and split hashes were verified. Model weights were not included, so training/inference was not independently rerun.',True)
    p('This general-domain result cannot be compared directly with the 15-paragraph Google diagnostic. Pretraining overlap is unknown; one seed does not establish statistical significance. The pretrained model remains the app default, and fine-tuned weights require separate retention and domain evaluation before integration.',True)

    title('7. Implementation, contribution and novelty')
    p('The implemented contribution is an auditable workflow combining semantic matching, exact multiset anchors, global one-to-one assignment, conservative critical-wording checks, explicit ambiguity review and bilingual evidence. Numbers, units, negation and modal changes remain inspectable rather than disappearing during aggressive preprocessing. Exported reports retain clause IDs, sections, cleaned-text offsets and model revisions.')
    p('Sentence-BERT, TextRank, Hungarian assignment, Flesch scores and Marian translation are established techniques. Novelty is the project-specific integration and evidence-oriented interface. No claim is made that these algorithms or pretrained models were invented here.')
    image('upload_interface.jpg','Figure 8. Local interface accepts earlier and later policy uploads.',2.5)
    image('comparison_interface.jpg','Figure 9. Semantic demo with category counts and extracted before/after evidence.',2.65)
    p('Application behavior was checked through automated Streamlit tests and browser interaction, including semantic comparison and Hindi translation. Uploaded markup is escaped in highlights, CSV cells are sanitized, input size/clause limits are enforced, and missing models cause visible errors. No hosted inference endpoint receives policy text.',True)

    title('8. Corpus exploration and preprocessing evidence')
    image('corpus_profile.png','Figure 10. Convenience-sample distributions; not population estimates.',2.8)
    p('Twenty-five loadable PrivaSeer demo policies were selected in filename order and profiled for word count, numeric tokens, Flesch Reading Ease and Flesch-Kincaid grade. Rejected empty/non-English files encountered during selection are recorded. Sample documents and hashes are included for reproducibility. The full locally available demo is retained separately.')
    table(['Preprocessing choice','Reason / demonstrated behavior'],[
        ['Keep numeric tokens','Retention periods, ages and durations can carry meaning. Numeric counts are included in corpus_profile.csv.'],
        ['Tag before filtering','POS tagging sees intact tokens, punctuation and sentence context before normalization/removal.'],
        ['Map Penn tags to WordNet POS','Verb retained becomes retain; unknown pronoun POS remains us rather than u.'],
        ['Normalize Unicode punctuation','Smart quotes are handled separately from ASCII punctuation.'],
        ['Preserve semantic prose','SBERT and readability receive grammatical clause text, not stems or stopword-stripped sequences.']],[155,325])
    p('The token experiment CSV preserves token, POS tag, stem, lemma and clause ID for direct lab inspection. This replaces the standalone assignment script within the maintained project package. Corpus profiling, baseline comparison and historical evaluation have separate documented roles.')
    p('Summary quality is demonstrated with source-preserving output and a regression check that consequential numeric edits precede centrality. An independent summary-coverage study was not performed, so no ROUGE or fabricated summary-quality score is reported. Readability scores likewise describe linguistic complexity rather than policy fairness.',True)

    title('9. Reproduction, submission mapping and limitations')
    table(['Step','Windows command / artifact'],[
        ['Install','python -m venv .venv; install requirements-lock.txt with .venv/Scripts/python.exe -m pip'],
        ['Explicit setup','python -m scripts.setup_models; python -m scripts.source_data'],
        ['Run interface','python -m streamlit run app.py'],
        ['Core verification','python -m pytest -q; python -m scripts.verify_project'],
        ['Evaluation','python -m scripts.benchmark; python -m scripts.stress_eval; python -m scripts.translation_eval'],
        ['Corpus / report','python -m scripts.corpus_profile; install requirements-report.txt; python -m scripts.build_submission']],[110,370])
    p('Commands above use the project virtual-environment Python on Windows; the README supplies exact runnable commands. Historical and Hindi source snapshots, provenance, annotations, model pins and a dependency lock are included. Initial downloads need internet. Model weights and the virtual environment are excluded from the compact submission archive.')
    table(['Rubric','Evidence supplied'],[
        ['Methodology & Workflow','Section 1, workflow figure, source loader, segmentation and alignment.'],
        ['Dataset, Techniques & Models','Section 2, corpus profile, publisher provenance, pinned pretrained models.'],
        ['Implementation & Novelty','Section 7, local upload demo, Hindi evidence, documented integration contribution.'],
        ['Results & Evaluation','Sections 3-6 and 8, labelled reference set, precision/recall/F1, confusion matrix, runtime, BLEU/chrF, error cases.'],
        ['Comparative Analysis','Section 4, three baselines, shared tuning protocol, fixed-threshold and synthetic ablations.']],[155,325])
    p('Independent human annotation and bilingual adequacy review remain pending. Development changes are few, the held-out publishers are only two, one-to-many groups are excluded, and thresholds may not generalize. Word-number normalization, entity/vendor sensitivity, clause split/merge matching and stronger parsing should be evaluated on a larger independently labelled set.')
    p('The report uses blank identity fields in a neutral academic layout. The official SIT Assignment-01 Word template was not present, so its exact formatting is not claimed. The existing dataset is retained; redundant assignment and superseded scripts have been removed.',True)

    title('10. References and artifact index')
    references=[
        ('PrivaSeer corpus paper','https://aclanthology.org/2021.acl-long.532/'),
        ('PrivaSeer source repository','https://git.psu.edu/hlt-lab/PrivaSeer-Corpus'),
        ('Sentence-BERT, Reimers and Gurevych (2019)','https://aclanthology.org/D19-1410/'),
        ('all-MiniLM-L6-v2 model card','https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2'),
        ('TextRank, Mihalcea and Tarau (2004)','https://aclanthology.org/W04-3252/'),
        ('Helsinki-NLP English-to-Hindi model card','https://huggingface.co/Helsinki-NLP/opus-mt-en-hi'),
        ('Google publisher archive','https://policies.google.com/privacy/archive'),
        ('Mozilla publisher history','https://github.com/mozilla/legal-docs'),
        ('Automattic publisher history','https://github.com/Automattic/legalmattic'),
        ('Published Hindi reference','https://policies.google.com/privacy/archive/20240328?hl=hi'),
        ('IIT Bombay English-Hindi corpus','https://huggingface.co/datasets/cfilt/iitb-english-hindi'),
        ('SacreBLEU implementation and metric signatures','https://github.com/mjpost/sacrebleu')]
    for i,(name,url) in enumerate(references,1):
        story.append(Paragraph(f'{i}. {escape(name)}<br/><link href="{escape(url)}" color="#325d82">{escape(url)}</link>',styles['SmallPD']))
        markdown.append(f'{i}. [{name}]({url})\n')
    story.append(Spacer(1,10))
    table(['Location','Contents'],[
        ['policydiff/','Reusable analysis, text, model and export modules.'],
        ['scripts/','Sourcing, setup, verification, benchmark, diagnostics and report builder.'],
        ['data/benchmark/','Reviewed decisions, rationale, source text, blank human-review fields, manifest and Hindi source.'],
        ['data/historical/','Seven historical pairs with provenance.'],
        ['data/corpus_sample/','25 PrivaSeer snapshots and sample-selection provenance.'],
        ['submission/results/','Comparative metrics, translation examples, corpus profile and error cases.'],
        ['submission/figures/','Workflow, measured plots and actual browser screenshots.'],
        ['submission/DEMO_GUIDE.md','Short demonstration and viva guide.']],[155,325])
    p('All reported numeric scores are read from generated JSON/CSV outputs. The report can be rebuilt from these records. Source-linked evidence and explicit limitations support review without claiming independent validation that has not occurred.',True)
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(55,28,'PolicyDiff | Project submission report')
        canvas.drawRightString(A4[0]-55,28,str(doc.page))
    document=SimpleDocTemplate(str(OUT/'PolicyDiff_Report.pdf'),pagesize=A4,rightMargin=55,leftMargin=55,topMargin=45,bottomMargin=45,
                               title='PolicyDiff - Methodology and Evaluation',author='Shivam Bhartiya and Amit Tiparadi')
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/'PolicyDiff_Report.md').write_text('\n'.join(markdown),encoding='utf-8')
    print('Generated PolicyDiff_Report.pdf, editable Markdown and measured figures.')


def main():
    bench, stress, mt=read('benchmark_results.json'),read('stress_results.json'),read('translation_results.json')
    figures(bench,stress,mt)
    report(bench,stress,mt)


if __name__=='__main__':
    main()
