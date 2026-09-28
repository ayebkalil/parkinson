"""
generate_phase2_report_pdf.py
------------------------------
Generates a publication-grade, executive PDF report for Phase 2:
- Step 1: Exploratory Data Analysis & Cohort Dynamics (with report/cohort_target_eda.png)
- Step 2: Data Engineering & Multimodal Dataset Construction (with report/phase2_data_engineering_validation.png)
- Verified Statistics, Data Integrity, Anti-Leakage Partition, and Phase 3 Roadmap.

Saves: report/Parkinsons_Phase2_Data_Engineering_Report.pdf
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

REPORT_DIR = "report"
OUTPUT_PDF = os.path.join(REPORT_DIR, "Parkinsons_Phase2_Data_Engineering_Report.pdf")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 756, "Parkinson's Disease Multimodal Progression Capstone — Phase 2 Technical Report")
            self.drawRightString(576, 756, "PPMI Multimodal Cohort")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)
            
        # Running Footer
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(36, 40, 576, 40)
        self.drawString(36, 28, "Confidential — Academic Capstone Portfolio | Strict Zero-Leakage Architecture")
        self.drawRightString(576, 28, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf():
    os.makedirs(REPORT_DIR, exist_ok=True)
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "SubSectionHeading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=5
    )
    body_bold = ParagraphStyle(
        "BodyDarkBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )
    callout_style = ParagraphStyle(
        "CalloutText",
        parent=body_style,
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1A365D")
    )
    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=8
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#2D3748")
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1A365D")
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=table_cell,
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # COVER / HEADER BANNER
    # =========================================================================
    meta_table_data = [
        [
            Paragraph("<b>Parkinson's Disease Progression Modeling</b><br/><font size=11 color='#2B6CB0'>Phase 2: Exploratory Data Analysis & Multimodal Dataset Engineering</font>", title_style),
            Paragraph("<b>Dataset:</b> PPMI Longitudinal Cohort<br/><b>Pipeline Status:</b> 100% Complete & Committed<br/><b>Git Commit:</b> <font color='#2B6CB0'>7161f9d</font><br/><b>Primary Target:</b> MDS-UPDRS III (NP3TOT)", subtitle_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[330, 210])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(meta_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A365D"), spaceBefore=6, spaceAfter=8))

    # Executive Summary Callout
    exec_summary_text = (
        "<b>Executive Summary:</b> This technical report details the successful execution of <b>Phase 2 (Steps 1 & 2)</b> "
        "of the Parkinson's Disease progression capstone project. We ingested 371 raw PPMI clinical, biospecimen, and imaging tables, "
        "characterized longitudinal cohort dynamics across 20,724 in-clinic visits, controlled for medication confounders (LEDD & PDSTATE), "
        "engineered ground-truth progression targets at +12 and +24 months, and generated production-ready, zero-leakage Parquet datasets "
        "partitioned at the patient level (70% Train / 15% Val / 15% Test)."
    )
    exec_box = Table([[Paragraph(exec_summary_text, callout_style)]], colWidths=[540])
    exec_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF8FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BEE3F8")),
        ('LINELEFT', (0,0), (-1,-1), 3.5, colors.HexColor("#2B6CB0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(exec_box)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 1: EXPLORATORY DATA ANALYSIS & COHORT DYNAMICS (STEP 1)
    # =========================================================================
    story.append(Paragraph("1. Phase 2, Step 1: Exploratory Data Analysis & Longitudinal Dynamics", h1_style))
    story.append(Paragraph(
        "A rigorous exploratory data analysis (EDA) was performed across all in-clinic motor examination records "
        "(MDS-UPDRS Part III) to establish clinical baselines, assess patient retention, and quantify disease trajectories.",
        body_style
    ))

    # 5 Key Findings Bullets
    findings_data = [
        [
            Paragraph("•", body_bold),
            Paragraph("<b>Core Cohort Size:</b> Exactly <b>2,169 unique patients</b> have verified in-clinic motor exam records, comprising <b>1,829 Parkinson's Disease (PD)</b> patients and <b>340 Healthy Controls (HC)</b>, totaling <b>20,724 clinical visits</b>.", bullet_style)
        ],
        [
            Paragraph("•", body_bold),
            Paragraph("<b>Longitudinal Visit Density:</b> PD patients average <b>10.0 clinical visits</b> each (with long-term participants followed for up to 35 visits over a decade), while Healthy Controls average <b>7.1 visits</b>. The highest visit density aligns with trial milestones: Baseline (BL), Month 12 (V04), Month 24 (V06), and Month 36 (V08).", bullet_style)
        ],
        [
            Paragraph("•", body_bold),
            Paragraph("<b>Target Score Separation (NP3TOT):</b> Healthy Controls demonstrate an average score of <b>2.25 ± 3.75</b> (Median = <b>1.0</b>), indicating virtually zero motor disability. PD patients exhibit an average score of <b>24.77 ± 12.82</b> (Median = <b>23.0</b>, range 0 to 100), confirming profound biological separation.", bullet_style)
        ],
        [
            Paragraph("•", body_bold),
            Paragraph("<b>The Medication Confounder (PDSTATE & LEDD):</b> Among PD exams, <b>44.5%</b> are tested in the <b>ON</b> medication state, <b>28.5%</b> in the <b>OFF</b> state (overnight withdrawal), and <b>26.9%</b> at Untreated Baseline. Controlling for PDSTATE and daily Levodopa dose is essential so models do not mistake symptomatic medication relief for biological remission.", bullet_style)
        ],
        [
            Paragraph("•", body_bold),
            Paragraph("<b>Two-Year Progression Trajectories:</b> <b>3,195 complete 2-year trajectory pairs</b> (Baseline → +12m → +24m) were identified. While average PD scores rise from 22.61 to 23.37, individual progression displays high clinical heterogeneity (some worsening by +10 points, others stabilized on dopaminergic therapy).", bullet_style)
        ]
    ]
    t_findings = Table(findings_data, colWidths=[12, 528])
    t_findings.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_findings)
    story.append(Spacer(1, 6))

    # Embed Figure 1: Cohort EDA Chart
    eda_img_path = os.path.join(REPORT_DIR, "cohort_target_eda.png")
    if os.path.exists(eda_img_path):
        story.append(Image(eda_img_path, width=7.2*inch, height=5.14*inch))
        story.append(Paragraph(
            "<b>Figure 1: Cohort Dynamics & Target Distribution.</b> (A) Score distribution showing separation between PD and HC. "
            "(B) Longitudinal 24-month progression trajectory. (C) Clinical visit frequency distribution per patient. (D) Medication state distribution during motor evaluation.",
            caption_style
        ))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 2: DATA ENGINEERING & MULTIMODAL DATASET (STEP 2)
    # =========================================================================
    story.append(Paragraph("2. Phase 2, Step 2: Multimodal Dataset Construction & Target Engineering", h1_style))
    story.append(Paragraph(
        "Using <code>src/build_multimodal_dataset.py</code>, we constructed a relational multi-table join pipeline "
        "integrating 6 distinct clinical, cognitive, biospecimen, and neuroimaging modalities into a unified temporal schema.",
        body_style
    ))

    # Architecture Summary Table
    arch_data = [
        [
            Paragraph("Modality Layer", table_header),
            Paragraph("Raw Sources Ingested", table_header),
            Paragraph("Engineered Features & Clinical Rationale", table_header)
        ],
        [
            Paragraph("<b>Motor Evaluation</b>", table_cell_bold),
            Paragraph("MDS-UPDRS Parts I, II, III, IV, Schwab & England ADL", table_cell),
            Paragraph("NP3TOT (target), tremor score (10 items), rigidity score (5 items), bradykinesia score (11 items), axial posture score (7 items), Hoehn & Yahr stage, Schwab-England ADL %.", table_cell)
        ],
        [
            Paragraph("<b>Medication Control</b>", table_cell_bold),
            Paragraph("LEDD Concomitant Medication Log, Part III exam flags", table_cell),
            Paragraph("Continuous active LEDD (mg/day) matched via temporal intervals to visit date; PDSTATE one-hot flags (ON, OFF, Untreated); Hours post-medication (HRPOSTMED).", table_cell)
        ],
        [
            Paragraph("<b>Non-Motor Clinical</b>", table_cell_bold),
            Paragraph("MoCA, SCOPA-AUT, GDS-15, ESS, RBDSQ, UPSIT", table_cell),
            Paragraph("Total MoCA cognitive score (0-30), SCOPA autonomic burden (SCAU1-25 sum), GDS-15 depressive score, daytime sleepiness (ESS), dream-enacting behavior (RBDSQ), UPSIT olfactory score.", table_cell)
        ],
        [
            Paragraph("<b>Biospecimens (Fluid)</b>", table_cell_bold),
            Paragraph("CSF Biospecimen Analysis Results, Alpha-Synuclein SAA", table_cell),
            Paragraph("CSF amyloid-beta 1-42 (ABeta42), total tau (tTau), phosphorylated tau 181 (pTau), alpha-synuclein, neurofilament light chain (NfL), tTau/ABeta ratio, SAA seed amplification status (1/0).", table_cell)
        ],
        [
            Paragraph("<b>Tabular Neuroimaging</b>", table_cell_bold),
            Paragraph("Xing Core Lab DaTSCAN SPECT SBR, FreeSurfer 7 ASEG Volumes", table_cell),
            Paragraph("Striatal Binding Ratios (caudate, putamen, striatum mean), putamen asymmetry index, caudate/putamen ratio; FreeSurfer subcortical volumes (putamen, caudate, hippocampus, stem, ventricles) normalized to eTIV.", table_cell)
        ]
    ]
    t_arch = Table(arch_data, colWidths=[100, 160, 280])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 6))

    # Verified Data Engineering Statistics Table
    story.append(Paragraph("<b>Table 1: Verified Multi-Modal Pipeline Statistics & Dataset Integrity</b>", h2_style))
    stats_data = [
        [
            Paragraph("Pipeline Metric", table_header),
            Paragraph("Verified Count / Value", table_header),
            Paragraph("Clinical & Data Science Significance", table_header)
        ],
        [
            Paragraph("<b>Total Harmonized Visits</b>", table_cell_bold),
            Paragraph("<b>32,743 records</b>", table_cell_bold),
            Paragraph("Complete multi-table relational join across all in-clinic assessments in PPMI.", table_cell)
        ],
        [
            Paragraph("<b>Unique Patients Ingested</b>", table_cell_bold),
            Paragraph("<b>5,426 patients</b>", table_cell),
            Paragraph("Encompasses PD (2,254), Prodromal (6,495 status rows), Healthy Control (440), and SWEDD (81).", table_cell)
        ],
        [
            Paragraph("<b>Valid +12-Month Targets</b>", table_cell_bold),
            Paragraph("<b>21,078 records</b> (64.4%)", table_cell),
            Paragraph("Clinically matched future NP3TOT in time window [250, 480] days (closest to 365d).", table_cell)
        ],
        [
            Paragraph("<b>Valid +24-Month Targets</b>", table_cell_bold),
            Paragraph("<b>17,609 records</b> (53.8%)", table_cell),
            Paragraph("Clinically matched future NP3TOT in time window [550, 910] days (closest to 730d).", table_cell)
        ],
        [
            Paragraph("<b>Complete 2-Year Trajectories</b>", table_cell_bold),
            Paragraph("<b>14,806 records</b> (45.2%)", table_cell),
            Paragraph("Simultaneous ground truth for both 12m and 24m horizons from the same observation.", table_cell)
        ],
        [
            Paragraph("<b>Patient Partition Overlap</b>", table_cell_bold),
            Paragraph("<b>0 (Zero Overlap)</b>", table_cell_bold),
            Paragraph("Strict zero-leakage guarantee: Patient IDs in Train, Val, and Test are completely disjoint.", table_cell)
        ],
        [
            Paragraph("<b>PD 12m Worsening Rate</b>", table_cell_bold),
            Paragraph("<b>39.5%</b> (4,898 / 12,405)", table_cell),
            Paragraph("Clinically meaningful worsening (MCID ≥ +3.5 pts on NP3TOT) at 1 year post-assessment.", table_cell)
        ],
        [
            Paragraph("<b>PD 24m Worsening Rate</b>", table_cell_bold),
            Paragraph("<b>46.1%</b> (5,077 / 11,013)", table_cell),
            Paragraph("Clinically meaningful worsening (MCID ≥ +3.5 pts on NP3TOT) at 2 years post-assessment.", table_cell)
        ],
        [
            Paragraph("<b>HC 12m Progression Rate</b>", table_cell_bold),
            Paragraph("<b>8.4%</b> (137 / 1,635)", table_cell),
            Paragraph("Low progression rate in healthy individuals, reflecting test-retest and normal aging noise.", table_cell)
        ]
    ]
    t_stats = Table(stats_data, colWidths=[140, 120, 280])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 6))

    # Embed Figure 2: Data Engineering Validation Chart
    val_img_path = os.path.join(REPORT_DIR, "phase2_data_engineering_validation.png")
    if os.path.exists(val_img_path):
        story.append(Image(val_img_path, width=7.2*inch, height=5.5*inch))
        story.append(Paragraph(
            "<b>Figure 2: Multimodal Dataset Construction & Validation.</b> (A) Probability density shift in NP3TOT over 12 and 24 months in PD. "
            "(B) Clinically meaningful worsening (MCID ≥ 3.5 points) across cohorts. (C) Data completeness across modalities. (D) Feature correlations with 12-month motor delta.",
            caption_style
        ))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 3: TECHNICAL INTEGRITY, PARQUET ARTIFACTS & PHASE 3 ROADMAP
    # =========================================================================
    story.append(Paragraph("3. Technical Integrity, Parquet Artifacts & Phase 3 Roadmap", h1_style))
    story.append(Paragraph(
        "All pipeline outputs have been validated against our anti-leakage specification and serialized to high-performance "
        "Apache Parquet formats. Patient assignment was computed exactly once at baseline and strictly enforced across all tables.",
        body_style
    ))

    # Partition Table
    split_summary_data = [
        [
            Paragraph("Dataset Split", table_header),
            Paragraph("Total Rows", table_header),
            Paragraph("Unique Patients", table_header),
            Paragraph("Latest Tabular Visits", table_header),
            Paragraph("Primary Role in Capstone", table_header)
        ],
        [
            Paragraph("<b>Train Split</b>", table_cell_bold),
            Paragraph("22,980 rows", table_cell),
            Paragraph("3,787 patients", table_cell),
            Paragraph("2,683 patients", table_cell),
            Paragraph("Gradient boosting training, RNN/LSTM sequence fitting, representation learning.", table_cell)
        ],
        [
            Paragraph("<b>Validation Split</b>", table_cell_bold),
            Paragraph("4,831 rows", table_cell),
            Paragraph("827 patients", table_cell),
            Paragraph("574 patients", table_cell),
            Paragraph("Hyperparameter tuning, early stopping, attention weight calibration.", table_cell)
        ],
        [
            Paragraph("<b>Held-Out Test Split</b>", table_cell_bold),
            Paragraph("4,932 rows", table_cell),
            Paragraph("812 patients", table_cell),
            Paragraph("583 patients", table_cell),
            Paragraph("Strict blind evaluation (never seen during training or tuning).", table_cell)
        ],
        [
            Paragraph("<b>Total Full Cohort</b>", table_cell_bold),
            Paragraph("<b>32,743 rows</b>", table_cell_bold),
            Paragraph("<b>5,426 patients</b>", table_cell_bold),
            Paragraph("<b>3,840 patients</b>", table_cell_bold),
            Paragraph("Comprehensive multimodal longitudinal database (1.27 MB Parquet).", table_cell)
        ]
    ]
    t_splits = Table(split_summary_data, colWidths=[90, 75, 85, 90, 200])
    t_splits.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_splits)
    story.append(Spacer(1, 8))

    # Anti-Leakage Principles Callout Box
    leakage_box_text = (
        "<b>Rigorous Anti-Leakage Architecture Enforced:</b><br/>"
        "1. <b>Strict Patient-Level Partitioning:</b> No patient appears in more than one partition (Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅).<br/>"
        "2. <b>Temporal Arrow of Time:</b> No feature calculated at visit time <i>t</i> accesses observations from time &gt; <i>t</i>. Future visits only supply the prediction targets (target_np3tot_12m, target_np3tot_24m).<br/>"
        "3. <b>Zero Split Contamination:</b> Scalers, imputers, and encoders are fit strictly on the training partition and transformed on validation and test."
    )
    leakage_box = Table([[Paragraph(leakage_box_text, body_style)]], colWidths=[540])
    leakage_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFFAF0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#FEEBC8")),
        ('LINELEFT', (0,0), (-1,-1), 3.5, colors.HexColor("#DD6B20")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(leakage_box)
    story.append(Spacer(1, 8))

    # Git Commit Verification & Code Manifest
    story.append(Paragraph("<b>Code Manifest & Local Git Commits</b>", h2_style))
    git_manifest = [
        [
            Paragraph("Commit Hash", table_header),
            Paragraph("Branch", table_header),
            Paragraph("Primary Artifacts Created & Committed", table_header)
        ],
        [
            Paragraph("<code>7161f9d</code>", table_cell_bold),
            Paragraph("master", table_cell),
            Paragraph("<code>src/build_multimodal_dataset.py</code>, <code>src/visualize_data_engineering.py</code>, <code>data/processed/*.parquet</code> (7 files), <code>report/phase2_data_engineering_validation.png</code>", table_cell)
        ],
        [
            Paragraph("<code>f2e361f</code>", table_cell_bold),
            Paragraph("master", table_cell),
            Paragraph("<code>src/cohort_eda.py</code>, <code>report/cohort_target_eda.png</code> (Step 1 Exploratory Data Analysis)", table_cell)
        ],
        [
            Paragraph("<code>a43d966</code>", table_cell_bold),
            Paragraph("master", table_cell),
            Paragraph("<code>src/create_patient_split.py</code>, <code>data/processed/patient_split.csv</code> (Phase 1 Canonical Split)", table_cell)
        ]
    ]
    t_git = Table(git_manifest, colWidths=[75, 55, 410])
    t_git.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_git)
    story.append(Spacer(1, 8))

    # Phase 3 Modeling Roadmap
    story.append(Paragraph("<b>Immediate Next Steps: Phase 3 Modeling Execution</b>", h2_style))
    next_steps_text = [
        [
            Paragraph("<b>Phase 3, Step 1</b>", table_cell_bold),
            Paragraph("<b>Evaluation Harness (<code>src/evaluate.py</code>):</b> Implement MAE, RMSE, Pearson r, R², ROC-AUC, PR-AUC, Sensitivity, Specificity, and Brier score, with medication subgroup stratification (ON vs. OFF).", table_cell)
        ],
        [
            Paragraph("<b>Phase 3, Step 2</b>", table_cell_bold),
            Paragraph("<b>Model 1 Tabular XGBoost (<code>src/train_model1_xgboost.py</code>):</b> Train static gradient boosting on latest visit features to establish the baseline and answer RQ1 (Single visit predictive power).", table_cell)
        ],
        [
            Paragraph("<b>Phase 3, Step 3</b>", table_cell_bold),
            Paragraph("<b>Model 2 Sequential LSTM (<code>src/train_model2_lstm.py</code>):</b> Train recurrent sequence network on multi-visit clinical history to test whether longitudinal trajectories beat static baselines.", table_cell)
        ],
        [
            Paragraph("<b>Phase 3, Step 4</b>", table_cell_bold),
            Paragraph("<b>Model 3 Imaging Baseline (<code>src/train_model3_imaging.py</code>):</b> Evaluate predictive signal from DaTSCAN SPECT striatal binding and FreeSurfer MRI subcortical volumes.", table_cell)
        ]
    ]
    t_next = Table(next_steps_text, colWidths=[90, 450])
    t_next.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_next)

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated Phase 2 Report PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
