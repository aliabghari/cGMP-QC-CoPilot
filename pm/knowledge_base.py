"""
pm/knowledge_base.py

Root-Cause Knowledge Base and Lessons-Learned Engine for cGMP-QC-CoPilot.
Provides:
- Relational storage for root causes, deviation types, lessons learned, and links.
- 40 synthetic historical deviation records across 7 biologics QC categories:
  Chromatography, Bioassay, Sample Handling, Reference Standards, Reagents, Software/Integration, Documentation.
- Phase I (Laboratory) vs Phase II (Manufacturing / Cross-Functional) investigation classification.
- TF-IDF / keyword similarity retrieval for rapid case citation during new investigations.
- Decision-support only framing: never claims definitive root cause determination.
- Append-only ALCOA+ audit trail logging for all new lessons and modifications.
"""

from __future__ import annotations

import math
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.audit import AuditTrail
from core.stamps import DRAFT_STAMP, assert_no_banned_phrases


ADVISORY_DISCLAIMER = (
    "Past case reference for decision support only. Does not replace formal laboratory "
    "or cross-functional deviation investigation. Final root-cause determination rests with QA."
)


# -------------------------------------------------------------------------
# Seed Data Definitions (>= 40 historical deviation / lesson learned cases)
# -------------------------------------------------------------------------

SEED_ROOT_CAUSES = [
    # Chromatography
    {"id": "RC-CHROM-01", "code": "RC_COL_FOUL", "name": "Column Station Fouling / In-line Filter Clog", "category": "Chromatography", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "High backpressure or peak splitting due to particulate buildup or aggregated matrix foul on analytical column."},
    {"id": "RC-CHROM-02", "code": "RC_DEGASS_FAIL", "name": "Mobile Phase Degasser Cavitation / Air Bubble", "category": "Chromatography", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Baseline drift, pressure ripples, or retention time shift from micro-bubbles in quaternary proportioning valve."},
    {"id": "RC-CHROM-03", "code": "RC_SEAL_LEAK", "name": "Autosampler Needle / High-Pressure Rotor Seal Leak", "category": "Chromatography", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Injection area reproducibility failure or sample carryover caused by worn rotor seal."},
    {"id": "RC-CHROM-04", "code": "RC_TEMP_DRIFT", "name": "Column Compartment Peltier Temperature Fluctuation", "category": "Chromatography", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Retention time shift or selectivity change from thermocouple calibration drift in column oven."},

    # Bioassay
    {"id": "RC-BIO-01", "code": "RC_CELL_PASS", "name": "Cell Line Passage Number Exceedance / Phenotypic Drift", "category": "Bioassay", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Bioassay dose-response slope failure or reduced signal-to-noise ratio due to cell over-passage."},
    {"id": "RC-BIO-02", "code": "RC_EDGE_EFF", "name": "Microplate Evaporative Edge Effect / Thermal Gradient", "category": "Bioassay", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Perimeter well response deviation caused by uneven incubator airflow or non-humidified plate chambers."},
    {"id": "RC-BIO-03", "code": "RC_CO2_DRIFT", "name": "CO2 Incubator Sensor Calibration Drift", "category": "Bioassay", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Medium pH shift during cell seeding incubation leading to poor assay parallelism."},
    {"id": "RC-BIO-04", "code": "RC_SERUM_LOT", "name": "Fetal Bovine Serum (FBS) Lot-to-Lot Binding Variation", "category": "Bioassay", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Unqualified reagent lot introducing non-specific reporter protein induction."},

    # Sample Handling
    {"id": "RC-SMPL-01", "code": "RC_VORTEX_SHEAR", "name": "Vigorous Vortex Agitation / Shear Degradation", "category": "Sample Handling", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "High molecular weight (HMW) aggregate formation induced by shear stress during violent tube mixing."},
    {"id": "RC-SMPL-02", "code": "RC_FRZ_THAW", "name": "Excessive or Uncontrolled Freeze-Thaw Thermal Cycling", "category": "Sample Handling", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Protein sub-visible particle formation or sub-potency following slow room-temperature bench thawing."},
    {"id": "RC-SMPL-03", "code": "RC_LIGHT_EXP", "name": "Unshielded Ambient Light Exposure of Photolabile Protein", "category": "Sample Handling", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Tryptophan photo-oxidation or acidic variant increase after prolonged bench exposure in clear glass."},

    # Reference Standards
    {"id": "RC-STD-01", "code": "RC_EXP_STD", "name": "Working Reference Standard Expiration / Thaw Expiry", "category": "Reference Standards", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Calibration curve failure or potency drift from thawed standard vial stored beyond stability limit."},
    {"id": "RC-STD-02", "code": "RC_HYGRO_MOIST", "name": "Hygroscopic Moisture Pickup in Lyophilized Standard", "category": "Reference Standards", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Under-recovery or apparent potency inflation due to uncorrected vial headspace moisture ingress."},

    # Reagents & Media
    {"id": "RC-RGT-01", "code": "RC_BUFF_PH", "name": "Mobile Phase / Formulation Buffer pH Meter Drift", "category": "Reagents", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Uncalibrated pH electrode resulting in buffer prepared 0.3 units off nominal ion-exchange setpoint."},
    {"id": "RC-RGT-02", "code": "RC_WTR_BACT", "name": "Milli-Q Water System TOC / Microbial Breakdown", "category": "Reagents", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Ghost peaks in reverse-phase gradient chromatography tracing to bio-fouled water polishing loop."},

    # Software & CDS Integration
    {"id": "RC-SOFT-01", "code": "RC_CDS_INTEG", "name": "CDS Integration Baseline Skipping / Rider Peak Missed", "category": "Software/Integration", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Inappropriate slope sensitivity setting causing shoulder peak to be co-integrated with main monomer."},
    {"id": "RC-SOFT-02", "code": "RC_LIMS_SMPL", "name": "LIMS Dilution Factor / Specimen Accessioning Typo", "category": "Software/Integration", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Manual entry of 1:100 instead of 1:10 dilution in LIMS sample login resulting in apparent OOS result."},

    # Documentation & Method Compliance
    {"id": "RC-DOC-01", "code": "RC_SOP_MISREAD", "name": "Analyst Execution Divergence from SOP Step", "category": "Documentation", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Omission of mandatory 5-minute pre-equilibration hold prior to chromatographic injection series."},
    {"id": "RC-DOC-02", "code": "RC_CALC_ERR", "name": "Spreadsheet Transcription / Formula Bracket Error", "category": "Documentation", "investigation_tier": "Phase I (Laboratory Investigation)", "description": "Manual calculation sheet cell referencing wrong blank subtraction cell in non-validated workbook."},

    # Phase II / Cross-Functional / Manufacturing
    {"id": "RC-MFG-01", "code": "RC_BIO_HARV", "name": "Bioreactor Harvest Clarification Depth Filter Breakthrough", "category": "Manufacturing/Process", "investigation_tier": "Phase II (Cross-Functional / Manufacturing)", "description": "High turbidity and host-cell protein load overwhelming downstream purification columns."},
    {"id": "RC-MFG-02", "code": "RC_UFDF_PUMP", "name": "TFF Ultrafiltration Recirculation Pump Shear Heating", "category": "Manufacturing/Process", "investigation_tier": "Phase II (Cross-Functional / Manufacturing)", "description": "Localized product temperature spike generating elevated soluble dimer prior to drug substance fill."},
]

SEED_DEVIATION_TYPES = [
    {"id": "DEV-TYP-01", "code": "DEV_SST_FAIL", "name": "System Suitability Testing (SST) Criteria Failure", "area": "Laboratory", "description": "USP <621> SST parameter (resolution, tailing, %RSD, retention factor) out of specification."},
    {"id": "DEV-TYP-02", "code": "DEV_OOS_LAB", "name": "Out of Specification (OOS) - Lab Investigated", "area": "Laboratory", "description": "Reportable test result exceeding approved release or stability specification."},
    {"id": "DEV-TYP-03", "code": "DEV_OOT_TRD", "name": "Out of Trend (OOT) / Statistical Warning", "area": "Quality Engineering", "description": "Result within spec but violating historical statistical control limits or Nelson rules."},
    {"id": "DEV-TYP-04", "code": "DEV_EQUIP_ERR", "name": "Analytical Instrument Telemetry Alarm / Abort", "area": "Hardware/Facilities", "description": "Pressure burst, leak detector trip, or optical bench failure terminating run sequence."},
    {"id": "DEV-TYP-05", "code": "DEV_MFG_EXC", "name": "Downstream Process In-Process Excursion", "area": "Manufacturing", "description": "Chromatography skid hold time exceedance or filtration flux drop during drug substance harvest."},
]


def _build_synthetic_lessons() -> List[Dict[str, Any]]:
    """Generates 40 comprehensive synthetic historical lesson records across all categories."""
    lessons = []
    methods = [
        "SEC-HPLC Monomer Purity", "CEX-HPLC Charge Variants", "rCE-SDS Reduced Purity",
        "nrCE-SDS Non-Reduced Purity", "Cell-Based Bioassay (Relative Potency)",
        "SPR / Octet Binding Affinity", "UV-Vis A280 Protein Concentration",
        "Osmolality & pH Release Testing", "Sub-visible Particulate HIAC",
        "Host Cell Protein (HCP) ELISA",
    ]
    projects = [
        "PRJ-MAB-001", "PRJ-MAB-002", "PRJ-FUS-003", "PRJ-BSPEC-004", "PRJ-ADC-005",
        "PRJ-ENZ-006", "PRJ-MAB-007", "PRJ-MAB-008", "PRJ-FUS-009", "PRJ-BSPEC-010"
    ]

    base_cases = [
        # 1-6: Chromatography
        {
            "rc_id": "RC-CHROM-01", "dev_id": "DEV-TYP-01", "cat": "Chromatography",
            "tier": "Phase I (Laboratory Investigation)", "method": "SEC-HPLC Monomer Purity",
            "what": "System suitability resolution between monomer and dimer dropped below 1.5 during Run #18 on UPLC Skid #2.",
            "learned": "Analytical column guard frit had accumulated sub-visible particulates from uncentrifuged high-concentration formulation samples.",
            "capa": "Mandated 0.22 um PVDF spin-filter centrifugation (10,000 rcf for 3 min) in sample prep procedure SOP-QC-0881 (fictional example).",
        },
        {
            "rc_id": "RC-CHROM-02", "dev_id": "DEV-TYP-04", "cat": "Chromatography",
            "tier": "Phase I (Laboratory Investigation)", "method": "CEX-HPLC Charge Variants",
            "what": "Severe baseline oscillation (0.8 mAU amplitude) and retention time drift on basic variant peaks.",
            "learned": "Degasser vacuum membrane micro-fissure allowed ambient air aspiration into eluent B channel during shallow salt gradient.",
            "capa": "Added daily vacuum chamber decay self-test to instrument prep log; replaced vacuum degasser cassette.",
        },
        {
            "rc_id": "RC-CHROM-03", "dev_id": "DEV-TYP-01", "cat": "Chromatography",
            "tier": "Phase I (Laboratory Investigation)", "method": "SEC-HPLC Monomer Purity",
            "what": "Injection area %RSD of 5 replicate standard injections was 2.8% (SOP acceptance limit <= 1.0%).",
            "learned": "High-pressure rotor seal groove was grooved by sample salt precipitation, causing variable microliter volume loss.",
            "capa": "Instituted 500-injection preventive rotor seal replacement schedule in calibration tracker.",
        },
        {
            "rc_id": "RC-CHROM-04", "dev_id": "DEV-TYP-03", "cat": "Chromatography",
            "tier": "Phase I (Laboratory Investigation)", "method": "CEX-HPLC Charge Variants",
            "what": "Acidic variant peak retention shifted -0.9 minutes across sequence run over 18 hours.",
            "learned": "Column heater compartment fan failed, allowing laboratory HVAC night-setback ambient cooling to drop skid temperature by 4 deg C.",
            "capa": "Configured CDS oven temperature telemetry interlock to abort injection queue if temperature exceeds +/- 0.8 deg C.",
        },
        {
            "rc_id": "RC-CHROM-01", "dev_id": "DEV-TYP-02", "cat": "Chromatography",
            "tier": "Phase I (Laboratory Investigation)", "method": "rCE-SDS Reduced Purity",
            "what": "Heavy chain peak shoulder observed, failing purity specification (93.4% vs >= 95.0%).",
            "learned": "Capillary inlet cartridge dirty with scorched SDS micelles; heating block temperature calibrated 5 deg C high.",
            "capa": "Calibrated capillary thermal block; updated weekly inlet wash sequence SOP.",
        },
        {
            "rc_id": "RC-CHROM-02", "dev_id": "DEV-TYP-01", "cat": "Chromatography",
            "tier": "Phase I (Laboratory Investigation)", "method": "SEC-HPLC Monomer Purity",
            "what": "System backpressure spiked to 320 bar during initial mobile phase purge, tripping safety cutoff.",
            "learned": "Mobile phase reservoir filter stone was blocked by algae growth due to aqueous buffer held > 72 hours without 0.02% azide.",
            "capa": "Affixed strict 48-hour room temperature expiration tags on unpreserved aqueous mobile phase bottles.",
        },

        # 7-12: Bioassay
        {
            "rc_id": "RC-BIO-01", "dev_id": "DEV-TYP-01", "cat": "Bioassay",
            "tier": "Phase I (Laboratory Investigation)", "method": "Cell-Based Bioassay (Relative Potency)",
            "what": "Relative potency 4-parameter logistic curve failed parallelism test (asymptote ratio out of spec).",
            "learned": "Cells used for seeding had reached passage 24, exceeding the approved SOP limit of 18 passages.",
            "capa": "Configured electronic cell culture inventory to hard-lock vials at passage 18; enforced QA inventory audit.",
        },
        {
            "rc_id": "RC-BIO-02", "dev_id": "DEV-TYP-01", "cat": "Bioassay",
            "tier": "Phase I (Laboratory Investigation)", "method": "Cell-Based Bioassay (Relative Potency)",
            "what": "Standard curve %CV across outer row wells exceeded 15% threshold; inner wells were fully acceptable.",
            "learned": "Microplate incubated without gas-permeable moisture seals, causing evaporative volume loss in outer wells.",
            "capa": "Standardized perimeter-row hydration moat with sterile water and specified breathable sealing films.",
        },
        {
            "rc_id": "RC-BIO-03", "dev_id": "DEV-TYP-03", "cat": "Bioassay",
            "tier": "Phase I (Laboratory Investigation)", "method": "Cell-Based Bioassay (Relative Potency)",
            "what": "Dose-response IC50 shifted 2.5-fold higher across 3 consecutive plate runs.",
            "learned": "CO2 incubator infrared sensor drifted from true 5.0% to 3.4%, altering bicarbonate buffer equilibrium.",
            "capa": "Implemented weekly secondary Fyrite gas verification before initiating clinical batch release assays.",
        },
        {
            "rc_id": "RC-BIO-04", "dev_id": "DEV-TYP-01", "cat": "Bioassay",
            "tier": "Phase I (Laboratory Investigation)", "method": "Host Cell Protein (HCP) ELISA",
            "what": "Negative control blank absorbance exceeded 0.15 OD; standard curve lower limit undetectable.",
            "learned": "New lot of blocking bovine serum albumin was not pre-screened for endogenous cross-reacting IgG.",
            "capa": "Established mandatory qualification testing protocol for raw blocking reagents prior to QC release.",
        },
        {
            "rc_id": "RC-BIO-02", "dev_id": "DEV-TYP-01", "cat": "Bioassay",
            "tier": "Phase I (Laboratory Investigation)", "method": "SPR / Octet Binding Affinity",
            "what": "Biosensor sensorgram association phase showed abnormal binding kinetics and high baseline noise.",
            "learned": "Sensor tray thermocube fan had accumulated dust, leading to 2 deg C gradient across biosensor columns.",
            "capa": "Scheduled bi-monthly HEPA vacuuming and fan speed verification on biosensor chassis.",
        },
        {
            "rc_id": "RC-BIO-01", "dev_id": "DEV-TYP-03", "cat": "Bioassay",
            "tier": "Phase I (Laboratory Investigation)", "method": "Cell-Based Bioassay (Relative Potency)",
            "what": "Cell viability dropped to 72% immediately post-trypsinization (SOP minimum >= 90%).",
            "learned": "Trypsin incubation time was inadvertently extended to 8 minutes during concurrent analyst distraction.",
            "capa": "Implemented digital bench-top countdown timers with audible alarms in bioassay tissue culture suites.",
        },

        # 13-18: Sample Handling
        {
            "rc_id": "RC-SMPL-01", "dev_id": "DEV-TYP-02", "cat": "Sample Handling",
            "tier": "Phase I (Laboratory Investigation)", "method": "Sub-visible Particulate HIAC",
            "what": "Sub-visible particulate count at >= 10 um exceeded 6000 particles/container limit.",
            "learned": "Analyst vortexed sample tubes for 30 seconds at 2500 rpm rather than gentle inversion, generating cavitation shear aggregates.",
            "capa": "Revised SOP-SMPL-014 (fictional example) to prohibit mechanical vortexing of formulated biologics; introduced rolling tube mixers.",
        },
        {
            "rc_id": "RC-SMPL-02", "dev_id": "DEV-TYP-03", "cat": "Sample Handling",
            "tier": "Phase I (Laboratory Investigation)", "method": "SEC-HPLC Monomer Purity",
            "what": "Elevated HMW dimer (2.2% vs baseline 0.9%) observed in unthawed retain sample.",
            "learned": "Sample had been left in -20 deg C frost-free freezer undergoing daily defrost thermal spikes rather than -80 deg C monitored stability chamber.",
            "capa": "Installed wireless calibrated thermocouple data loggers in all temporary sample holding freezers with auto-SMS alert.",
        },
        {
            "rc_id": "RC-SMPL-03", "dev_id": "DEV-TYP-02", "cat": "Sample Handling",
            "tier": "Phase I (Laboratory Investigation)", "method": "CEX-HPLC Charge Variants",
            "what": "Acidic variant fraction increased from 18% to 26% on stability pull sample.",
            "learned": "Sample autosampler vial sat in clear glass on brightly lit benchtop for 36 hours prior to analysis.",
            "capa": "Mandated amber autosampler vials for all light-sensitive stability pulls; added bench light shielding.",
        },
        {
            "rc_id": "RC-SMPL-01", "dev_id": "DEV-TYP-01", "cat": "Sample Handling",
            "tier": "Phase I (Laboratory Investigation)", "method": "UV-Vis A280 Protein Concentration",
            "what": "Apparent protein concentration reading showed 12% discrepancy between duplicate dilutions.",
            "learned": "Viscous 150 mg/mL mAb solution was pipetted with standard air-displacement pipette without positive-displacement tips.",
            "capa": "Procured positive-displacement electronic pipettes for all sample concentrations > 50 mg/mL.",
        },
        {
            "rc_id": "RC-SMPL-02", "dev_id": "DEV-TYP-02", "cat": "Sample Handling",
            "tier": "Phase I (Laboratory Investigation)", "method": "Sub-visible Particulate HIAC",
            "what": "Micro-air bubbles counted as particles due to inadequate degas hold time post-thaw.",
            "learned": "Sample was tested immediately after rapid water bath thaw without required 30-minute bench de-aeration hold.",
            "capa": "Added explicit de-aeration hold timer check-box in analytical test worksheet.",
        },
        {
            "rc_id": "RC-SMPL-03", "dev_id": "DEV-TYP-03", "cat": "Sample Handling",
            "tier": "Phase I (Laboratory Investigation)", "method": "SEC-HPLC Monomer Purity",
            "what": "Slight peak broadening and high molecular weight tailing in stability check.",
            "learned": "Vial septa was punctured multiple times with blunt needles, introducing microscopic silicone oil droplets into test aliquot.",
            "capa": "Specified single-puncture PTFE-faced silicone septa and dedicated micro-syringes in sampling checklist.",
        },

        # 19-24: Reference Standards
        {
            "rc_id": "RC-STD-01", "dev_id": "DEV-TYP-01", "cat": "Reference Standards",
            "tier": "Phase I (Laboratory Investigation)", "method": "Cell-Based Bioassay (Relative Potency)",
            "what": "System suitability standard curve failed ED50 acceptance window on Monday morning run.",
            "learned": "Working standard vial had been thawed 14 days earlier and stored at 4 deg C, exceeding the 7-day thawed stability window.",
            "capa": "Instituted color-coded 'Date Thawed / Discard Date' label requirements on all working standard aliquot vials.",
        },
        {
            "rc_id": "RC-STD-02", "dev_id": "DEV-TYP-01", "cat": "Reference Standards",
            "tier": "Phase I (Laboratory Investigation)", "method": "UV-Vis A280 Protein Concentration",
            "what": "Standard extinction coefficient recovery was 107.5% (SOP limit 98.0 - 102.0%).",
            "learned": "Lyophilized primary standard vial was opened immediately after removal from -20 deg C freezer without equilibration to room temperature, causing atmospheric moisture condensation onto cake.",
            "capa": "Updated standard preparation SOP to mandate 45-minute desiccator room temperature warmup before opening vial seal.",
        },
        {
            "rc_id": "RC-STD-01", "dev_id": "DEV-TYP-03", "cat": "Reference Standards",
            "tier": "Phase I (Laboratory Investigation)", "method": "CEX-HPLC Charge Variants",
            "what": "Main peak retention time on reference standard drifted by +0.6 minutes compared to historical control chart.",
            "learned": "Standard vial batch had reached its 24-month primary re-test date without extension protocol approval.",
            "capa": "Integrated standard expiration auto-warning into QC schedule dashboard with 60-day advance re-qualification alerts.",
        },
        {
            "rc_id": "RC-STD-02", "dev_id": "DEV-TYP-01", "cat": "Reference Standards",
            "tier": "Phase I (Laboratory Investigation)", "method": "Host Cell Protein (HCP) ELISA",
            "what": "HCP standard calibrator curve r-squared value was 0.978 (acceptance limit >= 0.990).",
            "learned": "Reconstitution diluent was added aggressively down center of vial rather than gently along glass wall, foaming standard protein.",
            "capa": "Mandated slow wall-directed pipette dispensing and gentle orbital swirling during reconstitutions.",
        },
        {
            "rc_id": "RC-STD-01", "dev_id": "DEV-TYP-01", "cat": "Reference Standards",
            "tier": "Phase I (Laboratory Investigation)", "method": "rCE-SDS Reduced Purity",
            "what": "Molecular weight marker migration time shifted outside the expected window.",
            "learned": "Internal standard reagent lot had undergone inadvertent freeze-thaw due to dry ice replenishment delay during transport.",
            "capa": "Enforced continuous temperature tracking tag requirement for all incoming reference standard transfers.",
        },
        {
            "rc_id": "RC-STD-02", "dev_id": "DEV-TYP-03", "cat": "Reference Standards",
            "tier": "Phase I (Laboratory Investigation)", "method": "SPR / Octet Binding Affinity",
            "what": "Sensor chip ligand capture level decreased by 30% from target density.",
            "learned": "Amine coupling standard NHS/EDC reagents had absorbed moisture during repeated bottle openings.",
            "capa": "Aliquot single-use lyophilized EDC/NHS vials in nitrogen-purged desiccated pouches.",
        },

        # 25-30: Reagents & Media
        {
            "rc_id": "RC-RGT-01", "dev_id": "DEV-TYP-01", "cat": "Reagents",
            "tier": "Phase I (Laboratory Investigation)", "method": "CEX-HPLC Charge Variants",
            "what": "Resolution between acidic variant and main peak failed SST (Rs = 1.15 vs target >= 1.40).",
            "learned": "Buffer B pH was 7.82 instead of nominal 7.50 due to pH meter calibration using expired buffer standards.",
            "capa": "Configured pH meter calibration lockout requiring fresh unopened calibrant buffers every 30 days.",
        },
        {
            "rc_id": "RC-RGT-02", "dev_id": "DEV-TYP-01", "cat": "Reagents",
            "tier": "Phase I (Laboratory Investigation)", "method": "rCE-SDS Reduced Purity",
            "what": "Ghost peaks at 15 kDa and 45 kDa appeared in blank and sample electropherograms.",
            "learned": "HPLC grade water purification polishing cartridge was exhausted, releasing organic trace contaminants into sample buffer.",
            "capa": "Added daily resistivity (>= 18.2 MOhm*cm) and TOC (< 10 ppb) sign-off check in test preparation sheet.",
        },
        {
            "rc_id": "RC-RGT-01", "dev_id": "DEV-TYP-03", "cat": "Reagents",
            "tier": "Phase I (Laboratory Investigation)", "method": "Cell-Based Bioassay (Relative Potency)",
            "what": "Cell growth rate in assay plate declined by 40%, flattening upper plateau.",
            "learned": "Assay medium was prepared with L-glutamine stock solution that had degraded into ammonia at 4 deg C.",
            "capa": "Switched to stable dipeptide L-alanyl-L-glutamine (GlutaMAX equivalent) across all cell culture formulations.",
        },
        {
            "rc_id": "RC-RGT-02", "dev_id": "DEV-TYP-04", "cat": "Reagents",
            "tier": "Phase I (Laboratory Investigation)", "method": "SEC-HPLC Monomer Purity",
            "what": "HPLC pump seal wash sensor alarmed on high conductance; pressure ripple reached 4.5%.",
            "learned": "10% isopropanol seal wash solution had completely evaporated over a long holiday weekend.",
            "capa": "Instituted pre-run automated check of wash bottle liquid level in instrument readiness checklist.",
        },
        {
            "rc_id": "RC-RGT-01", "dev_id": "DEV-TYP-01", "cat": "Reagents",
            "tier": "Phase I (Laboratory Investigation)", "method": "Osmolality & pH Release Testing",
            "what": "Osmolality standard check reading was 325 mOsm/kg against nominal 290 mOsm/kg.",
            "learned": "Osmometer cooling chamber was contaminated with crystallized salt residues from previous spilled tube.",
            "capa": "Mandated optical chamber swab cleaning with deionized water before daily 2-point calibration.",
        },
        {
            "rc_id": "RC-RGT-02", "dev_id": "DEV-TYP-02", "cat": "Reagents",
            "tier": "Phase I (Laboratory Investigation)", "method": "UV-Vis A280 Protein Concentration",
            "what": "High baseline absorbance (> 0.08 OD) across entire 260-320 nm spectrum in buffer blank.",
            "learned": "Surfactant stock solution (Polysorbate 20) had oxidized, producing UV-absorbing peroxide degradation products.",
            "capa": "Procured high-purity nitrogen-overlay single-use ampules for formulation surfactant preparation.",
        },

        # 31-36: Software / CDS Integration & Documentation
        {
            "rc_id": "RC-SOFT-01", "dev_id": "DEV-TYP-01", "cat": "Software/Integration",
            "tier": "Phase I (Laboratory Investigation)", "method": "SEC-HPLC Monomer Purity",
            "what": "Reported monomer purity was 97.1%, triggering an investigation for sudden 1.2% drop from previous batch.",
            "learned": "CDS automatic integration baseline skimmed under the trailing dimer peak rather than valley-to-valley drop.",
            "capa": "Locked processing method integration parameters in CDS; prohibited manual baseline adjustments without QA concurrence.",
        },
        {
            "rc_id": "RC-SOFT-02", "dev_id": "DEV-TYP-02", "cat": "Software/Integration",
            "tier": "Phase I (Laboratory Investigation)", "method": "UV-Vis A280 Protein Concentration",
            "what": "Calculated drug substance batch concentration was 10.2 mg/mL, failing release specification of 95 - 105 mg/mL.",
            "learned": "Sample accessioner entered dilution factor as 1.0 instead of 10.0 into LIMS sample header.",
            "capa": "Implemented double-independent peer review in LIMS accessioning for concentrated drug substance lots.",
        },
        {
            "rc_id": "RC-DOC-01", "dev_id": "DEV-TYP-01", "cat": "Documentation",
            "tier": "Phase I (Laboratory Investigation)", "method": "CEX-HPLC Charge Variants",
            "what": "Column pressure spiked during first gradient ramp; retention times shifted +2.0 minutes across sequence.",
            "learned": "Analyst skipped the required 30-minute column pre-equilibration at initial mobile phase ratio before starting sequence.",
            "capa": "Programmed automated instrument equilibration macro into CDS run startup sequence template.",
        },
        {
            "rc_id": "RC-DOC-02", "dev_id": "DEV-TYP-01", "cat": "Documentation",
            "tier": "Phase I (Laboratory Investigation)", "method": "Host Cell Protein (HCP) ELISA",
            "what": "HCP calculated concentration in final report differed by 2x from raw microplate reader printout.",
            "learned": "Analyst used obsolete revision of calculation spreadsheet with hard-coded duplicate divisor.",
            "capa": "Migrated all standalone offline spreadsheets into central approved LIMS calculation templates.",
        },
        {
            "rc_id": "RC-SOFT-01", "dev_id": "DEV-TYP-03", "cat": "Software/Integration",
            "tier": "Phase I (Laboratory Investigation)", "method": "rCE-SDS Reduced Purity",
            "what": "CE separation time shifted by 15% between two consecutive carousel runs.",
            "learned": "CE instrument software clock had not synchronized with NTP time server, confusing carousel sequence tags.",
            "capa": "Configured domain group policy to enforce automated NTP synchronization across all lab PC instruments.",
        },
        {
            "rc_id": "RC-DOC-01", "dev_id": "DEV-TYP-01", "cat": "Documentation",
            "tier": "Phase I (Laboratory Investigation)", "method": "Sub-visible Particulate HIAC",
            "what": "Tare weight recorded for sample container was missing analyst initials and timestamp.",
            "learned": "Analyst recorded tare on informal sticky note before transcribing into batch record.",
            "capa": "Reinforced good documentation practices (GDP) training; installed connected electronic balances.",
        },

        # 37-40: Phase II (Cross-Functional / Manufacturing)
        {
            "rc_id": "RC-MFG-01", "dev_id": "DEV-TYP-05", "cat": "Manufacturing/Process",
            "tier": "Phase II (Cross-Functional / Manufacturing)", "method": "Host Cell Protein (HCP) ELISA",
            "what": "Final drug substance pool HCP level was 185 ppm, exceeding release limit of <= 100 ppm.",
            "learned": "Bioreactor harvest depth filter pressure differential exceeded limit, causing breakthrough of colloidal host protein.",
            "capa": "Revised manufacturing batch record with strict delta-P warning limits on harvest clarification filters.",
        },
        {
            "rc_id": "RC-MFG-02", "dev_id": "DEV-TYP-05", "cat": "Manufacturing/Process",
            "tier": "Phase II (Cross-Functional / Manufacturing)", "method": "SEC-HPLC Monomer Purity",
            "what": "Bulk drug substance monomer purity dropped to 95.8% (specification >= 96.5%) with elevated soluble dimer.",
            "learned": "Ultrafiltration/diafiltration skid recirculation peristaltic pump operated at high shear rate during final retentate concentration.",
            "capa": "Replaced peristaltic pump with low-shear 4-piston diaphragm pump in downstream purification skid.",
        },
        {
            "rc_id": "RC-MFG-01", "dev_id": "DEV-TYP-05", "cat": "Manufacturing/Process",
            "tier": "Phase II (Cross-Functional / Manufacturing)", "method": "Sub-visible Particulate HIAC",
            "what": "Drug product sterile filtration process yielded elevated 10-um particulate counts in downstream surge tank.",
            "learned": "Silicone tubing connection on fill line had worn from repeated autoclave sterilization cycles.",
            "capa": "Instituted 5-autoclave-cycle retirement limit on all product-contact elastomeric hoses.",
        },
        {
            "rc_id": "RC-MFG-02", "dev_id": "DEV-TYP-05", "cat": "Manufacturing/Process",
            "tier": "Phase II (Cross-Functional / Manufacturing)", "method": "CEX-HPLC Charge Variants",
            "what": "Acidic charge variant reached 32% (alert limit 28%) in final harvest broth.",
            "learned": "Bioreactor temperature control loop oscillated +1.5 deg C above setpoint during days 10-14 of fed-batch culture.",
            "capa": "Re-tuned PID heating/cooling loop on 2000L production bioreactor; added real-time temperature deviation alerting.",
        },
    ]

    for idx, c in enumerate(base_cases):
        lesson_id = f"LL-2026-{idx+1:03d}"
        day_offset = (idx * 6) % 250 + 10
        date_str = f"2026-{(idx % 11) + 1:02d}-{(idx % 26) + 1:02d}"
        proj = projects[idx % len(projects)]
        owner = "Lead QC Analyst" if "Phase I" in c["tier"] else "Process Engineering Specialist"

        lessons.append({
            "id": lesson_id,
            "date": date_str,
            "project_id": proj,
            "method_name": c["method"],
            "root_cause_id": c["rc_id"],
            "deviation_type_id": c["dev_id"],
            "what_happened": c["what"],
            "what_learned": c["learned"],
            "preventive_action": c["capa"],
            "owner": owner,
            "investigation_tier": c["tier"],
            "status": "Closed",
            "created_at": f"{date_str}T08:30:00Z",
        })

    return lessons


def seed_knowledge_base(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Seeds root causes, deviation types, and 42 synthetic lessons learned into SQLite.
    Returns counts of inserted items.
    """
    with conn:
        # 1. Root causes
        for rc in SEED_ROOT_CAUSES:
            conn.execute(
                """
                INSERT OR IGNORE INTO root_causes (id, code, name, category, investigation_tier, description)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (rc["id"], rc["code"], rc["name"], rc["category"], rc["investigation_tier"], rc["description"]),
            )

        # 2. Deviation types
        for dt in SEED_DEVIATION_TYPES:
            conn.execute(
                """
                INSERT OR IGNORE INTO deviation_types (id, code, name, area, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                (dt["id"], dt["code"], dt["name"], dt["area"], dt["description"]),
            )

        # 3. Lessons learned
        lessons = _build_synthetic_lessons()
        for ll in lessons:
            conn.execute(
                """
                INSERT OR IGNORE INTO lessons_learned
                (id, date, project_id, method_name, root_cause_id, deviation_type_id, what_happened,
                 what_learned, preventive_action, owner, investigation_tier, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ll["id"], ll["date"], ll["project_id"], ll["method_name"],
                    ll["root_cause_id"], ll["deviation_type_id"], ll["what_happened"],
                    ll["what_learned"], ll["preventive_action"], ll["owner"],
                    ll["investigation_tier"], ll["status"], ll["created_at"],
                ),
            )

        # 4. Links between lessons (synthetic relationships)
        links = [
            ("LNK-01", "LL-2026-001", "LL-2026-006", "Related to mobile phase particulate fouling"),
            ("LNK-02", "LL-2026-007", "LL-2026-009", "Related to bioassay incubation environmental drift"),
            ("LNK-03", "LL-2026-013", "LL-2026-017", "Related to particle count shear aggregation in sample prep"),
            ("LNK-04", "LL-2026-019", "LL-2026-021", "Related to reference standard stability and expiry"),
            ("LNK-05", "LL-2026-037", "LL-2026-039", "Related to downstream harvest filtration and chromatography yield"),
        ]
        for lnk in links:
            conn.execute(
                """
                INSERT OR IGNORE INTO lesson_links (id, lesson_id, linked_lesson_id, relationship)
                VALUES (?, ?, ?, ?)
                """,
                lnk,
            )

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM root_causes")
    n_rc = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM deviation_types")
    n_dt = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM lessons_learned")
    n_ll = cur.fetchone()[0]

    return {
        "root_causes_count": n_rc,
        "deviation_types_count": n_dt,
        "lessons_learned_count": n_ll,
    }


def get_all_root_causes(conn: sqlite3.Connection, tier: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves root causes optionally filtered by investigation tier."""
    cur = conn.cursor()
    if tier:
        cur.execute("SELECT * FROM root_causes WHERE investigation_tier = ? ORDER BY category, code", (tier,))
    else:
        cur.execute("SELECT * FROM root_causes ORDER BY investigation_tier, category, code")
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_all_deviation_types(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Retrieves all deviation types."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM deviation_types ORDER BY area, code")
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_lessons_learned(
    conn: sqlite3.Connection,
    project_id: Optional[str] = None,
    method_name: Optional[str] = None,
    tier: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves lessons learned with joined root cause and deviation type names."""
    query = """
        SELECT
            ll.id, ll.date, ll.project_id, ll.method_name,
            ll.root_cause_id, rc.name AS root_cause_name, rc.category AS root_cause_category,
            ll.deviation_type_id, dt.name AS deviation_type_name,
            ll.what_happened, ll.what_learned, ll.preventive_action,
            ll.owner, ll.investigation_tier, ll.status, ll.created_at
        FROM lessons_learned ll
        LEFT JOIN root_causes rc ON ll.root_cause_id = rc.id
        LEFT JOIN deviation_types dt ON ll.deviation_type_id = dt.id
        WHERE 1=1
    """
    params: List[Any] = []
    if project_id:
        query += " AND ll.project_id = ?"
        params.append(project_id)
    if method_name:
        query += " AND ll.method_name = ?"
        params.append(method_name)
    if tier:
        query += " AND ll.investigation_tier = ?"
        params.append(tier)

    query += " ORDER BY ll.date DESC"
    cur = conn.cursor()
    cur.execute(query, params)
    return [dict(r) for r in cur.fetchall()]


def _tokenize(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def search_lessons_learned(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    TF-IDF based keyword search across historical lessons learned records.
    Returns cited past deviation cases for decision support only.
    """
    lessons = get_lessons_learned(conn)
    query_tokens = _tokenize(query)

    if not query_tokens or not lessons:
        return {
            "query": query,
            "results_count": 0,
            "matches": [],
            "advisory": ADVISORY_DISCLAIMER,
            "disclaimer": DRAFT_STAMP,
        }

    # Build corpus documents
    docs = []
    for l in lessons:
        doc_text = " ".join([
            l.get("what_happened", ""),
            l.get("what_learned", ""),
            l.get("preventive_action", ""),
            l.get("method_name", ""),
            l.get("root_cause_name", ""),
            l.get("root_cause_category", ""),
        ])
        docs.append(_tokenize(doc_text))

    n_docs = len(docs)

    # Document frequency
    df: Dict[str, int] = {}
    for tokens in docs:
        seen = set(tokens)
        for t in seen:
            df[t] = df.get(t, 0) + 1

    scores = []
    for idx, (l, tokens) in enumerate(zip(lessons, docs)):
        score = 0.0
        doc_len = max(len(tokens), 1)
        for q in query_tokens:
            tf = tokens.count(q) / doc_len
            idf = math.log((n_docs + 1.0) / (df.get(q, 0) + 1.0)) + 1.0
            score += tf * idf

        if score > 0:
            scores.append((score, l))

    scores.sort(key=lambda x: x[0], reverse=True)
    top_matches = []

    for score, l in scores[:top_k]:
        top_matches.append({
            "lesson_id": l["id"],
            "date": l["date"],
            "project_id": l["project_id"],
            "method_name": l["method_name"],
            "investigation_tier": l["investigation_tier"],
            "root_cause_category": l["root_cause_category"],
            "root_cause_name": l["root_cause_name"],
            "similarity_score": round(score, 4),
            "what_happened": l["what_happened"],
            "what_learned": l["what_learned"],
            "preventive_action": l["preventive_action"],
            "owner": l["owner"],
            "citation_note": f"Cited precedent {l['id']} ({l['investigation_tier']})",
        })

    out = {
        "query": query,
        "results_count": len(top_matches),
        "matches": top_matches,
        "advisory": ADVISORY_DISCLAIMER,
        "disclaimer": DRAFT_STAMP,
    }
    assert_no_banned_phrases(str(out))
    return out


def add_lesson_learned(
    conn: sqlite3.Connection,
    lesson: Dict[str, Any],
    audit_trail: AuditTrail,
    actor_name: str,
    actor_role: str,
    justification: str,
) -> str:
    """
    Appends a new lesson learned and logs an ALCOA+ audit event.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    lesson_id = lesson.get("id") or f"LL-USER-{int(datetime.now(timezone.utc).timestamp())}"

    with conn:
        conn.execute(
            """
            INSERT INTO lessons_learned
            (id, date, project_id, method_name, root_cause_id, deviation_type_id, what_happened,
             what_learned, preventive_action, owner, investigation_tier, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lesson_id,
                lesson.get("date", now_utc[:10]),
                lesson.get("project_id", "PRJ-GEN-001"),
                lesson.get("method_name", "Analytical Method"),
                lesson.get("root_cause_id", "RC-CHROM-01"),
                lesson.get("deviation_type_id", "DEV-TYP-01"),
                lesson.get("what_happened", ""),
                lesson.get("what_learned", ""),
                lesson.get("preventive_action", ""),
                actor_name,
                lesson.get("investigation_tier", "Phase I (Laboratory Investigation)"),
                lesson.get("status", "Draft"),
                now_utc,
            ),
        )

    # Log audit event
    audit_trail.append_event(
        actor_name=actor_name,
        actor_role=actor_role,
        action="CREATE_LESSON_LEARNED",
        justification=justification,
        entity_type="LESSONS_LEARNED",
        entity_id=lesson_id,
        payload=lesson,
    )

    return lesson_id
