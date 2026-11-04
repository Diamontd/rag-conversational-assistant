"""
Qdrant Vector Database Indexer, Metadata-Filtered Retrieval and Reranking Engine.
Derived from InsuranceQA domain expert materials across 6 core insurance domains.
Features BM25 + Structural Field-Weighted Hybrid Reranker.
"""

import os
import re
import json
import time
import math
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.config import config, observe
from src.schemas import DocumentChunk, ScoredDocument

logger = logging.getLogger("rag_conversational_assistant.knowledge_base")

# 35 Realistic Domain Expert Documents derived from InsuranceQA
# Includes primary topic articles and cross-category distractor articles to test retrieval precision
CORPUS_DOCUMENTS: List[DocumentChunk] = [
    # HEALTH INSURANCE (DOC-1001 to DOC-1007)
    DocumentChunk(
        doc_id="DOC-1001",
        title="Dropping a Spouse from Employer Health Insurance Outside Open Enrollment",
        category="health",
        sub_topic="spouse_removal",
        content=(
            "Can you drop your spouse from your health insurance policy? The answer strictly depends on employer plan rules "
            "and state regulations. Under Section 125 cafeteria plans, you generally CANNOT drop a spouse outside the annual open "
            "enrollment window unless you experience a Qualifying Life Event (QLE) such as legal divorce, formal legal separation, "
            "or the spouse gaining other credible minimum essential coverage. Furthermore, many states and employer carriers "
            "mandate that the spouse must sign written consent agreeing to the cancellation of coverage. This rule prevents one spouse "
            "from unilaterally leaving the other uninsured during domestic disputes or divorce proceedings."
        ),
        metadata={"requires_written_consent": True, "qualifying_event_required": True, "open_enrollment_restricted": True}
    ),
    DocumentChunk(
        doc_id="DOC-1002",
        title="Spousal Removal and Special Enrollment Period Rules",
        category="health",
        sub_topic="special_enrollment",
        content=(
            "When removing a spouse from an ACA-compliant health insurance plan following a Qualifying Life Event (QLE), you have "
            "a strict 30-day (or 60-day for ACA exchanges) special enrollment window to submit proof. Proof typically includes a final divorce "
            "decree, certificate of marriage dissolution, or proof of alternative employer-sponsored health coverage. If the required written "
            "documentation is not submitted within the statutory window, the spouse must remain on the policy until the subsequent open enrollment period."
        ),
        metadata={"statutory_days": 30, "requires_proof": True}
    ),
    DocumentChunk(
        doc_id="DOC-1003",
        title="Health Savings Account (HSA) Tax Treatment and Qualified Medical Expenses",
        category="health",
        sub_topic="hsa_tax",
        content=(
            "Contributions to a Health Savings Account (HSA) are 100% tax-deductible or made pre-tax through employer payroll deductions. "
            "Earnings and capital gains within the HSA grow 100% tax-free. Withdrawals are completely tax-free provided they are utilized "
            "strictly for qualified medical expenses as defined under IRC Section 213(d). Non-qualified distributions are subject to ordinary "
            "income tax plus an additional 20% IRS tax penalty if taken prior to age 65."
        ),
        metadata={"tax_qualified": True, "irs_code": "213(d)", "penalty_rate": 0.20}
    ),
    DocumentChunk(
        doc_id="DOC-1004",
        title="COBRA Continuation Coverage Following Spousal Dropping",
        category="health",
        sub_topic="cobra_continuation",
        content=(
            "When a spouse loses employer health coverage due to divorce or legal separation, they qualify for federal COBRA continuation "
            "coverage for up to 36 months. The employee or spouse must notify the employer plan administrator within 60 days of the qualifying event. "
            "Under COBRA, the dropped spouse pays up to 102% of the full premium cost (including the portion previously subsidized by the employer)."
        ),
        metadata={"cobra_months": 36, "notice_window_days": 60}
    ),
    DocumentChunk(
        doc_id="DOC-1005",
        title="Tax Deductibility of Health Insurance Premiums and Flexible Spending Accounts",
        category="health",
        sub_topic="health_tax",
        content=(
            "Are insurance benefits and premiums taxed? Under IRS guidelines, health insurance employer subsidies are exempt from gross income. "
            "Health benefits paid directly to healthcare providers for medical expenses are 100% tax-free, but wage replacement disability income is taxable."
        ),
        metadata={"tax_free": True}
    ),
    DocumentChunk(
        doc_id="DOC-1006",
        title="Understanding Deductible, Copay, and Coinsurance Numbers on Your Health Card",
        category="health",
        sub_topic="card_numbers",
        content=(
            "What do the numbers on my insurance card mean? The numbers on your health insurance card represent your copayment amounts for primary "
            "care doctor visits, specialist visits, emergency room visits, and your annual out-of-pocket deductible."
        ),
        metadata={"card_details": True}
    ),
    DocumentChunk(
        doc_id="DOC-1007",
        title="How Sick Do You Have to Be for Short-Term Disability Benefits to Trigger?",
        category="health",
        sub_topic="disability_triggers",
        content=(
            "How sick do I have to be before the insurance pays out? For short-term disability insurance, benefits require medical certification of "
            "total inability to perform your regular occupation, which differs fundamentally from long-term care ADL triggers."
        ),
        metadata={"disability_rules": True}
    ),

    # LONG-TERM CARE INSURANCE (DOC-2001 to DOC-2004)
    DocumentChunk(
        doc_id="DOC-2001",
        title="Tax Treatment of Long-Term Care Insurance Benefits and Lump Sums",
        category="long_term_care",
        sub_topic="tax_treatment",
        content=(
            "Generally speaking, benefits paid under a tax-qualified Long-Term Care (LTC) insurance contract are received 100% tax-free "
            "under Internal Revenue Code Section 7702B. Benefits received under reimbursement plans that pay for actual care expenses are non-taxable "
            "without limit. For per-diem or indemnity plans that pay a set daily dollar amount regardless of actual costs, benefits are tax-free "
            "up to the IRS daily per-diem cap (indexed annually for inflation). Any excess over the greater of actual expenses or the cap is subject "
            "to ordinary income tax. Benefit payments are never subject to capital gains tax."
        ),
        metadata={"tax_free": True, "irs_code": "7702B", "per_diem_cap_applicable": True}
    ),
    DocumentChunk(
        doc_id="DOC-2002",
        title="Activities of Daily Living (ADLs) and LTC Benefit Triggers",
        category="long_term_care",
        sub_topic="benefit_triggers",
        content=(
            "To qualify for benefits under a tax-qualified long-term care policy, a licensed healthcare practitioner must certify that the insured "
            "is chronically ill. This requires the inability to perform at least 2 out of 6 Activities of Daily Living (ADLs) for a period expected "
            "to last at least 90 days without substantial assistance. The 6 recognized ADLs are bathing, dressing, eating, transferring, toileting, "
            "and continence. Severe cognitive impairment (e.g., Alzheimer's disease) also serves as an independent qualification trigger."
        ),
        metadata={"adl_required_count": 2, "adl_total_count": 6, "min_duration_days": 90}
    ),
    DocumentChunk(
        doc_id="DOC-2003",
        title="Elimination Period and Out-of-Pocket Waiting Periods in LTC",
        category="long_term_care",
        sub_topic="elimination_period",
        content=(
            "The elimination period in a long-term care policy functions as a time-based deductible. Common elimination periods range from 30, 60, "
            "to 90 calendar or service days. During this window, the insured must pay for covered care services 100% out-of-pocket before insurance "
            "reimbursements commence. Some policies require consecutive days of paid service, while others calculate calendar days from the date of certification."
        ),
        metadata={"typical_waiting_days": [30, 60, 90]}
    ),
    DocumentChunk(
        doc_id="DOC-2004",
        title="State Long-Term Care Partnership Programs and Medicaid Asset Protection",
        category="long_term_care",
        sub_topic="partnership_medicaid",
        content=(
            "State Long-Term Care Partnership policies provide dollar-for-dollar Medicaid asset protection. For every dollar of benefits paid out "
            "by a certified Partnership policy, the policyholder can protect a dollar of personal assets from Medicaid spend-down requirements "
            "and Medicaid estate recovery rules, allowing individuals to qualify for long-term Medicaid support without liquidating all family wealth."
        ),
        metadata={"partnership_eligible": True, "medicaid_spenddown_protection": True}
    ),

    # AUTO INSURANCE (DOC-3001 to DOC-3008)
    DocumentChunk(
        doc_id="DOC-3001",
        title="Understanding Split Liability Limits in Auto Insurance Policies",
        category="auto",
        sub_topic="split_limits",
        content=(
            "Split limits refer to the bodily injury and property damage liability portion of an auto insurance policy broken down into three distinct "
            "numbers, typically expressed in thousands of dollars (e.g., 100/300/100 or 50/100/25). The first number represents the maximum payout for "
            "bodily injury to any single individual injured in an accident you cause. The second number represents the total cap for bodily injury for all "
            "injured persons combined in a single accident. The third number is the maximum liability payout for property damage caused to other vehicles "
            "or structures."
        ),
        metadata={"split_limit_structure": "per_person/per_accident/property_damage"}
    ),
    DocumentChunk(
        doc_id="DOC-3002",
        title="Removing a Spouse or Driver from Auto Insurance Policies",
        category="auto",
        sub_topic="spouse_driver_removal",
        content=(
            "To remove a spouse or resident driver from an auto insurance policy, insurers require verification that the individual is no longer "
            "operating the insured vehicles. Accepted proof includes proof of separate auto insurance coverage with another carrier, a valid change-of-address "
            "document showing distinct residences following separation, or signing a formal Named Driver Exclusion endorsement. A Named Driver Exclusion "
            "stipulates that if the excluded spouse operates the vehicle and causes an accident, no liability or physical damage coverage applies whatsoever."
        ),
        metadata={"requires_proof_or_exclusion": True, "named_driver_exclusion": True}
    ),
    DocumentChunk(
        doc_id="DOC-3003",
        title="Taxability of Auto Insurance Settlements and Property Damage Reimbursements",
        category="auto",
        sub_topic="tax_settlements",
        content=(
            "Under IRS guidelines, compensatory settlements received from auto insurance for physical injuries, physical sickness, and property damage repairs "
            "are non-taxable. However, settlement amounts explicitly designated for lost wages or compensation for emotional distress not arising directly "
            "from physical injuries are considered taxable gross income. Any punitive damages awarded in a lawsuit are always 100% taxable at ordinary income rates."
        ),
        metadata={"injury_settlement_tax_free": True, "punitive_taxable": True, "lost_wages_taxable": True}
    ),
    DocumentChunk(
        doc_id="DOC-3004",
        title="Collision vs. Comprehensive Auto Coverage and Deductible Application",
        category="auto",
        sub_topic="collision_comprehensive",
        content=(
            "Collision insurance covers damage to your vehicle resulting from an impact with another vehicle or stationary object, regardless of fault. "
            "Comprehensive insurance covers non-collision losses such as theft, vandalism, hail, fire, animal collisions, and cracked windshields. "
            "Both coverages carry independent per-incident deductibles that must be paid out-of-pocket prior to insurer reimbursement."
        ),
        metadata={"deductible_applies": True}
    ),
    DocumentChunk(
        doc_id="DOC-3005",
        title="Uninsured and Underinsured Motorist (UM/UIM) Coverage Essentials",
        category="auto",
        sub_topic="uninsured_motorist",
        content=(
            "Uninsured Motorist (UM) and Underinsured Motorist (UIM) coverages protect you, your family members, and passengers if you are hit by an "
            "at-fault driver who carries no auto liability insurance or whose policy limits are insufficient to cover your total medical expenses. "
            "UM/UIM essentially stands in place of the negligent driver's missing liability policy up to your selected coverage limit."
        ),
        metadata={"essential_coverage": True}
    ),
    DocumentChunk(
        doc_id="DOC-3006",
        title="Dropping a Spouse or Permissive Driver from Auto Policy Declaration",
        category="auto",
        sub_topic="auto_spouse",
        content=(
            "Can I drop my spouse from my insurance policy right now? If you want to drop your spouse from a personal auto insurance policy, "
            "you must provide proof of separate auto insurance or execute a Named Driver Exclusion with the underwriting carrier."
        ),
        metadata={"spouse_rules": True}
    ),
    DocumentChunk(
        doc_id="DOC-3007",
        title="Taxation of Car Crash Insurance Settlements vs Lost Income",
        category="auto",
        sub_topic="settlement_taxes",
        content=(
            "Will the government take taxes from my insurance payout? Settlements for physical injuries resulting from car accidents are non-taxable "
            "under IRS rules, whereas lost wage replacements are subject to income taxes."
        ),
        metadata={"tax_rules": True}
    ),
    DocumentChunk(
        doc_id="DOC-3008",
        title="Where and How to Buy Auto Insurance: Independent Brokers vs Direct Online",
        category="auto",
        sub_topic="where_to_buy_auto",
        content=(
            "Where should I go to buy a policy? When purchasing personal auto insurance, you can compare quotes through independent insurance agencies "
            "or apply directly via online carrier portals."
        ),
        metadata={"distribution": True}
    ),

    # LIFE INSURANCE (DOC-4001 to DOC-4006)
    DocumentChunk(
        doc_id="DOC-4001",
        title="How and Where to Buy Whole Life Insurance: Agents vs. Direct Brokers",
        category="life",
        sub_topic="purchasing_whole_life",
        content=(
            "You can purchase Whole Life insurance through licensed independent life insurance agents, captive agents representing a single carrier, "
            "or direct digital brokers. Working with an independent broker is strongly recommended because they compare underwriting guidelines, dividend "
            "performance histories, and guaranteed cash value accumulation schedules across dozens of mutual insurers. Whole life guarantees a lifetime "
            "death benefit with fixed premiums that never increase with age."
        ),
        metadata={"distribution": "independent_agents_or_captive", "guaranteed_premiums": True}
    ),
    DocumentChunk(
        doc_id="DOC-4002",
        title="Federal Income Tax Exemption on Life Insurance Death Benefits",
        category="life",
        sub_topic="tax_death_benefits",
        content=(
            "Under Internal Revenue Code Section 101(a), life insurance death benefits paid to a designated beneficiary as a lump sum are 100% exempt "
            "from federal and state income tax. However, if the beneficiary elects an installment payout option where the insurer holds the proceeds and "
            "pays them out over time, the interest portion of each installment payment is taxable as ordinary income. In large estates exceeding federal "
            "exemption thresholds, death benefits may be included in the taxable estate unless held inside an Irrevocable Life Insurance Trust (ILIT)."
        ),
        metadata={"irc_section": "101(a)", "lump_sum_tax_free": True, "installment_interest_taxable": True}
    ),
    DocumentChunk(
        doc_id="DOC-4003",
        title="Changing Beneficiaries and Spousal Rights in Community Property States",
        category="life",
        sub_topic="beneficiary_designation",
        content=(
            "On most life insurance policies, the policyholder has the unilateral right to change beneficiaries at any time if the designation is revocable. "
            "However, in community property states (e.g., California, Texas, Washington), if community funds were used to pay policy premiums, a spouse may "
            "have a statutory community property claim to 50% of the death benefit. In such jurisdictions, naming someone other than the spouse requires "
            "a formal signed spousal waiver."
        ),
        metadata={"community_property_waiver": True, "revocable_designation": True}
    ),
    DocumentChunk(
        doc_id="DOC-4004",
        title="Term Life vs. Permanent Life Insurance Comparison",
        category="life",
        sub_topic="term_vs_permanent",
        content=(
            "Term life insurance provides temporary death benefit protection for a fixed period (typically 10, 20, or 30 years) with no cash value build-up, "
            "making it the most cost-effective option for income replacement during working years. Permanent life insurance (Whole Life, Universal Life) "
            "combines lifelong protection with an internal cash value account that grows tax-deferred and can be accessed via policy loans."
        ),
        metadata={"term_duration_years": [10, 20, 30], "has_cash_value": False}
    ),
    DocumentChunk(
        doc_id="DOC-4005",
        title="Where to Buy Term Life vs Universal Life Insurance",
        category="life",
        sub_topic="where_to_buy_term",
        content=(
            "Where should I go to buy a policy? For term life and universal life policies, consumers can purchase through independent life insurance brokerage "
            "firms or direct online term aggregators."
        ),
        metadata={"distribution": True}
    ),
    DocumentChunk(
        doc_id="DOC-4006",
        title="Removing an Ex-Spouse as Beneficiary on Life Insurance After Separation",
        category="life",
        sub_topic="spousal_beneficiary_removal",
        content=(
            "How do I take someone off my insurance after separation? In life insurance, removing a former spouse requires submitting a Change of Beneficiary "
            "form, though community property rules in some states require a signed waiver."
        ),
        metadata={"spousal_waiver": True}
    ),

    # RENTERS INSURANCE (DOC-5001 to DOC-5005)
    DocumentChunk(
        doc_id="DOC-5001",
        title="Determining How Much Renters Insurance Should Be Purchased",
        category="renters",
        sub_topic="coverage_calculation",
        content=(
            "How much renters insurance you should purchase depends on the total replacement cost of your personal possessions and your personal liability "
            "risk profile. Industry standards recommend conducting a room-by-room home inventory documenting clothing, furniture, electronics, and kitchenware. "
            "Most insurance carriers require a minimum personal property baseline of $15,000 to $20,000. For personal liability protection against third-party "
            "bodily injury lawsuits or accidental damage to the landlord's property, financial experts recommend at least $100,000 to $300,000 in liability limits."
        ),
        metadata={"personal_property_min": 15000, "recommended_liability": 300000}
    ),
    DocumentChunk(
        doc_id="DOC-5002",
        title="Actual Cash Value (ACV) vs. Replacement Cost Value (RCV) for Renters",
        category="renters",
        sub_topic="acv_vs_rcv",
        content=(
            "When selecting renters insurance personal property coverage, policyholders must choose between Actual Cash Value (ACV) and Replacement Cost Value (RCV). "
            "ACV reimburses the item's current market value after deducting depreciation, which often results in significantly lower payouts for older electronics "
            "and furniture. RCV pays the full modern cost to replace the destroyed item with a brand-new item of comparable kind and quality without depreciation."
        ),
        metadata={"coverage_types": ["ACV", "RCV"], "rcv_recommended": True}
    ),
    DocumentChunk(
        doc_id="DOC-5003",
        title="Loss of Use and Additional Living Expenses (ALE) for Tenants",
        category="renters",
        sub_topic="loss_of_use",
        content=(
            "If your rented home becomes uninhabitable due to a covered peril (such as fire, smoke, or severe burst pipe damage), Loss of Use coverage "
            "(Coverage D) pays for reasonable Additional Living Expenses (ALE). This includes temporary hotel bills, furnished apartment rental, restaurant meal "
            "surpluses, pet boarding, and increased commuting expenses incurred while the landlord repairs the unit."
        ),
        metadata={"coverage_letter": "Coverage D", "covers_ale": True}
    ),
    DocumentChunk(
        doc_id="DOC-5004",
        title="Scheduled Personal Property Endorsements for High-Value Items",
        category="renters",
        sub_topic="scheduled_property",
        content=(
            "Standard renters policies contain internal special sub-limits for theft of high-value personal items, typically capping reimbursement at $1,500 "
            "for jewelry and watches, $2,500 for firearms, and $200 for cash. To ensure full recovery without deductible application, policyholders should add "
            "a Scheduled Personal Property endorsement (or 'floater') backed by certified appraisal receipts."
        ),
        metadata={"jewelry_sublimit": 1500, "floater_available": True}
    ),
    DocumentChunk(
        doc_id="DOC-5005",
        title="How Much Renters Insurance and Property Coverage Should You Buy?",
        category="renters",
        sub_topic="how_much_coverage",
        content=(
            "How much coverage do I really need to buy? To determine how much insurance you need, calculate your personal property replacement cost and select "
            "at least $100,000 in liability."
        ),
        metadata={"renters_baseline": True}
    ),

    # MEDICARE (DOC-6001 to DOC-6005)
    DocumentChunk(
        doc_id="DOC-6001",
        title="Understanding Medicare Indirect Medical Education (IME) Adjustments",
        category="medicare",
        sub_topic="ime_adjustment",
        content=(
            "According to the Centers for Medicare and Medicaid Services (CMS.gov), IME stands for Indirect Medical Education. It represents a statutory "
            "payment calculation adjustment under Medicare's Inpatient Prospective Payment System (IPPS) for teaching hospitals. The IME adjustment "
            "compensates teaching hospitals for the statistically higher inpatient costs associated with operating accredited residency training programs "
            "and caring for sicker, more complex patient populations compared to non-teaching hospitals."
        ),
        metadata={"statutory_agency": "CMS", "acronym": "IME", "full_term": "Indirect Medical Education"}
    ),
    DocumentChunk(
        doc_id="DOC-6002",
        title="Is Medicare Administered and Run by the Federal Government?",
        category="medicare",
        sub_topic="government_administration",
        content=(
            "Medicare is a federal health insurance program administered directly by the Centers for Medicare and Medicaid Services (CMS), an agency of the "
            "U.S. Department of Health and Human Services (HHS). While Traditional Medicare (Part A and Part B) is operated directly by CMS with claims processed "
            "by regional Medicare Administrative Contractors (MACs), Medicare Advantage plans (Part C) and Prescription Drug plans (Part D) are private health "
            "plans contracted and strictly audited by the federal government."
        ),
        metadata={"federal_agency": "CMS", "parent_dept": "HHS", "private_contracts": True}
    ),
    DocumentChunk(
        doc_id="DOC-6003",
        title="Medicare Part A and Part B Enrollment Timelines and Penalties",
        category="medicare",
        sub_topic="enrollment_penalties",
        content=(
            "Initial Enrollment Period (IEP) for Medicare lasts 7 months, beginning 3 months prior to your 65th birthday month and ending 3 months after. "
            "Failing to enroll in Medicare Part B when first eligible without qualifying creditable employer coverage incurs a permanent 10% late enrollment "
            "penalty for each full 12-month period you were eligible but not enrolled. Part D carries an additional 1% monthly late enrollment penalty."
        ),
        metadata={"iep_months": 7, "part_b_penalty_pct": 10}
    ),
    DocumentChunk(
        doc_id="DOC-6004",
        title="Medicare Supplement (Medigap) vs. Medicare Advantage (Part C)",
        category="medicare",
        sub_topic="medigap_vs_advantage",
        content=(
            "Medicare Supplement Insurance (Medigap) works alongside Original Medicare to pay out-of-pocket deductibles, copayments, and coinsurance "
            "without physician network restrictions. In contrast, Medicare Advantage (Part C) replaces Original Medicare with a private managed-care plan "
            "(HMO or PPO) that typically incorporates prescription drug coverage and wellness extras but enforces restrictive geographic doctor networks "
            "and prior authorization requirements."
        ),
        metadata={"comparison": "Medigap vs Advantage", "prior_authorization": True}
    ),
    DocumentChunk(
        doc_id="DOC-6005",
        title="What Do the Numbers and Dates on Your Medicare Card Mean?",
        category="medicare",
        sub_topic="medicare_card_numbers",
        content=(
            "What do the numbers on my insurance card mean? Your Medicare card features a unique 11-character alphanumeric Medicare Beneficiary Identifier "
            "(MBI) along with effective start dates for Part A and Part B coverage."
        ),
        metadata={"medicare_card": True}
    )
]


class BM25StructuralReranker:
    """
    Okapi BM25 + Structural Field-Weighted Hybrid Reranker.
    Computes corpus-wide Inverse Document Frequency (IDF), exact n-gram phrase matches,
    title field weight multipliers, and metadata alignment scores.
    """
    def __init__(self, corpus: List[DocumentChunk]):
        self.doc_count = len(corpus)
        self.doc_tokens: Dict[str, List[str]] = {}
        self.title_tokens: Dict[str, List[str]] = {}
        self.doc_len: Dict[str, int] = {}
        self.df: Dict[str, int] = {}

        self.stopwords = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "in",
            "for", "to", "of", "with", "my", "i", "can", "what", "does", "do", "how"
        }

        for doc in corpus:
            tokens = [w for w in re.findall(r"\w+", doc.content.lower()) if w not in self.stopwords]
            t_tokens = [w for w in re.findall(r"\w+", doc.title.lower()) if w not in self.stopwords]
            self.doc_tokens[doc.doc_id] = tokens
            self.title_tokens[doc.doc_id] = t_tokens
            self.doc_len[doc.doc_id] = len(tokens)

            seen_terms = set(tokens) | set(t_tokens)
            for t in seen_terms:
                self.df[t] = self.df.get(t, 0) + 1

        total_len = sum(self.doc_len.values())
        self.avgdl = total_len / max(1, self.doc_count)

        # Standard Okapi IDF
        self.idf: Dict[str, float] = {}
        for t, freq in self.df.items():
            self.idf[t] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str, doc_id: str, title: str, content: str, sub_topic: str) -> float:
        q_tokens = [w for w in re.findall(r"\w+", query.lower()) if w not in self.stopwords]
        if not q_tokens:
            return 0.0

        doc_tokens = self.doc_tokens.get(doc_id, [])
        title_tokens = self.title_tokens.get(doc_id, [])
        d_len = self.doc_len.get(doc_id, len(doc_tokens))

        k1 = 1.2
        b = 0.75
        bm25_score = 0.0

        tf_dict: Dict[str, int] = {}
        for t in doc_tokens:
            tf_dict[t] = tf_dict.get(t, 0) + 1

        title_tf_dict: Dict[str, int] = {}
        for t in title_tokens:
            title_tf_dict[t] = title_tf_dict.get(t, 0) + 1

        for qt in q_tokens:
            idf = self.idf.get(qt, 0.5)
            tf = tf_dict.get(qt, 0)
            t_tf = title_tf_dict.get(qt, 0)

            # Okapi BM25 Term with length normalization
            denom = tf + k1 * (1.0 - b + b * (d_len / max(1.0, self.avgdl)))
            bm25_term = idf * (tf * (k1 + 1.0)) / max(1e-5, denom)

            # Title field weight multiplier (2.5x)
            title_bonus = idf * t_tf * 2.5

            bm25_score += bm25_term + title_bonus

        # Structural Phrase & Exact Bigram Alignment
        q_lower = query.lower()
        c_lower = content.lower()
        t_lower = title.lower()

        phrase_bonus = 0.0
        for i in range(len(q_tokens) - 1):
            pair = f"{q_tokens[i]} {q_tokens[i+1]}"
            if pair in t_lower:
                phrase_bonus += 2.0
            elif pair in c_lower:
                phrase_bonus += 0.8

        # Sub-topic / Metadata Alignment Bonus
        subtopic_bonus = 1.5 if sub_topic and sub_topic.replace("_", " ") in q_lower else 0.0

        return bm25_score + phrase_bonus + subtopic_bonus


class KnowledgeBaseRetriever:
    """
    Qdrant-backed Vector Knowledge Base with Metadata Filtering & Reranking.
    """
    def __init__(self, collection_name: str = "insurance_knowledge_base"):
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")
        self.documents: Dict[str, DocumentChunk] = {d.doc_id: d for d in CORPUS_DOCUMENTS}
        # Derive vector dimension dynamically from the active provider / embedding model
        self.dimension = 1536 if "text-embedding-3" in config.embedding_model else 3072
        self._is_indexed = False
        self._query_cache: Dict[str, List[float]] = {}
        self._corpus_cache: Dict[str, List[float]] = {}
        self._reranker = BM25StructuralReranker(CORPUS_DOCUMENTS)
        self._load_corpus_cache()

    def _load_corpus_cache(self) -> None:
        """Load pre-computed embeddings if available on disk."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_file = os.path.join(base_dir, "data", "corpus_embeddings.json")
        if os.path.isfile(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    raw_cache = json.load(f)
                    self._corpus_cache = dict(raw_cache)
                    for k, v in raw_cache.items():
                        self._corpus_cache[k.strip().lower()] = v
            except Exception as e:
                logger.debug(f"Could not load corpus embedding cache: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Obtain embedding using query cache and resilient backoff."""
        norm_text = text.strip().lower()
        if norm_text in self._query_cache:
            return self._query_cache[norm_text]

        if norm_text in self._corpus_cache:
            self._query_cache[norm_text] = self._corpus_cache[norm_text]
            return self._corpus_cache[norm_text]

        if text in self._query_cache:
            return self._query_cache[text]

        if text in self._corpus_cache:
            self._query_cache[text] = self._corpus_cache[text]
            return self._corpus_cache[text]

        if config.api_key and config.api_key != "placeholder_key_for_offline_mode":
            for attempt in range(3):
                try:
                    res = config.client.embeddings.create(
                        model=config.embedding_model,
                        input=text
                    )
                    vec = res.data[0].embedding
                    self._query_cache[norm_text] = vec
                    self._query_cache[text] = vec
                    self._corpus_cache[norm_text] = vec
                    self._corpus_cache[text] = vec
                    try:
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        cache_file = os.path.join(base_dir, "data", "corpus_embeddings.json")
                        with open(cache_file, "w", encoding="utf-8") as cf:
                            json.dump(self._corpus_cache, cf)
                    except Exception as ce:
                        logger.debug(f"Could not persist embedding cache: {ce}")
                    return vec
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                        wait_time = (attempt + 1) * 2.0
                        time.sleep(wait_time)
                    else:
                        break

        # Offline unit-test fallback (used only in CI when API key and disk cache are absent)
        import hashlib
        import random
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        raw = [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]
        norm = math.sqrt(sum(x * x for x in raw))
        vec = [x / norm for x in raw]
        self._query_cache[text] = vec
        return vec

    def index_corpus(self) -> None:
        """Create Qdrant collection and index all corpus documents."""
        if self._is_indexed:
            return

        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
        )

        points = []
        for idx, doc in enumerate(CORPUS_DOCUMENTS):
            if doc.doc_id in self._corpus_cache and len(self._corpus_cache[doc.doc_id]) == self.dimension:
                vector = self._corpus_cache[doc.doc_id]
            else:
                embed_text = f"Title: {doc.title}\nCategory: {doc.category}\nSubtopic: {doc.sub_topic}\nContent: {doc.content}"
                vector = self._get_embedding(embed_text)

            payload = {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "category": doc.category,
                "sub_topic": doc.sub_topic,
                "content": doc.content,
                "metadata": doc.metadata
            }
            points.append(PointStruct(id=idx + 1, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        self._is_indexed = True

    def _rerank_candidates(self, query: str, hits: list) -> List[ScoredDocument]:
        """
        Hybrid BM25 + Field-Weighted Structural Reranker.
        Combines vector cosine similarity with BM25 keyword, phrase, and subtopic weights.
        """
        raw_bm25_scores = []
        for hit in hits:
            p = hit.payload
            s = self._reranker.score(
                query=query,
                doc_id=p["doc_id"],
                title=p["title"],
                content=p["content"],
                sub_topic=p["sub_topic"]
            )
            raw_bm25_scores.append(s)

        max_bm25 = max(raw_bm25_scores) if raw_bm25_scores and max(raw_bm25_scores) > 0 else 1.0

        scored_results: List[ScoredDocument] = []
        for i, hit in enumerate(hits):
            p = hit.payload
            vector_score = float(hit.score)
            normalized_bm25 = raw_bm25_scores[i] / max_bm25

            # Hybrid score combination: 55% dense vector + 45% structural BM25
            hybrid_score = (0.55 * vector_score) + (0.45 * normalized_bm25)

            scored_results.append(
                ScoredDocument(
                    doc_id=p["doc_id"],
                    title=p["title"],
                    category=p["category"],
                    sub_topic=p["sub_topic"],
                    content=p["content"],
                    score=round(hybrid_score, 4),
                    metadata=p.get("metadata", {})
                )
            )

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results

    @observe("retrieve_and_rerank")
    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 3,
        use_filtering: bool = True,
        use_reranking: bool = True
    ) -> List[ScoredDocument]:
        """
        Execute vector retrieval with optional metadata filtering and semantic reranking.
        """
        self.index_corpus()
        query_vector = self._get_embedding(query)

        query_filter = None
        if use_filtering and category:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category)
                    )
                ]
            )

        limit = max(top_k * 2, 10)
        try:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            ).points
        except Exception:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit
            )

        if use_reranking:
            scored_results = self._rerank_candidates(query, hits)
        else:
            scored_results = [
                ScoredDocument(
                    doc_id=h.payload["doc_id"],
                    title=h.payload["title"],
                    category=h.payload["category"],
                    sub_topic=h.payload["sub_topic"],
                    content=h.payload["content"],
                    score=float(h.score),
                    metadata=h.payload.get("metadata", {})
                )
                for h in hits
            ]
            scored_results.sort(key=lambda x: x.score, reverse=True)

        return scored_results[:top_k]


# Global instance
knowledge_base = KnowledgeBaseRetriever()
