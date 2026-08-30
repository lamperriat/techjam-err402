# V2 Offline Attribute Extraction Proposal

## Decision

Use one small shared schema for every product, plus a list of open-ended factual
attributes. Do not maintain separate closed schemas for clothing, shoes, and
jewelry.

This is a deliberate simplification of the published approaches:

- Amazon's attribute-graph system defines reusable category attributes and also
  permits additional product-specific attributes during extraction. It uses
  structured catalog fields directly and invokes the LLM only for missing
  attributes.
- ExtractGPT supplies a target schema, asks for JSON, preserves the surface form
  of values, and represents missing values explicitly.
- MAVE grounds every value in a source paragraph and character span.
- OA-Mine demonstrates why open-world attributes are needed when the useful
  attribute set is not known in advance.

The full Amazon system maintains thousands of category schemas. Reproducing
that hierarchy is not appropriate for this catalog or this project. The shared
core keeps common values comparable, while `specific_attributes` handles
watches, bags, sunglasses, gemstones, and other long-tail product types without
enumerating them in advance.

The papers do not identify one universally optimal attribute list. Their common
result is that schemas are category-aware, values should be grounded, and the
long tail needs either many schemas or open attributes. The six LLM fields below
are therefore a project decision: they are the intersection of generic facets
used in production extraction work (such as material, color, size, style, and
feature), the evaluator vocabulary, and fields visibly supported by this
catalog. The open list is what prevents that pragmatic core from becoming a
claim of universal completeness.

## Proposed stored representation

Fields already present in the catalog are copied without an LLM call:

| Field | Source | Notes |
|---|---|---|
| `category_path` | `categories` | Preserve the complete path. |
| `department` | `details.Department` | Extract from text only when missing. |
| `brand` | brand-like detail fields, then `store` | Keep the source so later calibration can distinguish a catalog brand from a store fallback. |
| `price` | `price` | Keep missing prices as `null`; never estimate them. |
| `average_rating` | `average_rating` | Numeric ranking feature, not an extracted attribute. |
| `rating_number` | `rating_number` | Numeric popularity feature, not an extracted attribute. |

The LLM extracts only these six fields:

| Field | Definition | Examples of valid evidence |
|---|---|---|
| `material` | Explicit component, fabric, metal, or gemstone material. Multiple values are allowed. | `100% cotton`, `sterling silver`, `rubber outsole` |
| `color` | Explicit color or color combination. | `navy blue`, `rose gold tone` |
| `size_fit` | Explicit size, width, fit, length, or wearable dimension. | `6.5 Wide`, `slim fit`, `18-inch chain` |
| `style` | Explicit design style, pattern, shape, silhouette, or finish. | `A-line`, `floral print`, `matte finish` |
| `use_case` | Explicit activity, occasion, audience scenario, terrain, or weather context. | `trail running`, `wedding`, `cold weather` |
| `specific_attributes` | Objective product-specific attribute/value pairs that do not fit the shared fields. | `movement: Japanese quartz`, `water_resistance: 50 m`, `cushioning: memory foam footbed`, `gemstone_cut: princess cut` |

`specific_attributes` is a list rather than one undifferentiated `feature`
value. Each entry has a short normalized attribute name, a value, and exact
catalog evidence:

```json
{
  "name": "cushioning",
  "value": "memory foam footbed",
  "evidence": "Memory foam footbed provides responsive cushioning"
}
```

The five shared fields use the same evidence-bearing value representation:

```json
{
  "material": [
    {
      "value": "sterling silver",
      "evidence": "crafted from 925 sterling silver"
    }
  ],
  "color": [],
  "size_fit": [],
  "style": [],
  "use_case": [],
  "specific_attributes": []
}
```

An empty list means that no supported value was found. It does not mean that
the product lacks that property.

## Grounding and subjectivity rules

Every stored value must be supported by an exact substring of the supplied
title, feature bullets, description, or details. The preprocessing code should
drop an entry when its evidence cannot be found after conservative whitespace
and case normalization.

The LLM performs extraction, not product evaluation:

- Do not output scores, probabilities, rankings, or inferred quality levels.
- Do not infer a use case, material, weather capability, or audience from
  general product knowledge.
- Do not store unsupported judgments such as `comfortable`, `durable`,
  `premium`, `reliable`, `beautiful`, or `good value`.
- Concrete mechanisms and specifications are valid. For example, extract
  `cushioning: memory foam footbed`, not `comfort: high`; extract
  `construction: reinforced toe`, not `durability: 0.91`.
- A vendor claim such as `waterproof` may be extracted only as a claim present
  in the catalog text. The pipeline does not certify that the claim is true.
- Replace the proposed `gemstone_quality` field with objective facts such as
  gemstone type, natural/lab-created origin, treatment, grade, clarity, cut,
  and carat weight when explicitly stated.
- Do not extract `long_term_value`. It is a prediction, not a catalog
  attribute.
- Ethical sourcing is a valid product-specific fact only when the text contains
  a concrete certification or explicit sourcing claim.

## Why these fields fit this catalog

The current catalog routing produces 31,608 fallback clothing products, 12,820
shoe products, and 5,572 jewelry products. The fallback is intentionally broad:
its common leaves include T-shirts, 1,034 wrist watches, 540 sunglasses,
wallets, and costumes. A garment-only schema would therefore create systematic
nulls for legitimate products.

Structured detail coverage is also too sparse to replace text extraction. For
example, explicit `Material` detail coverage is 4.0% for the fallback group,
0.7% for shoes, and 12.6% for jewelry. Text contains useful objective facts much
more often. Conservative phrase checks found, among other examples:

| Evidence family | Fallback clothing | Shoes | Jewelry |
|---|---:|---:|---:|
| closure | 29.5% | 30.7% | 2.3% |
| stretch/flexibility | 23.1% | 16.1% | 4.7% |
| breathability | 14.1% | 14.8% | 0.2% |
| cushioning | 1.2% | 24.0% | 0.6% |
| water protection | 5.7% | 7.3% | 1.5% |
| pockets | 20.6% | 0.3% | 0.8% |
| metal | 4.4% | 0.5% | 66.2% |
| gemstone | 2.0% | 1.6% | 29.3% |
| personalization | 3.0% | 1.8% | 12.3% |
| watch movement | 2.4% | 0.0% | 0.0% |

These are regex-based lower-bound indicators, not extraction labels or quality
measurements. Their purpose is only to show that several useful product-specific
facts recur, but their relevance depends on the actual product type.

## Relationship to the evaluator

The richer offline fields remain compatible with the evaluator's broad
`ask_attribute` contract:

| Offline field | Evaluator attribute |
|---|---|
| `material` | `material` |
| `color` | `color` |
| `size_fit` | `size` |
| `style` or `department` | `style` |
| `use_case` | `use_case` |
| one named `specific_attributes` facet | `feature` |
| catalog brand | `brand` |
| catalog price | `budget` |

For example, V2 can ask “Would cushioning or arch support matter to you?” while
returning `ask_attribute="feature"`. The candidate statistics are calculated
for the fine-grained name (`cushioning`), not for a generic “feature present”
flag. A product-specific attribute remains eligible only if it occurs on at
least the existing minimum number of products in the top candidate set.

## Pilot experiment

Before preprocessing all 50,000 products:

1. Sample 300 products with a fixed seed: 100 from each current question group,
   with a per-leaf cap so the fallback sample includes watches and accessories
   rather than mostly T-shirts.
2. Extract one product per non-streaming JSON request with model thinking
   disabled, temperature zero, and an 800-token output cap. Record prompt and
   completion tokens.
3. Keep at most three values per shared field and five product-specific facts.
   Evidence is the shortest supporting phrase and is limited to 80 characters.
4. Validate the JSON shape and exact evidence. Invalid entries are logged and
   discarded rather than repaired by inference. Store only the extracted delta;
   direct catalog fields remain joinable through `parent_asin` and are not
   duplicated in the pilot output.
5. Manually audit a fixed 60-product subset for unsupported values, missed
   useful facts, and subjective judgments.
6. Report field coverage, values per product, evidence rejection rate, null
   rate, attribute-name frequency, and name fragmentation such as
   `waterproofing` versus `water_resistance`.
7. Freeze a small alias table only for recurring naming collisions observed in
   the pilot. Do not invent a comprehensive ontology.

## Frozen post-processing

The full-catalog post-processor keeps the raw JSONL immutable and writes a
separate processed artifact. Its alias table consolidates only recurring names
observed in this extraction, centered on the `closure`, `cushioning`,
`slip_resistance`, and `water_resistance` families. Values remain unchanged, so
claims such as `waterproof` and `water resistant` are still distinguishable.

An evidence-rejected entry is restored only when its value matches a complete
token-level substring in the same compact catalog source supplied to the LLM.
The stored evidence becomes that exact source slice and remains subject to the
80-character limit and the original subjectivity and schema validation. No
fuzzy matching, value rewriting, reclassification, or inference is performed.

The pilot succeeds if stored values are fully evidence-grounded after
validation, manual precision is high enough to use the data for filtering, and
the recurring product-specific names are consistent enough to pass the current
five-product question threshold. Recall is secondary: missing a fact is safer
than adding an unsupported constraint.

## Research basis

- [From Unstructured to Structured: LLM-Guided Attribute Graphs for Entity Search and Ranking](https://www.amazon.science/publications/from-unstructured-to-structured-llm-guided-attribute-graphs-for-entity-search-and-ranking)
- [Generative Models for Product Attribute Extraction](https://aclanthology.org/2023.emnlp-industry.55/)
- [ExtractGPT: Exploring the Potential of Large Language Models for Product Attribute Value Extraction](https://arxiv.org/abs/2310.12537)
- [MAVE: A Product Dataset for Multi-source Attribute Value Extraction](https://github.com/google-research-datasets/MAVE)
- [OA-Mine: Open-World Attribute Mining for E-Commerce Products with Weak Supervision](https://arxiv.org/abs/2204.13874)
- [Explicit Attribute Extraction in E-Commerce Search](https://aclanthology.org/2024.ecnlp-1.13/)
