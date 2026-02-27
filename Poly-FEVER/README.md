---
language:
- en
- zh
- hi
- ar
- bn
- ja
- ko
- ta
- th
- ka
- am
size_categories:
- 10K<n<100K
task_categories:
- text-classification
---

# Dataset Card for Poly-FEVER

<!-- Provide a quick summary of the dataset. -->

This dataset card aims to be a base template for new datasets. It has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md?plain=1).

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->
Poly-FEVER is a multilingual fact verification benchmark designed to evaluate hallucination detection in large language models (LLMs). It extends three widely used fact-checking datasets—FEVER, Climate-FEVER, and SciFact—by translating claims into 11 languages, enabling cross-linguistic analysis of LLM performance.

Poly-FEVER consists of 77,973 factual claims with binary labels (SUPPORTS or REFUTES), making it suitable for benchmarking multilingual hallucination detection. The dataset covers various domains, including Arts, Science, Politics, and History.

- **Funded by [optional]:** Google Cloud Translation
- **Language(s) (NLP):** English(en), Mandarin Chinese (zh-CN), Hindi (hi), Arabic (ar), Bengali (bn), Japanese (ja), Korean (ko), Tamil (ta), Thai (th), Georgian (ka), and Amharic (am)

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [https://huggingface.co/papers/2503.16541]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

[More Information Needed]

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

[More Information Needed]

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

[More Information Needed]

### Source Data

FEVER: https://fever.ai/resources.html
CLIMATE-FEVER: https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html
SciFact: https://huggingface.co/datasets/allenai/scifact

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

[More Information Needed]

### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be made aware of the risks, biases and limitations of the dataset. More information needed for further recommendations.

## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Dataset Card Authors [optional]

[More Information Needed]

## Dataset Card Contact

Hanzhi Zhang