# Self-Reflective Generation at Test Time (SRGen)

## Introduction

Self-Reflective Generation at Test Time (SRGen), a lightweight test-time framework that reflects before generating at uncertain points. During token generation, SRGen utilizes dynamic entropy thresholding to identify high-uncertainty tokens. For each identified token, it trains a specific corrective vector, which fully exploits the already generated context for a self-reflective generation to correct the token probability distribution. By retrospectively analyzing the partial output, this self-reflection enables more trustworthy decisions, thereby significantly reducing the probability of errors at highly uncertain points.

## Getting Start

### Installation

```shell
pip install -r requirements.txt
```

### Run Example
```shell
bash scripts/parallel_aime_distill_qwen.sh
```