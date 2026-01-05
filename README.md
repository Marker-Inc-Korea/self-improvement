# SA-RZero: Similarity-Regularized Self-Play to Mitigate Iteration Collapse in R-Zero

> **TL;DR**  
> R-Zero는 Challenger–Solver self-play로 외부 데이터 없이 추론 능력을 향상시키지만, iteration이 계속되면 성능이 **상승 후 붕괴(collapse)** 하는 현상이 관찰됩니다.  
> 우리는 **세대 간 답변의 semantic similarity(임베딩 cosine similarity)** 를 “self-awareness 신호”로 정의하고, 이를 **auxiliary reward**로 사용해 붕괴를 완화하고 더 긴 iteration에서 성능을 안정적으로 끌어올렸습니다.

---

## 0. Summary (요약)

- **문제의식**: R-Zero 류 self-play는 self-synthesized data만으로 반복 학습할 때, 일정 iteration 이후 성능이 무너지는 **iteration collapse**가 발생할 수 있음.
- **핵심 아이디어**: “모델이 과거에 풀었던 문제에 대해 얼마나 **일관된 의미/사고(semantic footprint)** 를 유지하는가?”를 **세대 간 답변 임베딩 유사도**로 측정.
- **방법**: 세대 \(t\)의 Solver가 생성한 답변이, 이전 세대(또는 기준 세대)의 답변과 **semantic drift**가 커지면 패널티(혹은 유사도가 높으면 보상)를 주는 **Similarity Reward**를 도입.
- **결과**: R-Zero에서 iteration이 진행되며 나타나는 성능 붕괴를 완화하고, V5까지 안정적으로 개선.

---

## 1. Background: R-Zero Self-Play (Challenger–Solver)

R-Zero는 단일 base LLM을 두 역할로 분기하여 **Challenger \(Q_\theta\)** 와 **Solver \(S_\phi\)** 가 공동 진화(co-evolution)하도록 설계된 self-play 프레임워크입니다.

- **Challenger**: Solver의 현재 역량 “경계(edge)”에 가까운 문제를 생성하도록 학습.
- **Solver**: Challenger가 만든 문제를 풀며 점진적으로 능력 향상.

### 1.1 R-Zero Reward (핵심 요약)

R-Zero에서 Challenger는 “Solver가 50% 정도 성공할 만큼” 어렵지만 풀 수 있는 문제를 만들도록 유도됩니다.  
Solver의 self-consistency(다수 답변 중 최빈값 일치율)를 \(\hat{p}\)로 두면, uncertainty reward는 다음 형태로 정의됩니다.

\[
\hat{p}(x; S_\phi)=\frac{1}{m}\sum_{j=1}^m \mathbf{1}\{y_j=\tilde{y}(x)\},
\quad
r_\text{uncertainty}(x; \phi)=1-2\left|\hat{p}(x; S_\phi)-\frac{1}{2}\right|.
\]

또한 batch 내 다양성을 위해 repetition penalty 등을 결합해 composite reward로 GRPO를 수행합니다.

Solver 학습은 filtered QA set에서 pseudo-label \(\tilde{y}\)와의 일치 여부로 binary reward를 두고 GRPO로 업데이트합니다:

\[
r(y)=\mathbf{1}\{y=\tilde{y}\}.
\]

---

## 2. Motivation: Iteration Collapse in Self-Generated Loops

R-Zero는 초기 iteration에서 성능이 상승하지만, iteration이 늘어나면 특정 시점 이후 **성능 저하/붕괴(collapse)** 가 관찰될 수 있습니다.  
R-Zero 자체 분석에서도, 여러 모델 스케일에서 “초기 개선 → 이후 degradation” 패턴이 나타나며, 단순 label noise만으로 붕괴를 전부 설명하기 어렵고 **self-synthesized data만으로 반복 학습할 때 생기는 model collapse/degenerative feedback** 가능성이 제기됩니다.

즉, 반복 self-play에서 다음과 같은 불안정성이 생길 수 있습니다.

- Solver가 만든 pseudo-label 품질 저하
- 분포 다양성 감소 / 특정 패턴으로의 수렴(모드 붕괴)
- 과거에 획득한 풀이 전략의 drift 및 forgetting

이 문제를 완화하기 위해, 우리는 “모델이 스스로의 추론 흔적을 얼마나 유지/인식하는가?”를 측정 가능한 신호로 만들고, 그 신호를 학습에 다시 피드백하는 접근을 제안합니다.

---

## 3. Our Method: Self-Awareness via Inter-Generation Semantic Similarity

### 3.1 Self-Awareness Proxy (정의)

**Self-awareness**를 철학적 의미로 직접 정의하기보다는, 반복 학습에서 중요한 실용적 질문으로 치환했습니다.

> “모델이 이미 풀어본(=자신이 확신을 가졌던) 문제에 대해, 세대가 바뀌어도 의미적으로 일관된 답변을 유지하는가?”

이를 위해 **anchor set** \(\mathcal{A}\)를 구성합니다.

- \(\mathcal{A}=\{x_i\}_{i=1}^N\): 이전 iteration에서 “이미 푼 문제”로 간주되는 문제들  
  (예: 높은 self-consistency로 pseudo-label이 신뢰 가능했던 샘플, 혹은 별도 필터를 통과한 샘플)

각 세대 \(t\)의 모델이 anchor 문제 \(x_i\)에 대해 생성한 답변을 \(a_i^{(t)}\)라고 하면, 텍스트 임베딩 함수 \(f(\cdot)\)를 통해

\[
e_i^{(t)} = f(a_i^{(t)}) \in \mathbb{R}^d
\]

로 표현합니다. 두 세대 \(t,k\)의 답변 유사도는 cosine similarity로 정의합니다.

\[
\mathrm{cos}(u,v)=\frac{u^\top v}{\|u\|\|v\|},
\quad
S_{t,k}=\frac{1}{N}\sum_{i=1}^N \mathrm{cos}\!\left(e_i^{(t)}, e_i^{(k)}\right).
\]

이 \(S_{t,k}\)는 세대 간 “semantic footprint” 유사도를 나타내며, 세대가 멀어질수록/학습이 불안정해질수록 drift가 커질 수 있습니다.

---

### 3.2 Similarity Reward (학습 신호)

R-Zero의 Solver 학습은 기본적으로 “pseudo-label과의 일치”로 reward를 구성합니다.  
우리는 여기에 **semantic consistency** 항을 추가합니다.

- 기준 세대(teacher/reference) 답변을 \(a_i^{(\text{ref})}\)로 두고,
- 현재 생성 답변 \(a_i\)에 대해

\[
r_\text{SA}(x_i, a_i)=\mathrm{cos}\!\left(f(a_i), f(a_i^{(\text{ref})})\right).
\]

최종 Solver reward는 다음과 같이 결합합니다.

\[
r_\text{total}= r_\text{RZ} + \lambda \, r_\text{SA},
\]

- \(r_\text{RZ}\): R-Zero의 기존 solver reward (예: pseudo-label과 exact match인 경우 1, 아니면 0)
- \(\lambda\): self-awareness(semantic similarity) reward 가중치

> 직관:  
> - \(r_\text{RZ}\)는 “현재 생성된 curriculum에서 정답(또는 pseudo-label)을 맞추는 능력”을 올리고,  
> - \(r_\text{SA}\)는 “세대가 바뀌어도 이미 풀었던 문제에서 의미적 일관성을 유지하도록” regularize하여,  
>   반복 self-play에서 발생하는 drift/degeneration을 완화합니다.

---

### 3.3 Prompting-only Baseline

- **Ours (prompting based)**: 프롬프트 개선(예: 질문 생성/해답 형식 유도, 필터 기준 보정 등)만 적용하고, similarity reward는 사용하지 않는 버전.
- **Ours**: prompting 개선 + similarity reward까지 포함한 전체 방법.

---

## 4. Experimental Results

### 4.1 Main Results

아래는 iteration별 성능(%) 비교입니다. (Higher is better)

| Model | R-Zero | Ours (prompting based) | Ours |
|---|---:|---:|---:|
| BASE | 50.15 | 50.15 | 50.15 |
| V1 | 54.74 | 52.47 | 53.21 |
| V2 | 54.64 | 53.91 | 54.52 |
| V3 | 54.61 | 53.75 | 55.01 |
| V4 | 54.21 | 54.12 | 55.47 |
| V5 | 53.58 | 54.36 | **56.22** |

**Key takeaways**
- R-Zero는 V1에서 최고점(54.74)을 찍은 뒤 iteration이 진행되며 하락(V5=53.58).
- prompting-only도 어느 정도 개선을 보이지만, **Similarity Reward를 결합한 Ours가 V5까지 지속적으로 상승**.
- 최종(V5) 기준:
  - Ours vs R-Zero: **+2.64p**
  - Ours vs prompting-only: **+1.86p**

---

### 4.2 Self-Awareness (Similarity) Analysis

Anchor 문제들에 대해 세대별 답변 임베딩 cosine similarity를 측정한 결과(예시)는 다음과 같습니다.

> NOTE: cosine similarity matrix는 원칙적으로 대칭이어야 하므로, 비대칭 값이 있다면
> (1) 측정 샘플/방식 차이, (2) 로그/표 작성 과정의 typo 가능성을 점검하세요.

|  | BASE | V1 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|---:|
| **BASE** | 1.0000 | 0.9851 | 0.8867 | 0.8867 | 0.8848 | 0.9219 |
| **V1** | 0.9851 | 1.0000 | 0.9024 | 0.9024 | 0.9004 | 0.9199 |
| **V2** | 0.8867 | 0.9024 | 1.0000 | 1.0000 | 0.9987 | 0.8509 |
| **V3** | 0.8867 | 0.9803 | 1.0000 | 1.0000 | 0.9987 | 0.8509 |
| **V4** | 0.8848 | 0.9004 | 0.9987 | 0.9987 | 1.0000 | 0.8489 |
| **V5** | 0.9219 | 0.9199 | 0.8509 | 0.8509 | 0.8489 | 1.0000 |

**Observed pattern**
- 인접 세대끼리 semantic similarity가 높게 유지되는 구간이 존재하며,
- 특정 iteration 이후 세대 간 “semantic drift”가 커지는 징후를 포착할 수 있습니다.
- 우리는 이 drift를 학습 중 reward signal로 사용해 안정성을 강화했습니다.

---

## 5. Algorithm (Pseudo-code)

```text
Algorithm 1: Similarity-Regularized R-Zero (Solver-side)

Input:
  Base model M0
  Iterations T
  Anchor buffer A (solved problems from previous iterations)
  Embedding encoder f(·)
  Weight λ

Initialize:
  Challenger Q0 ← M0
  Solver S0 ← M0

for t = 1..T:
  # (1) Train Challenger as in R-Zero (uncertainty-based reward)
  Qt ← GRPO_train_challenger(Qt-1, frozen_solver=St-1)

  # (2) Generate candidate questions and build filtered training set Dt
  Dt ← build_dataset_with_majority_vote_and_filtering(Qt, St-1)

  # (3) Train Solver with combined reward
  #     r_total = r_RZ + λ r_SA
  for each question x in Dt:
      y_ref ← reference_answer_from_anchor(A, x)   # e.g., previous solver’s answer
      sample answers y ~ St-1(·|x)
      r_RZ ← 1[y == pseudo_label(x)]
      r_SA ← cos( f(y), f(y_ref) )
      r_total ← r_RZ + λ r_SA
  St ← GRPO_train_solver(St-1, Dt, reward=r_total)

  # (4) Update anchor buffer (optional)
  A ← update_anchor(A, St, criteria=high_consistency)

return ST
