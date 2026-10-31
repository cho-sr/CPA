# CPA 공개 코드 리딩 노트 (Stage 0, 과제 2 · 3)

대상: `facebookresearch/cocktail_party_attack` @ `main`
로컬 스냅샷: `official_cpa/src/` (= `main/src/`)

## 0. 스냅샷 무결성 확인

`official_cpa/src/` 13개 파일 전부 upstream `main` 브랜치와 **바이트 단위 동일**
(줄바꿈 정규화 후 SHA-256 비교). 해시 목록은
[upstream_verification.csv](upstream_verification.csv).

즉 아래 서술은 로컬 포크의 특성이 아니라 **저자가 공개한 코드 그 자체**의 특성이다.
(참고: `cocktail_party_attack/src/`와 `baseline/src/`는 이전 작업에서 수정된
변형본이므로 baseline 기준으로 쓰면 안 된다. `main/src/`는 official과 동일.)

---

## 1. 과제 2 — `--nv` 인자의 정체

**결론: `--nv`는 노이즈 분산(noise variance)과 무관하다. "negative value penalty",
즉 복원된 각 source를 한쪽 부호로 몰아주는 부호 모호성 해소 prior다.
따라서 이후 단계의 whitening 설계에 직접적인 영향을 주지 않는다.**

### 정의 위치

선언 — [attack.py:205](../../official_cpa/src/attack.py#L205):

```python
parser.add_argument("--nv", type=float, default=0, help="negative value penalty")
```

구현 — [gradient_inversion.py:177-180](../../official_cpa/src/gradient_inversion.py#L177-L180):

```python
if self.nv > 0:
    loss_nv = torch.minimum(
        F.relu(-S_hat).norm(dim=-1), F.relu(S_hat).norm(dim=-1)
    ).mean()
```

즉

$$R_{\mathrm{nv}}(\hat S) \;=\; \frac{1}{n}\sum_{i=1}^{n}
\min\Big(\;\big\|\,[-\hat s_i]_+\,\big\|_2,\;\; \big\|\,[\hat s_i]_+\,\big\|_2\;\Big)$$

### 왜 이 항이 필요한가

ICA는 source를 **부호와 스케일까지만** 복원한다 ($\hat s_i \approx \pm c\, s_i$).
VGG-16에서 공격 대상 source는 `classifier[0]`의 입력, 즉 **post-ReLU / post-avgpool
feature $z \ge 0$** 이다 (`vgg16.py`의 `forward`: `z = flatten(avgpool(features(x)))`,
`features`의 마지막 층이 `MaxPool2d(ReLU(...))`). 따라서 참 source는 한쪽 부호(비음수)다.

$R_{\mathrm{nv}}$는 각 행에서 **양수부 에너지와 음수부 에너지 중 작은 쪽**을 벌점화한다.
따라서 행 전체를 "전부 ≥ 0" 또는 "전부 ≤ 0" 중 가까운 쪽으로 밀어붙인다.
전역 부호는 여전히 자유롭지만 — 그건 [attack.py:118](../../official_cpa/src/attack.py#L118)에서
해결한다:

```python
rec_emb = rec_gi_reord.abs()   # FIA 입력으로 |ŝ| 를 사용
```

`|ŝ|`가 참 $z$가 되려면 $\hat s$가 한쪽 부호여야 한다. 즉 `--nv`는 이 `.abs()`를
정당화하는 항이다. 노이즈 모델링이 아니다.

### 부수적으로 발견한 게이팅 (중요)

[gradient_inversion.py:174](../../official_cpa/src/gradient_inversion.py#L174):

```python
if self.tv > 0 and self.nv == 0:  # if nv > 0, tv is meant for the generator
    loss_tv = self.total_variation(S_hat.view(self.inp_shape))
```

`--nv > 0`이면 **`--tv`는 gradient inversion 단계에서 아예 쓰이지 않는다.**
대신 같은 `args.tv`가 feature inversion 단계에서 소비된다
([feature_inversion.py:25](../../official_cpa/src/feature_inversion.py#L25) `self.tv = self.args.tv`).

→ README의 ImageNet headline 명령 `--nv=0.13 --tv=0.1`에서 `tv=0.1`은
**GI의 prior가 아니라 FIA의 TV prior다.** conv 모델은 `inp_shape = [n_comp, -1]`이라
GI 단계에서 이미지 TV를 계산할 수도 없다(2-D view). 일관적이다.

### 헤드라인 설정에서의 실제 값

| 설정 | `--nv` | GI에서 `--tv` 적용? | `--l1` |
|---|---|---|---|
| fc2 / TinyImageNet / b128 | 0 (기본값) | **적용** (tv=3.1) | 0 |
| VGG-16 / ImageNet / b256 | 0.13 | **미적용** (FIA로 넘어감) | 5 |

---

## 2. 과제 3 — PCA whitening 사용 여부와 정확한 형태

**결론: 쓴다. PCA whitening + 차원축소(`n_obs → n_comp = batch_size`)를
공격 루프 진입 전에 딱 한 번 수행하고, 결과를 디스크에 캐시한다.**

### 관측 행렬 $X$

[gradient_inversion.py:72](../../official_cpa/src/gradient_inversion.py#L72):

```python
self.X = self.grads[self.model.attack_index]
```

`grads`는 `model.parameters()` 순서의 gradient 리스트다.

| 모델 | `attack_index` | 대상 파라미터 | $X$ shape | source |
|---|---|---|---|---|
| `FC2` | 0 ([models.py:74](../../official_cpa/src/models.py#L74)) | `net[0].weight` | `[256, 12288]` | 입력 이미지 $x_b$ (3·64·64) |
| `VGG_mod` | 26 ([vgg16.py:92](../../official_cpa/src/vgg16.py#L92)) | `classifier[0].weight` | `[4096, 25088]` | post-ReLU feature $z_b$ |

(VGG16: `features`의 conv 13개 × (weight, bias) = 인덱스 0..25, 따라서 26 =
`classifier[0].weight`, 즉 `Linear(25088, 4096)`.)

$\partial L/\partial W$의 $i$번째 **행**은
$\sum_b \delta_{b,i}\, s_b$ 로, source들의 선형결합이다.
→ 관측 수 $n_{\mathrm{obs}}$ = out_features (fc2: 256 = `h_dim`, vgg16: 4096),
관측 차원 $D$ = in_features.

**제약**: `n_comp = args.batch_size`이므로 `batch_size ≤ n_obs`.
fc2는 `h_dim=256`이라 batch 256이 상한이고, batch 256에서는 $X$가 정사각(256×256 cov,
전 고유값 사용)이 되어 차원축소가 사라진다.

### 전처리 1: zero-centering

[gradient_inversion.py:109-111](../../official_cpa/src/gradient_inversion.py#L109-L111):

```python
def zero_center(self, x):
    x_mu = x.mean(dim=-1, keepdims=True)
    return x - x_mu, x_mu
```

`dim=-1`, 즉 **feature 축(D) 방향 평균**을 관측 행마다 뺀다.
표준 ICA의 "샘플 평균 제거"가 아니라 각 관측 행의 DC 성분 제거다.
나중에 [line 166](../../official_cpa/src/gradient_inversion.py#L166)에서 복원한다:

```python
S_hat = S_hat + torch.matmul(torch.matmul(W_hat_norm, self.W_w), self.X_mu)
```

### 전처리 2: whitening

[gradient_inversion.py:113-123](../../official_cpa/src/gradient_inversion.py#L113-L123):

```python
def whiten(self, x):
    cov = torch.matmul(x, x.T) / (x.shape[1] - 1)
    eig_vals, eig_vecs = torch.linalg.eig(cov)
    topk_indices = torch.topk(eig_vals.float().abs(), self.n_comp)[1]

    U = eig_vecs.float()
    lamb = eig_vals.float()[topk_indices].abs()
    lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + self.eps)).float()
    W = torch.matmul(lamb_inv_sqrt, U.T[topk_indices]).float()
    x_w = torch.matmul(W, x)
    return x_w, W
```

수식으로:

$$\Sigma = \frac{X_{zc} X_{zc}^\top}{D-1} \in \mathbb{R}^{n_{\mathrm{obs}} \times n_{\mathrm{obs}}},
\qquad \Sigma = U \Lambda U^{-1}$$

$\mathcal{K}$ = $|\lambda|$ 상위 `n_comp`개 인덱스,

$$W_w = \Lambda_{\mathcal{K}}^{-1/2} U_{:,\mathcal{K}}^\top \in \mathbb{R}^{n_{\mathrm{comp}} \times n_{\mathrm{obs}}},
\qquad X_w = W_w X_{zc} \in \mathbb{R}^{n_{\mathrm{comp}} \times D}$$

결과적으로 $X_w X_w^\top / (D-1) \approx I$. **표준 PCA whitening이다** (ZCA 아님,
Cholesky 아님).

### 코드 품질 관련 주의점 (이후 단계에 영향 있음)

1. **`torch.linalg.eig` (일반 고유분해)를 대칭행렬에 사용한다.** `eigh`가 아니다.
   반환값이 complex이고, `eig_vecs.float()` / `eig_vals.float()`로 **complex→float
   캐스팅으로 허수부를 버린다**. 수치적으로 대칭행렬이면 허수부가 ~0이라 실질 문제는
   없지만, 조건수가 나빠지면(저비트 quantization이 정확히 그 상황) 허수부가 커지고
   고유벡터 정렬도 흔들릴 수 있다. **Stage 1 이후 `eigh`로 바꿀지 여부는
   의도적 결정이 필요하며, 바꾸면 baseline과 비교 불가**하므로
   "eig 유지 + 허수부 크기 로깅"을 권한다.

2. **$1/\sqrt{\lambda}$ 증폭.** `eps = 1e-20`이라 사실상 clipping이 없다.
   작은 고유값 방향이 그대로 $1/\sqrt{\lambda}$배 증폭된다.
   양자화 오차는 gradient에 거의 등방성(+heavy-tail) 성분을 얹어 하위 고유값을
   부풀리므로, **whitening이 양자화 노이즈를 선택적으로 증폭하는 경로**가 된다.
   이것이 Stage 1 이후 실패 메커니즘의 1차 후보다.

3. **whitening 캐시 키가 gradient 내용을 반영하지 않는다.**
   [gradient_inversion.py:71](../../official_cpa/src/gradient_inversion.py#L71):

   ```python
   self.w_pkl_file = f"{exp_path}/w_{self.n_comp}_{self.batch_id}.pkl"
   ```

   `exp_path`는 `decor/T/tv/nv/l1`(+`defense`, `ds`, `model`, `attack`)만으로 결정된다
   ([utils.py:247-266](../../official_cpa/src/utils.py#L247-L266)).
   → **gradient를 양자화해도 `exp_name`이 같으면 FP32 whitening을 그대로 재사용한다.**
   Stage 1 이후 최악의 silent bug 후보. 실험마다 `--exp_name`을 반드시 분리하고,
   `--fresh_start`가 whitening 캐시를 지우지 않는다는 점(지우는 건 `_ckpt.pkl`뿐,
   [log_utils.py:151-154](../../official_cpa/src/log_utils.py#L151-L154))도 기억해야 한다.

### unmixing 최적화

```python
W_hat = eye(n_comp, n_comp)                       # 결정론적 항등 초기화
W_hat_norm = W_hat / ||W_hat||_row                # 매 step 행 정규화
S_hat = W_hat_norm @ X_w                          # 여기서 negentropy 평가
S_hat = S_hat + W_hat_norm @ W_w @ X_mu           # centering 복원 후 prior 평가
```

Adam, `lr=1e-3`, `n_iter=25000`, scheduler 없음.

---

## 3. 추가 발견 — 공개 코드에는 negentropy 항이 없다 (중대)

[gradient_inversion.py:155-163](../../official_cpa/src/gradient_inversion.py#L155-L163):

```python
if self.ne > 0:
    loss_ne = -(
        (
            (1 / self.a)
            * torch.log(torch.cosh(self.a * S_hat) + self.eps).mean(dim=-1)
        )
        ** 2
    ).mean()
    loss_ne = torch.tensor(0.0, device=self.device)   # <-- 계산 결과를 0으로 덮어씀
```

그리고 [line 185-191](../../official_cpa/src/gradient_inversion.py#L185-L191):

```python
loss = (
    loss_ne                       # <-- self.ne 가중치도 곱하지 않음
    + (self.decor * loss_decor)
    + (self.tv * loss_tv)
    + (self.nv * loss_nv)
    + (self.l1 * loss_l1)
)
```

**따라서 공개된 CPA는 실제로 다음을 최적화한다:**

$$U = \arg\min_U \;\; \lambda_{\mathrm{decor}} R_{\mathrm{MI}}(U)
+ \lambda_{\mathrm{tv}} R_{\mathrm{TV}} + \lambda_{\mathrm{nv}} R_{\mathrm{nv}}
+ \lambda_{\ell_1} R_{\ell_1}$$

즉 **negentropy $J$ 항이 없다.** `--ne` 인자는 죽은 분기를 게이팅할 뿐 손실에 전혀
기여하지 않는다. 논문 식과 어긋난다:

$$U = \arg\max_U \; \mathbb{E}\big[J(u_i G) - \lambda_P R_P(u_i G)\big] - \lambda_{\mathrm{MI}} R_{\mathrm{MI}}$$

덧붙여, 죽은 코드 자체도 논문의 negentropy 대리항이 아니다. 올바른 log-cosh contrast는

$$J(y) \propto \big(\mathbb{E}[\log\cosh y] - \mathbb{E}[\log\cosh v]\big)^2,\quad v \sim \mathcal N(0,1)$$

이고 $y$가 **평균 0 · 분산 1로 표준화**되어 있어야 한다. 공개 코드에는
(a) 가우시안 기준항 $\mathbb{E}[\log\cosh v] \approx 0.37457$ 이 없고,
(b) source 표준화가 없다 (whitening은 $X_w$의 행을 백색화하지만
$U X_w$의 각 행 분산은 $\|u_i\|_2^2$에 비례하며 행 정규화만 걸려 있어 1이 아니다).

### 왜 이게 Stage 0의 핵심인가

산출물 규칙에 "최적화 실패와 정보 손실을 반드시 구분"하라고 되어 있다.
그 구분은 목적함수가 무엇인지 확정된 다음에만 가능하다. 지금 상태로는
"CPA의 상한선"이 두 가지로 갈린다:

* **A. `--src official`** — upstream 그대로. negentropy 없음.
  실제로 논문 표 숫자를 만든 코드일 가능성이 높다(공개 스냅샷이므로).
  이 경우 CPA는 사실상 *decorrelation + source prior* 최적화이며,
  분리를 만들어내는 힘은 `R_MI`(행 간 비유사성)와 prior다.
* **B. `--src negentropy`** — 논문 식대로 log-cosh contrast를 복원
  (`cocktail_party_attack/src/gradient_inversion.py`, 표준화 + 가우시안 기준항 +
  수치 안정 `log cosh`, 그리고 `self.ne * loss_ne`로 가중).

Stage 0에서는 **두 변형을 모두 측정**한다. 이유:
저비트 quantization의 오차 분포는 극도로 super-Gaussian(sparse, heavy-tailed)이므로,
negentropy 항이 있는지 없는지가 실패 메커니즘 예측을 정반대로 바꾼다.
negentropy가 있으면 공격은 "가장 non-Gaussian한 방향"을 찾는데, 양자화 오차 자체가
non-Gaussian이라 **오차 방향을 source로 오인**할 수 있다. negentropy가 없으면
그 경로는 존재하지 않고 whitening 증폭과 prior만 남는다.
이 구분 없이 Stage 1을 진행하면 결론을 해석할 수 없다.

---

## 4. 그 밖의 상세 사항 (재현 시 영향)

### 4.1 `R_MI` (decorrelation)는 대각 성분을 포함한다

[gradient_inversion.py:169-171](../../official_cpa/src/gradient_inversion.py#L169-L171):

```python
if self.decor > 0:
    cos_matrix = torch.matmul(W_hat_norm, W_hat_norm.T).abs()
    loss_decor = (torch.exp(cos_matrix * self.T) - 1).mean()
```

주석은 "j>i 상삼각"이라고 하지만 구현은 **전체 $n^2$ 성분 평균**이다.
행 정규화 후 대각은 정확히 1이므로 상수 $\frac{n(e^{T}-1)}{n^2}$ 만큼의 바닥값이 생긴다.
$T=12.4, n=128$에서 약 $1.9\times10^3$ — 로그에 찍히는 `loss_decor`는 이 상수가 지배한다.
gradient에는 (대각항이 상수 1이라) 영향이 없지만 **손실 절대값으로 수렴을 판단하면 안 된다.**

### 4.2 평가는 oracle 정렬 · oracle 부호 선택이다

`eval_utils.Eval.fix_order`는 **참값 $S$를 써서** 복원 source를 greedy 매칭한다
(fc/image는 SSIM, embedding은 $|\cos|$). `fix_sign_method="best"`는
참값 MSE를 비교해 샘플별로 $x$ 대 $1-x$ 중 좋은 쪽을 고른다
([eval_utils.py:120-125](../../official_cpa/src/eval_utils.py#L120-L125)).

→ 보고되는 PSNR/SSIM/LPIPS는 모두 **oracle-aligned upper bound**다.
Stage 0의 BRR/WSRR도 같은 정렬을 쓰는 한 같은 성질을 물려받는다.
표에 반드시 "oracle permutation/sign aligned, evaluation only"로 표기한다.

### 4.3 seed와 초기화 — WSRR을 위해 반드시 손을 봐야 한다

* `utils.setup()`이 `set_seed(42)`를 호출한다. **CLI seed 인자가 없다.**
* `W_hat` 초기화는 `torch.nn.init.eye_`, 즉 **완전 결정론적 항등행렬**이다
  ([gradient_inversion.py:94](../../official_cpa/src/gradient_inversion.py#L94)).

→ 같은 gradient에 대해 upstream CPA를 여러 번 돌리면 **항상 같은 결과**가 나온다.
"10개 독립 초기화"(WSRR)와 "10개 random seed 평균"은 upstream 코드로는 불가능하다.
Stage 0에서 `--seed`와 `W_hat` 랜덤 초기화 옵션을 추가해야 하며,
이는 명시적 deviation으로 기록한다. 상세는 [deviations.md](deviations.md).

upstream이 평균을 내는 축은 `--n_batch=10`, 즉 **서로 다른 데이터 배치 10개**다.
이 축과 초기화 축을 혼동하면 안 된다.

### 4.4 `exp_path` 충돌 위험

`exp_name`은 `decor/T/tv/nv/l1`만으로 만들어진다. `lr`, `n_iter`, `ne`, `seed`,
gradient 출처는 경로에 반영되지 않는다. 서로 다른 실험이 같은 디렉터리를 공유하면
`_ckpt.pkl` 이어받기와 whitening 캐시 재사용이 조용히 발생한다.
**Stage 0 이후 모든 실행은 `--exp_name`을 명시한다.**

### 4.5 upstream 버그: `collect_grads.py`가 conv 모델의 `z`를 저장하지 않는다

[collect_grads.py:104](../../official_cpa/src/collect_grads.py#L104):

```python
model_type = model.type          # 모델이 정의한 속성명은 model_type
```

`nn.Module.type`은 실제 메서드이므로 `MyDataParallel.__getattr__`가 호출되지 않고
바운드 메서드가 반환된다. 따라서 `model_type == "conv"`는 **항상 False**이고
[line 131-134](../../official_cpa/src/collect_grads.py#L131-L134)의 `return_z=True`
분기를 타지 않아 `data["z"]`가 빈 리스트로 남는다.
그 결과 VGG-16 경로에서 `attack.py`의 `EmbEval(emb, ...)`가 빈 텐서를 받는다.

Stage 0은 러너에서 최소 수정(`model.type`에 model_type 문자열을 shadow)으로 우회한다
(`stage0/run_cpa.py: patch_model_type_attr`). upstream 파일은 건드리지 않는다.

### 4.6 환경 shim

| 문제 | 조치 |
|---|---|
| `datasets.py`가 module scope에서 `torchtext` import (torch 2.13에 설치 불가) | 사용 시 raise하는 stub. 이미지 파이프라인은 torchtext를 쓰지 않음 |
| `log_utils`가 wandb 필수 | inert stub. 모든 지표는 `_iter.pkl`/`_summary.pkl`에도 기록되므로 손실 없음 |
| `log_utils.get_pil_plot`가 `FigureCanvasAgg.tostring_rgb` (Matplotlib 3.10에서 제거) | `buffer_rgba` 기반으로 재정의 |
| `get_dataloaders(num_workers=8)` + Windows spawn | 러너에서 shim을 import 시점에 설치해 `__mp_main__` 자식도 동작하게 함 |
| `torchmetrics` LPIPS가 `lpips` 패키지 요구 | `lpips==0.1.4` 설치 |

### 4.7 `train.py`의 무해한 버그

[train.py:64](../../official_cpa/src/train.py#L64): `get_sch(args.sch, args.opt, args.epochs)`
— optimizer 자리에 optimizer **문자열**을 넘긴다. 기본값 `--sch=none`에서는
`get_sch`가 즉시 `None`을 반환하므로 문제되지 않는다. `--sch=cosine`으로 학습하면
터진다. Stage 0은 기본값(`none`)을 쓰므로 그대로 둔다.

---

## 5. 요약: `--nv` / whitening이 이후 단계에 주는 함의

1. `--nv`는 노이즈 분산이 아니라 **source 단측 부호 prior**다.
   whitening 설계와 무관. 단, 양자화 후에도 참 source가 여전히 $z \ge 0$이므로
   이 prior는 계속 유효하며, 오히려 양자화로 부호 정보가 망가질 때
   `.abs()` 경로의 신뢰도가 떨어지는 것을 방어하는 항으로 재해석할 수 있다.
2. whitening은 **PCA whitening + `n_obs → batch_size` 차원축소**이며
   $1/\sqrt{\lambda}$에 clipping이 없다. 이것이 양자화 노이즈 증폭의 주 경로다.
   Stage 1에서 다뤄야 할 것은 "uniform 노이즈 가정"이 아니라
   **$\Sigma$의 스펙트럼이 어떻게 오염되는지**다.
3. 공개 코드에 negentropy가 없다는 점 때문에, "CPA baseline"은 두 변형으로
   측정해야 한다. 그렇지 않으면 저비트에서의 실패를 잘못된 메커니즘에 귀속시킨다.
