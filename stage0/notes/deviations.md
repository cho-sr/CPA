# Stage 0 — upstream 대비 deviation 목록

원칙: upstream `official_cpa/src/` 파일은 **한 줄도 수정하지 않는다**
(upstream `main`과 바이트 동일 상태를 유지, [upstream_verification.csv](upstream_verification.csv)).
필요한 변경은 전부 러너(`stage0/run_cpa.py`)의 monkeypatch 또는 별도 스크립트로 처리한다.

범례: **[BUG]** upstream 버그 우회 · **[ENV]** 환경 호환 · **[ADD]** 없는 기능 추가 ·
**[METRIC]** 지표 계산 정정

---

## [BUG] 1. conv 모델의 embedding `z`가 수집되지 않음

`collect_grads.py:104`가 `model.type`을 읽지만 모델이 정의한 속성은 `model_type`이다.
`nn.Module.type`은 실제 메서드이므로 `MyDataParallel.__getattr__`가 호출되지 않고
바운드 메서드가 반환된다 → `model_type == "conv"`가 항상 False →
`return_z=True` 분기를 타지 않아 `data["z"]`가 빈 리스트.
VGG-16 경로에서 `attack.py`가 `EmbEval`에 빈 텐서를 넘기게 된다.

* 처리: `stage0/collect_grads_stage0.py`가 `base.model_type`을 직접 읽는다.
  (러너에도 `patch_model_type_attr()`가 있어 upstream 수집기를 쓸 때도 우회 가능.)
* 영향: 없음(버그를 고치지 않으면 VGG-16 경로가 아예 실행 불가).

## [METRIC] 2. PSNR이 +9.54 dB 부풀려져 있음 — torchmetrics ≥1.0

`eval_utils.py:12-15`는

```python
from torchmetrics.functional import peak_signal_noise_ratio
```

로 import한다. torchmetrics 1.9.0에서 이 경로는 **deprecated shim**
(`torchmetrics/functional/image/_deprecated.py`)이고, 그 시그니처는

```python
def _peak_signal_noise_ratio(preds, target, data_range=3.0, ...)
```

즉 **`data_range` 기본값이 3.0**이다 (docstring 예제에서 흘러들어온 값).
정식 경로 `torchmetrics.functional.image.psnr.peak_signal_noise_ratio`는
`data_range`가 **필수 인자**다.

이미지는 `[0, 1]`이므로 올바른 `data_range`는 1.0이다. 따라서 upstream 코드가
찍는 PSNR은 전부 20·log10(3) = 9.54 dB 과대평가다. 실측 확인:

| | data_range=1.0 (정정) | upstream 경로 (data_range=3.0) |
|---|---|---|
| smoke b8 sample 0 | 11.83 | 21.37 |
| smoke b8 전 8샘플 | 9.40 – 12.96 | 18.95 – 22.50 |

**BRR/WSRR에 치명적이다.** 판정 지표가 `1[PSNR > 18 ∧ LPIPS < 0.1]`인데,
upstream 경로로 계산하면 `PSNR_upstream > 18` ⟺ `PSNR_true > 8.46 dB`가 되어
사실상 모든 이미지가 PSNR 조건을 통과한다. 위 smoke 예에서 참 PSNR은 8개 전부
18 미달인데 upstream 값은 8개 전부 18 초과다.

* 처리: `stage0/evaluate.py`가 `data_range=1.0`을 **명시**해 재계산한다.
  비교용으로 upstream 규약 값도 `psnr_upstream_datarange` 열에 함께 남긴다.
* SSIM은 무해: deprecated shim의 `data_range` 기본값이 `None`이라 데이터에서
  추론(≈0.98)하므로 1.0과 거의 같다 (smoke: upstream 0.0590 vs 정정 0.0605).
* LPIPS는 무해: `normalize_lpips`로 `[-1,1]`로 옮겨 넣는 것이 torchmetrics LPIPS의
  기본 규약(`normalize=False`)과 일치한다. **논문이 보고하는 지표는 LPIPS뿐이므로
  논문 숫자 자체는 이 문제의 영향을 받지 않는다.**
  (`official_cpa/requirements.txt`는 `torchmetrics`를 버전 고정하지 않는다.
  2023년 당시 0.x에서는 PSNR의 `data_range` 기본값이 `None`이었다.)

## [ENV/METRIC] 3. TF32를 끈다 — 기본 설정의 "FP32 baseline"은 FP32가 아니다

PyTorch는 `torch.backends.cudnn.allow_tf32`를 **기본 True**로 둔다. 이 환경 실측:

```
cudnn.allow_tf32     = True     <-- 기본값
matmul.allow_tf32    = False
f32_matmul_precision = highest
```

즉 conv는 TF32(10-bit 만티사), matmul은 full fp32라는 비대칭 상태다.
VGG-16 gradient에 대한 영향 (B=64, 동일 입력):

| 설정 | 전체배치 재실행 간 오차 | 마이크로배치(32) 합 vs 전체배치 |
|---|---|---|
| TF32 on (기본) | 0 | **4.8e-03** |
| TF32 off + cudnn.deterministic | 0 | **1.75e-07** |

그리고 전체배치 ‖dW‖_F 자체가 TF32 on/off에서 14.671363 vs 14.674352로
약 2e-4 상대 차이가 난다. B=128에서 재확인:

```
reference |g| = 13.398328   (unsplit B=128, tf32 off)
micro=128  rel_L2_err = 0.000e+00
micro= 64  rel_L2_err = 6.908e-07
micro= 32  rel_L2_err = 6.888e-07
```

**왜 무시할 수 없는가.** 저비트 quantization 노이즈는 이보다 훨씬 크므로
헤드라인 결론이 뒤집히지는 않는다. 그러나 (a) 이 프로젝트의 목적이
"gradient에 얹힌 노이즈가 공격을 어떻게 망가뜨리는가"를 정량화하는 것이므로
기준선에 통제되지 않은 0.5% 섭동이 있으면 안 되고, (b) CPA의 whitening은
1/√λ를 clipping 없이 적용하므로 (eps = 1e-20) 작은 고유값 방향의 5e-3 섭동이
whitened 좌표계에서 얼마나 커지는지는 자명하지 않다.

* 처리: `collect_grads_stage0.py`는 TF32를 **기본으로 끈다**
  (`cudnn.allow_tf32=False`, `matmul.allow_tf32=False`,
  `cudnn.deterministic=True`, `float32_matmul_precision="highest"`).
  `--tf32`로 켤 수 있고, 설정값은 pickle의 `stage0_meta`에 기록된다.
* attack 단계(FIA)는 TF32 기본값을 유지한다. FIA는 측정이 아니라 최적화이고
  달성 LPIPS(~0.48)가 5e-3 섭동보다 훨씬 조악하므로 무해하다. 대신 이 비대칭을
  여기에 명시해 둔다.
* 부수 효과: TF32를 끄면 마이크로배치 분할이 정당해진다(6.9e-07).
  TF32가 켜진 상태에서는 분할 합이 전체배치와 4.8e-03 어긋나므로
  `--micro_batch`를 쓰면서 TF32를 켜는 조합은 쓰지 말아야 한다.

## [ENV] 3b. `cudnn.benchmark`를 명시적으로 끈다

`utils.setup()` → `enable_gpu_benchmarking()`가 `cudnn.benchmark=True`를 설정한다.
`cudnn.deterministic=True`(위 3번)와 같이 켜져 있으면, 매 새로운 입력 shape마다
결정론적 알고리즘 집합 안에서 탐색을 한다. VRAM이 빠듯한 상태(마이크로배치 VGG-16
224px, GPU가 이미 거의 가득 참)에서 이 탐색이 수 GB 워크스페이스를 요구하는 후보들을
순서대로 시도하다 매번 OOM 경고만 찍고 수 분간 진행 없이 멈추는 것을 실측했다
(batch_size=256 수집 시 CPU는 계속 올라가지만 로그는 몇 분간 그대로).
`collect_grads_stage0.py`는 `setup()` 이후 `cudnn.benchmark=False`로 강제한다.

## [OPS] 3c. 장시간 실행 중 GPU 드라이버 리셋(TDR)이 관측됨 — checkpoint로 복구

headline 재현 실행(batch=256, official) 도중 batch 3 FIA iter ~8000 부근에서
파이썬 프로세스가 traceback 없이 조용히 종료됐다(exit code 1). GPU는 즉시
유휴 상태로 복귀했고, Windows System 이벤트 로그에 `nvlddmkm`(NVIDIA 디스플레이
드라이버) **Event ID 153**이 프로세스 종료 직후 시각에 기록되어 있었다 —
Windows TDR(Timeout Detection and Recovery) 리셋의 전형적인 신호다.
같은 세션 초반(수집 단계)에도 한 차례 더 기록되어 있었다.

이건 upstream 코드나 이 프로젝트의 버그가 아니라 **환경(GPU 드라이버) 안정성
문제**로 취급한다. 대응은 코드 수정이 아니라 운영 절차:

* upstream `AttackLog`는 매 로그 지점(`n_log`/`n_log_fi`)마다 `{batch_size}_ckpt.pkl`에
  `(batch, iter, attack_mode, W_hat/S_hat, optimizer state, 지금까지의 df)`를
  체크포인트한다. `--fresh_start`를 주지 않는 한 재실행 시 upstream이 자동으로
  이어받는다(`log_utils.py: check_preemption`).
* `run_grid.py`는 기본적으로 `--fresh_start`를 주지 않으므로, 죽은 job을 **동일한
  명령으로 재실행**하면 체크포인트에서 재개된다. 실측: batch 3, iter 8000,
  attack_mode="fi"에서 정확히 재개됨을 확인.
* 레지스트리의 TDR 지연시간(`TdrDelay`) 조정은 시스템 전역 설정이라 사용자 승인
  없이 변경하지 않는다. 리셋이 반복되면 별도로 제안한다.

## [ADD] 4. `--micro_batch` — batch 256은 분할 없이는 OOM

VGG-16 forward+backward를 256×3×224×224로 돌리면 16 GB VRAM에서
`cudaErrorMemoryAllocation`으로 죽는다(128은 통과). criterion이 평균이므로
전체배치 gradient는 정확히 Σ_c (n_c/B)·∇L_c 이고, 위 표대로 TF32를 끈 상태에서
fp32 round-off 수준으로 정확하다.

Stage 0은 `--micro_batch=128`을 쓴다 → batch 8–128은 **분할 없음(upstream과 동일)**,
batch 256만 2×128로 분할.

## [ADD] 5. `--seed` 추가

upstream은 `utils.setup()`에서 `set_seed(42)`를 하드코딩하고 CLI seed 인자가 없다.
`run_cpa.py: patch_seed()`가 `setup`을 감싸 이후 원하는 seed로 재설정한다.
**`--seed 42`는 upstream과 완전히 동일한 동작**이다.

## [ADD] 6. `--w_init` 추가 — WSRR을 위해 필수

`gradient_inversion.py:94`의 `W_hat` 초기화는 `torch.nn.init.eye_`,
즉 **결정론적 항등행렬**이다. 같은 gradient에 대해 upstream CPA를 몇 번 돌려도
결과가 완전히 동일하므로, WSRR이 요구하는 "10개 독립 초기화"를
upstream 코드로는 만들 수 없다.

`run_cpa.py: patch_w_init()`가 `gradient_inversion.attack_dict["cp"]`를
서브클래스로 교체해 `eye`(기본, upstream) / `randn` / `orthogonal` /
`eye_perturbed`를 선택 가능하게 한다.

* 주의: upstream이 평균을 내는 축은 `--n_batch`(서로 다른 데이터 배치)다.
  초기화 축과 혼동하지 않는다. Stage 0은 두 축을 분리해 보고한다.

## [ADD] 7. `--grad_scope=attack_layer`

upstream 수집기는 모든 파라미터의 gradient를 저장한다. VGG-16은 138.4 M 파라미터라
배치당 553 MB이고 6×10 격자면 약 33 GB다. CPA와 Direct FIA가 실제로 읽는 텐서는
`grads[26]` 하나뿐이므로(Direct FIA는 gradient를 아예 쓰지 않음), 나머지는
zero-size placeholder로 저장해 리스트 인덱싱을 유지한다.

* `--grad_scope=full`로 upstream과 동일한 내용을 저장할 수 있다.
* **제약**: `--attack=gm`(GMA baseline)이나 `--fi_method=gm`은 전체 gradient가
  필요하므로 `attack_layer` 스코프의 pickle로는 실행 불가. Stage 0 범위는 CPA +
  Direct FIA뿐이라 문제되지 않지만, GMA 대조군을 돌릴 때는 `full`로 다시 수집해야 한다.

## [ENV] 8. torchtext shim

`datasets.py`가 module scope에서 torchtext를 import하지만(범위 밖인 IMDB 파이프라인용),
설치된 torchtext 0.18은 torch 2.13 아래에서 `libtorchtext.pyd` 로드에 실패한다
(`OSError: [WinError 127]`).

`sys.modules` 런타임 stub만으로는 부족하다: Windows `spawn`이 DataLoader worker에서
**upstream 스크립트를 파일 경로로 재실행**하기 때문에(runpy가 `__main__.__file__`을
upstream 스크립트로 바꿔놓음) 자식은 stub 없는 새 인터프리터에서 import한다.
`sys.path`는 spawn preparation data에 포함되어 상속되므로,
`stage0/shims/torchtext/`를 `sys.path` 앞에 꽂아 부모·자식 모두에서 shadow한다.
실제로 **사용**하려 하면 raise하므로 텍스트 실험 결과를 조용히 바꿀 수 없다.

## [ENV] 9. wandb / matplotlib / lpips

* wandb: inert stub. 모든 지표는 `_iter.pkl` / `_summary.pkl` / `_rec.pkl`에도
  기록되므로 정보 손실이 없다.
* matplotlib: `log_utils.get_pil_plot`이 쓰는 `FigureCanvasAgg.tostring_rgb`가
  Matplotlib 3.10에서 제거되어 `buffer_rgba` 기반으로 재정의.
* `lpips==0.1.4` 설치 (torchmetrics LPIPS의 런타임 의존성).

## [ENV] 10. DataLoader `num_workers`

upstream 기본값 8. Windows spawn 문제는 위 8번 shim으로 해결되어 8을 쓸 수 있다.
러너의 `--num_workers`로 조절 (attack 단계는 dataloader를 쓰지 않으므로 0).

---

## 재현 범위에 대한 정직한 한계

### L1. FIA는 배치 전체가 아니라 16 샘플에만 적용된다

README headline 명령의 `--n_sample_fi=16`이 그대로다. `utils.subsample`은
**앞쪽 16개**를 취한다(`input[:n_sample]`). 배치 순서는 `shuffle_test=True`로
무작위이므로 배치 평균 LPIPS의 부분표본 추정치로 볼 수 있으나,
논문 Table 2가 배치 전원을 역변환한 값인지는 논문에 명시되어 있지 않다.

배치 전원(256장) FIA는 이 환경에서 불가능하다: VGG-16 forward+backward를
256×3×224×224로 25 000 iter 돌리는 것은 16 GB VRAM에 들어가지 않고,
16샘플 실측 속도(≈80 ms/iter)로 환산하면 배치당 약 9시간이다.
논문 저자 설정은 submitit 기준 GPU 8장 / 메모리 400 GB다.

→ Stage 0은 배치당 16 샘플 × 배치 10개 = 샘플 160개로 LPIPS를 추정하고
표준편차를 함께 보고한다.

### L2. 논문은 5배치 평균, upstream 기본값은 10배치

논문: "We repeat the attack on 5 batches of data and report the average LPIPS scores."
upstream `--n_batch` 기본값은 10. Stage 0은 10을 쓰고(요구사항의 "≥10" 충족),
논문과의 대조 시 이 차이를 명시한다.

### L3. batch 512 / 1024는 미수집

논문 Table 2는 32–1024를 다룬다. 512/1024의 FedSGD gradient는
`--micro_batch`로 수집 자체는 가능하다(위 4번). Stage 0 요구 격자는 8–256이므로
후순위로 둔다.

### L4. batch 8 / 16은 논문에 없음

요구 격자의 8, 16은 논문 Table 2 범위(32–1024) 밖이다. 대조 불가, 신규 측정치로만 보고.

### L5. 판정선 LPIPS < 0.1은 VGG-16/ImageNet에서 달성 불가능하다

논문 Table 2의 CPA+FIA LPIPS는 batch 32–1024에서 **0.479 – 0.509**다.
즉 공격의 상한선 자체가 판정선(LPIPS < 0.1)보다 5배 나쁘다.
FP32 baseline에서 이미 BRR = WSRR = 0이 되므로, 이 축으로는 quantization의
저하폭을 측정할 여지가 없다. 상세와 대안은 [README.md](../README.md) 참조.
