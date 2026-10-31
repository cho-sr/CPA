# 이전 80장 그대로 사용하는 packed4 CPA + FIA

기존 M6/M8의 80장을 **원래 순서와 16장 × 5개 배치**로 사용합니다.
`n=16`은 한 클라이언트 업데이트의 이미지 수이고, `n_images=80`은 전체 평가 이미지 수입니다.
80장을 하나의 클라이언트 업데이트에 합치는 구성이 아닙니다.

- 실행 코드: `qacpa/run_m10_packed4_same80.py`
- 설정: `configs/m10_cpa_fia_packed4_same80_B16.json`
- 입력: `results/m6_uia_fp32_B16_fia/batch00`부터 `batch04`까지의 `inputs.pt`, `observation.pt`
- 출력: `results/m10_cpa_fia_packed4_same80_B16/`
- 확정 이미지 목록: 출력 폴더의 `image_manifest.csv` (`global_index` 0–79)

원본 이미지와 VGG16 특징은 `inputs.pt`에서 읽습니다. ImageNet을 다시 샘플링하거나
과거 LPIPS로 이미지를 고르지 않습니다. 원래의 전처리 텐서, 레이블, 이미지 순서가
M8의 이미지 목록과 일치하는지 확인합니다. 캐시 파일은 기존 summary, meta,
compression protocol의 SHA256에 모두 대조합니다.

동일한 배치 구성이므로 기존의 실제 `(W_initial-W_after10)/lr` 업데이트를 재사용합니다.
학습을 다시 하지 않습니다. 각 업데이트를 기존의 실제 packed4 코덱으로 인코딩하고,
CPA에는 저장된 `observation.bin`을 디코딩한 텐서만 전달합니다.
FIA 입력은 추정 순서의 `abs(S_hat)`이며, 정답과의 매칭은 FIA 이후 평가에만 적용됩니다.

학습 조건은 기존 VGG16 round0, frozen conv, FC SGD lr=0.001, T=10입니다.
CPA와 FIA는 각각 배치당 25,000회입니다. 코덱 seed는 `1000003 * batch_index`,
CPA seed는 `batch_index`, FIA seed는 `42`입니다.

CPA + FIA packed4 조건만 포함합니다. 압축 크기는 `classifier.0.weight` 패킷에
한정하며, FP32 크기 계산은 FP32 공격 실험 결과가 아닙니다.

CPA 작업 폴더에서 PowerShell로 실행:

```powershell
Set-Location .\quantization_aware_cpa
# CPA + FIA 실행 (CUDA), 전체 완료 후 보고서 생성
& 'C:\miniconda\envs\DLG\python.exe' -u -m qacpa.run_m10_packed4_same80 --phase run
```

이미 `quantization_aware_cpa` 폴더에 있다면 `Set-Location` 줄을 생략합니다.
현재 PC에서는 `.ps1` 실행이 로컬 실행 정책에 의해 차단되어 Python을 직접 호출했습니다.
위 명령은 실행 정책을 변경하지 않습니다.

공격 실행 없이 준비만 확인하려면 같은 폴더에서 다음 명령을 사용합니다.

```powershell
# 원래 이미지/업데이트 검사만 (CPU)
& 'C:\miniconda\envs\DLG\python.exe' -u -m qacpa.run_m10_packed4_same80 --phase prepare
# 실제 패킷 5개 생성과 디코딩 검사 (CPU)
& 'C:\miniconda\envs\DLG\python.exe' -u -m qacpa.run_m10_packed4_same80 --phase packet
```

상태는 출력 폴더의 `execution_status.json`에 기록됩니다. `packet_ready`는
패킷 준비 완료 상태이고, 이미지 복원 완료 상태는 `complete`입니다.
동일 명령을 재실행하면 해시가 일치하는 완료 배치를 재사용합니다.
중단된 배치의 CPA/FIA는 해당 배치 처음부터 다시 실행합니다.

전체 완료 시 80장 전체와 배치별 원본/복원 그림, 모든 이미지의 LPIPS/PSNR/SSIM 및
소스 cosine, 배치별 수치, 패킷 크기, 실행 시간과 `results/runs.jsonl` 기록을 생성합니다.
기존 best10 실행 코드, 설정, 결과는 그대로 보존됩니다.

2026-09-10 준비 검증: 관련 CPU 테스트 90개 통과. 실제 원본80 및 업데이트5개의
해시를 확인했고, 패킷5개가 각각 51,396,640 bytes로 생성·디코딩 검증되었습니다.
이 준비 단계에서는 CPA/FIA GPU 공격을 시작하지 않았습니다.
