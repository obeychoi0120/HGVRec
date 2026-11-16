# 실행·저장 규칙

데이터 입력, 산출물 경로, 재실행 규칙을 정리합니다. 설치와 기본 실행은 [README](../README.md), 장면 스키마는 [장면 추출](scene_extraction.md), 모델·평가는 [추천 학습](graph_training.md)을 참고하세요.

## 데이터 입력과 cohort

모든 명령은 `--dataset microlens_100k` 또는 `--dataset m3l_10m`을 받으며 한 번에 한 데이터셋을 처리합니다. `config.yaml`의 상대 경로는 저장소 루트 기준이고, 선택하지 않은 데이터셋의 원본 파일은 없어도 됩니다.

### MicroLens

`datasets.microlens_100k.data`에 interaction CSV(`pairs_csv`), 영상 디렉터리(`videos_dir`), 제목 CSV(`titles_csv`), 선택적 보완 제목 CSV(`titles_supplement_csv`)를 지정합니다.

- interaction CSV는 `user,item,timestamp` 형식이며 timestamp는 정수 밀리초입니다.
- 영상 파일 이름은 `{item_id}.mp4`입니다.
- 제목은 원본의 비어 있지 않은 값을 먼저 쓰고, 보완 CSV로 빈 제목과 누락 행을 채웁니다. 보완 CSV로도 채우지 못한 제목은 빈 문자열로 남습니다.

### M³L-10M

`datasets.m3l_10m.data.root` 아래 다음 파일을 사용합니다.

```text
metadata/ratings.dat
metadata/movies.dat
metadata/id_mmrec_to_grouplens.tsv
videos/{original_GroupLens_movieId}.mp4
```

- ID 매핑에 있는 영화의 평점만 사용하고, 평점 timestamp(초)를 밀리초로 바꿉니다.
- 제목은 `movies.dat`의 연도 포함 제목입니다.
- 모든 평점을 상호작용으로 쓰며 평점 값은 가중치로 쓰지 않습니다.
- 영상이 없거나 재생시간을 확인할 수 없는 영화와 그 상호작용은 제외합니다.
- `source_counts.interaction_count`는 진행 표시줄의 전체 길이로만 씁니다.

이 설정은 평가한 영화 이력으로 다음 영화를 예측하는 문제이며, 공식 MMRec 분할을 재현하지 않습니다.

### 공유 준비와 Run별 분할

- `prepare-cohort`는 원본 데이터·영상·제목을 읽어 데이터셋별 공유 cohort를 만듭니다.
- `prepare-input-data`는 영상의 시간 창과 키프레임을 준비합니다.
- 다른 원본 corpus를 쓰려면 별도 `artifacts_root`를 지정합니다.

Run별 `validation/cohort/`는 `validation.cohort.min_user_interactions`개 이상 이력이 있는 사용자의 최근 `retained_sequence_length`개 이벤트를 남기고, 마지막 두 이벤트를 validation/test로 둡니다. 기본값은 5와 13입니다. 후보 catalog는 남긴 시퀀스에 나오는 아이템입니다. cohort 설정을 바꿨다면 새 Run ID를 쓰거나 `validation/cohort/`를 지운 뒤 다시 만듭니다.

## 설정과 실행 자원

- `config.yaml`은 데이터·모델·추출·학습·평가 설정입니다.
  - 필수 섹션은 `datasets`, `models`, `extraction`(`visual_evidence` 포함), `validation`뿐입니다.
  - 그 밖의 키를 추가해도 거부하지 않습니다.
  - `validation` 아래 값은 [기본값](../src/validation/config.py)이 있어 생략할 수 있고, 모델 크기·seed·cutoff도 자유롭게 바꿀 수 있습니다.
- `rec_experiments_config.yaml`은 추천 실험 목록입니다.
  - validation 명령은 이 파일 전체를 YAML 순서대로 처리합니다.
  - 추출·전처리 단계는 이 파일 없이 실행할 수 있습니다.
- Qwen은 로컬 checkpoint를 vLLM으로 불러오며, 보이는 GPU마다 worker를 하나씩 띄웁니다. `CUDA_VISIBLE_DEVICES`로 사용할 GPU를 고릅니다.
- Gemini의 동시 요청 수는 `extraction.gemini.threads`입니다.
- 추천 학습의 `--workers-per-gpu`는 GPU당 동시에 학습하는 독립 조합 수입니다. GPU가 없으면 1로 두고 CPU를 씁니다.

## 산출물 경로

```text
<artifacts_root>/
├── preparation/<dataset>/
│   ├── cohort/leave_one_out/
│   ├── resized_keyframes/<width>_<height>/<content_id>/
│   └── source_assets/
├── runs/<RUN_ID>/<dataset>/
│   ├── extraction/<source>/
│   ├── validation/
│   │   ├── cohort/
│   │   ├── representations/<source>/
│   │   ├── recommendations/seed_<seed>/<label>/
│   │   └── diagnosis/<experiment-set>/diagnosis.json
│   └── reports/
└── shared_cache/<dataset>/
    ├── scene_inputs/
    └── recommendations/
```

- `<experiment-set>`은 YAML의 label을 순서대로 밑줄로 이은 이름입니다.
- source는 `metadata`이거나 `graph_<model>`·`desc_<model>` 형식의 시각 source입니다.
- 같은 source의 고정 BGE 특징은 여러 실험이 공유하고, 학습 파라미터는 실험마다 따로 학습합니다.
- 별도로 작성하는 Run 분석 코드와 보고서는 [AGENTS.md](../AGENTS.md)에 따라 `artifacts/runs/<RUN_ID>/reports/`에 저장합니다.

## 장면 저장

Graph·Description·실패 결과는 source 디렉터리의 `<content_id>.jsonl`에 저장합니다. 행 필드는 [장면 추출 문서](scene_extraction.md#저장과-실패-확인)를 따릅니다.

- `(content_id, scene_idx)`마다 최신 결과 한 행을 유지하고, 임시 파일에 쓴 뒤 교체합니다.
- `generation_stats.json`은 결과별 장면 수를 기록합니다.
- `generation_identity.json`은 처음 생성할 때의 프롬프트 경로·해시, 모델 경로, 생성 설정을 남기는 기록입니다. 재사용 여부 판단에는 쓰지 않습니다. 생성 옵션을 바꿔 이어서 실행하면 한 폴더에 서로 다른 조건의 장면이 섞일 수 있습니다.

## 재실행과 캐시

결과 파일이 있으면 완료로 보고 건너뜁니다. 설정·코드·프롬프트를 바꾼 뒤 결과를 새로 만들려면 해당 단계를 `--force`로 다시 실행하거나 새 Run ID를 씁니다.

| 단계 | 일반 재실행 | `--force` |
| --- | --- | --- |
| 장면 추출 | 정상 장면은 재사용하고, 누락·error·warning 장면만 생성 | 해당 source의 장면을 모두 지우고 다시 생성 |
| 고정 특징(`embed-representations`) | `manifest.json`이 있으면 재사용. 없으면 공유 캐시를 찾고, 그래도 없으면 계산 | 다시 계산 |
| 활동 구간(`prepare-module`) | 필요한 구간 파일을 다시 생성 | 같음 |
| 추천 조합 | `complete.json`이 있으면 재사용. 없으면 공유 캐시를 찾고, 그래도 없으면 학습. epoch 중간 재개는 없음 | 다시 학습 |
| 진단 | 완료된 조합의 결과를 모아 통계와 보고서를 생성 | — |

- 장면 추출 결과는 다른 Run에서 자동으로 가져오지 않습니다.
- 공유 캐시는 키 디렉터리 단위로 복사하고 rename으로 게시합니다.
  - 특징 키: 장면 파일 내용, catalog·제목, BGE 경로, 인코더 설정
  - 추천 키: 실험 정의, 특징 키, 분할, 학습 설정(seed 제외), seed
- 추천 결과는 학습 당시의 키로 공유 캐시에 올립니다. 그래서 설정을 바꾼 뒤 기존 결과를 재사용해도, 그 결과가 새 설정의 결과로 공유되지는 않습니다.
- 특징을 `--force`로 다시 만들면, 이전 bundle로 만든 활동 구간은 사용하지 않습니다. `prepare-module --force`로 구간을 다시 만듭니다.

## 학습 결과와 진단 필드

| 파일·필드 | 의미 |
| --- | --- |
| `sasrec.pt` | 선택된 checkpoint의 `state_dict`와 metadata |
| `training.json` | validation 이력, best epoch, 학습 조건, 파라미터 수, 실행 통계, 환경 정보 |
| `per_event_metrics.jsonl` | test 이벤트별 순위와 지표 |
| `catalog_vectors.npy` | 선택된 모델의 catalog 영상 벡터 |
| `complete.json` | 완료 표시. 학습 당시 identity와 이벤트 수 |
| 진단 `experiments` | label·source·modules·readout |
| `recommendations.means`, `standard_deviations`, `per_seed` | 최종 test의 평균, 표본 표준편차, seed별 값 |
| `recommendations.baseline`, `base_relative_changes` | metadata source label과 상대 변화율. 기준 실험이 없으면 `null` |
| `statistics.comparisons` | 모든 쌍의 NDCG 차이와 보정 신뢰구간 |
| `runtime_decision` | 진단 실행의 성공 여부와 오류 |

- checkpoint는 validation HR@10으로 고르고, 최종 보고와 paired bootstrap의 주 지표는 test NDCG@10입니다.
- `--compare-run-id`는 두 Run의 cohort 이벤트와 분할이 같을 때만 비교합니다.

## 결정적 실행

`validation.model.deterministic: true`이면 PyTorch 결정적 알고리즘을 켭니다. CUDA 행렬 연산용 `CUBLAS_WORKSPACE_CONFIG=:4096:8`은 값이 없을 때만 설정합니다. 하드웨어·드라이버·라이브러리 버전 차이까지 맞춰 주지는 않습니다.
