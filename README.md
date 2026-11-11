# agent_deep_research

리서치 에이전트(Claude Code + deep-research 스킬)의 작업 공간입니다. 에이전트가 주제별로 자료를 조사해 노트를 쌓고 보고서를 작성하며, 그 결과물과 작업 기록이 이 저장소에 남습니다.

이 문서는 사람을 위한 안내서입니다. 에이전트가 따르는 규칙은 [`CLAUDE.md`](CLAUDE.md)에 있습니다.

## 어디부터 보면 되나

| 알고 싶은 것 | 볼 곳 |
|---|---|
| 지금까지 어떤 리서치를 했나 | [`INDEX.md`](INDEX.md)의 Tasks 표 |
| 특정 주제의 문서가 어디 있나 | [`INDEX.md`](INDEX.md)의 Tags 목록 |
| 어떤 작업을 언제 맡겨서 언제 끝났나 | [`log/events.tsv`](log/events.tsv) |
| 특정 논문·URL을 인용한 문서 | [`index/sources.tsv`](index/sources.tsv) |
| 보고서 본문 | `tasks/T-xxxx_<주제>/R-xxxx_<제목>.md` |

## 디렉토리 구조

```
CLAUDE.md            에이전트 작업 규칙
INDEX.md             진입점: 작업 목록, 태그 색인 (자동 생성)
index/               전체 문서 목록, 문서 간 연결, 인용 URL 색인 (자동 생성)
log/events.tsv       작업·통신 기록 (추가만 가능)
tasks/T-0001_<slug>/ 작업 1건 = 디렉토리 1개
  task.md            의뢰 내용, 범위, 결과 요약
  N-0001_<slug>.md   리서치 노트 (하위 질문 하나당 한 개)
  R-0001_<slug>.md   보고서 (납품물)
tools/dr.py          관리 도구
requirements.txt     도구 의존성
```

문서 종류는 세 가지이고 각각 고유 ID를 가집니다.

| ID | 종류 | 내용 |
|---|---|---|
| `T-0001` | 작업 | 의뢰받은 리서치 한 건 |
| `N-0001` | 노트 | 하위 질문 하나에 대해 수집한 근거 |
| `R-0001` | 보고서 | 노트를 종합한 최종 결과물 |

문서끼리는 경로가 아니라 ID로 연결됩니다. 파일을 옮기거나 이름을 바꿔도 연결이 유지됩니다.

## 설치

Python 3.9 이상이 필요합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

의존성은 PyYAML 하나입니다. 시스템 Python에 이미 설치되어 있으면 가상환경 없이 실행해도 됩니다.

## 리서치 맡기기

이 디렉토리에서 Claude Code를 실행하고 평소처럼 리서치를 요청하면 됩니다. 한국어로 요청해도 됩니다. 에이전트가 `CLAUDE.md`에 따라 다음을 수행합니다.

1. 작업을 등록하고 ID(`T-xxxx`)를 발급합니다.
2. 기존 노트와 인용 색인을 먼저 확인해 이미 조사한 내용은 재사용합니다.
3. 하위 질문별로 노트를 작성하고 보고서로 종합합니다.
4. 납품과 종료를 로그에 남기고 색인을 갱신합니다.

저장되는 문서는 용량과 토큰을 줄이기 위해 모두 영어로 작성됩니다. 대화 중 답변은 요청한 언어로 받습니다. 이 README만 예외로 한국어입니다.

## 직접 조회하고 관리하기

저장소 루트에서 `tools/dr.py <명령>`으로 실행합니다. 각 명령의 옵션은 `-h`로 확인할 수 있습니다.

### 조회

```bash
tools/dr.py show T-0001                  # 작업 하나: 소속 문서, 연결, 로그
tools/dr.py show R-0001                  # 보고서가 어떤 노트를 근거로 했는지
tools/dr.py find --tag prompt-ensemble   # 태그로 검색
tools/dr.py find --text "medoid"         # 제목·본문 정규식 검색
tools/dr.py find --url arxiv.org/abs/2203.11171   # 특정 URL을 인용한 문서
```

검색 조건(`--ids --type --task --tag --status --text --url`)은 함께 쓰면 모두 만족하는 문서만 나옵니다.

### 일괄 수정

```bash
tools/dr.py set status=superseded --ids N-0002,N-0004 --dry-run   # 미리 보기
tools/dr.py set status=superseded --ids N-0002,N-0004             # 적용
tools/dr.py retag clustering text-clustering                      # 태그 일괄 변경
tools/dr.py mv N-0003 --slug new-name                             # 파일명 변경
tools/dr.py mv N-0003 --to T-0002                                 # 다른 작업으로 이동
```

일괄 수정은 `--dry-run`으로 먼저 확인하는 것을 권합니다. 수정 내역은 로그에 `admin` 이벤트로 남습니다.

### 점검

```bash
tools/dr.py check    # 누락 필드, 끊어진 연결, 미등록 파일, 색인 불일치 검사
tools/dr.py index    # 색인 재생성
```

## 주의할 점

- `INDEX.md`와 `index/` 아래 파일은 자동 생성물입니다. 직접 고치면 다음 색인 때 덮어써집니다.
- 문서 머리말(frontmatter)을 손으로 고쳤다면 `tools/dr.py index`를 실행해 색인을 맞춰 주세요.
- `log/events.tsv`는 기록 보존용이므로 행을 수정하거나 지우지 않습니다.
- 문서는 삭제하지 않고 상태를 `superseded`(대체됨)나 `dropped`(중단)로 바꿉니다.
- 다른 위치에 만들어진 Markdown 파일은 `tools/dr.py ingest note T-xxxx <파일>`로 등록해야 색인에 잡힙니다.

## 로그 읽는 법

`log/events.tsv`는 탭으로 구분된 6개 열입니다: 시각, 작업 ID, 이벤트, 행위자, 관련 문서, 내용.

| 이벤트 | 의미 |
|---|---|
| received | 의뢰 접수 |
| ask / answer | 에이전트의 질문 / 의뢰자의 답변 |
| plan | 하위 질문 확정 |
| dispatch / return | 서브에이전트에 위임 / 결과 회수 |
| add | 문서 생성 또는 등록 |
| deliver | 결과 납품 |
| revise | 납품 후 수정 |
| close / reopen | 작업 종료 / 재개 |
| admin | 일괄 수정, 이동 |
