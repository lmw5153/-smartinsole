# 스마트 인솔 Streamlit 앱

이 앱은 스마트 인솔의 left/right JSON 또는 CSV를 받아 다음을 수행합니다.

- JSON -> CSV 변환
- 기본 통계량 계산
- 쿼터니언 기반 3D 분석 영상 생성
- 압력 히트맵 및 총압 추세 표시
- CSV, summary, mp4, ZIP 다운로드

## 실행 방법

```bash
pip install -r requirements_streamlit_smart_insole.txt
streamlit run streamlit_smart_insole_app.py
```

## 파일 구성

- `streamlit_smart_insole_app.py`: Streamlit UI
- `smart_insole_core.py`: 파싱, 분석, 영상 렌더링, 결과 패키징
- `requirements_streamlit_smart_insole.txt`: 의존성

## 배포

### Streamlit Community Cloud
1. 위 3개 파일을 GitHub 저장소에 올립니다.
2. Community Cloud에서 저장소, 브랜치, 앱 파일(`streamlit_smart_insole_app.py`)을 선택합니다.
3. Deploy 하면 됩니다.

### 권장 사항
- 업로드 파일이 큰 경우 앱 메모리 사용량이 커질 수 있습니다.
- 동시 접속이 많거나 긴 영상을 자주 생성하면 Render, Docker, VM 배포가 더 안정적일 수 있습니다.

## 입력 형식

### 통합 JSON 예시 구조
- `leftInsole.data`
- `rightInsole.data`

각 data 항목은 아래 구조를 포함하면 됩니다.
- `ts`
- `semiRawImuPs.acc.{x,y,z}`
- `semiRawImuPs.gyro.{x,y,z}`
- `semiRawImuPs.ps`
- `semiRawImuPs.quat`

## 참고
압력 히트맵은 현재 `ps_0~ps_8`를 3x3 도식 배치로 보여줍니다. 실제 센서 물리 배치가 있으면 `PRESSURE_LAYOUT`를 수정해서 더 현실적으로 바꿀 수 있습니다.
