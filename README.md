# 스마트 인솔 분석 대시보드 (제자리 3D)

이 앱은 스마트 인솔의 **쿼터니언(자세)**, **자이로(회전)**, **압력(ps_0~ps_8)** 시계열을 이용해,
**전진(절대 위치) 복원은 제거**하고 앱 안에서 바로 분석 장면을 렌더링합니다.

- 전진/정지 판단은 **v_fwd(전진 속도 지표)** 및 **stance(접지)** 기반으로 표시합니다.
- 압력 히트맵은 센서 물리 배치 정보가 없기 때문에 **ps_0~ps_8을 3×3 도식 배치**로 표시합니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 입력

- **통합 JSON 1개**: `leftInsole.data`, `rightInsole.data` 또는 자동 탐지
- **좌/우 JSON 2개**
- **이미 변환된 CSV 2개**: `ts, acc_x/y/z, gyro_x/y/z, quat_0~3, ps_0~8` 포함

## 출력

- 앱 내 실시간(프레임 슬라이더) 분석 화면
- 다운로드(맨 아래): `left.csv`, `right.csv`, `summary.csv`, `analysis.mp4`, `results.zip`

## 참고

- mp4는 **다운로드용 기록 결과**입니다(앱은 mp4 재생이 아니라, 분석 장면을 직접 렌더링).
- 긴 파일/동시 사용자가 많아지면 Streamlit Community Cloud보다 VM/Docker 배포가 안정적일 수 있습니다.
