<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/integrations/zoom.md -->
<!-- i18n: source-sha256=1409948a4743338cc6c2e8ebdb8d5970b6458bf7229ed6f9e1459b7bf3d06632 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 581e20f1

# Zoom RTMS 연동

Zoom RTMS는 회의의 트랜스크립트를 WebSocket으로 수신하고, 설정된 Anima의 inbox로 청크 단위로 전달한다. AnimaWorks는 이 경로로 회의에 참여하여 발언하는 것이 아니라, 수신한 받아쓰기 내용을 처리한다.

## 전제 조건

- Zoom 측에서 RTMS를 지원하는 App과 이벤트 구독을 설정한다.
- Zoom에서 도착하는 webhook용으로 서버가 외부에서 접근 가능한 HTTPS endpoint를 준비한다.
- `ZOOM_SECRET_TOKEN`을 서명 검증과 URL validation에 사용하고, RTMS 연결용 `ZOOM_CLIENT_ID`과 `ZOOM_CLIENT_SECRET`를 자격 증명으로 등록한다.
- AnimaWorks의 설정과 자격 증명은 값을 공개하지 않고 안전한 보관 장소에 등록한다.

Zoom의 계약 조건, RTMS 이용 가능 여부, 받아쓰기 이용 조건은 계정과 Zoom 측 설정에서 확인한다. 참가자에 대한 알림이나 동의 요건도 따른다.

## AnimaWorks 설정

`config.json`의 `external_messaging.zoom`에서 `enabled`을 활성화한다. `meeting_mapping`는 회의 ID에서 수신 대상 Anima로의 대응표이며, 대응이 없는 회의에는 `default_anima`가 사용된다. 둘 다 할당되지 않으면 수신 대상이 없다. 청크 간격과 문자 수 상한은 용도에 따라 조정한다. 항목과 기본값은 [설정 참조](../reference/config.md)를 참조한다.

```json
{
  "external_messaging": {
    "zoom": {
      "enabled": true,
      "default_anima": "sample-anima",
      "meeting_mapping": {
        "1234567890": "sample-anima"
      },
      "chunk_interval_seconds": 300,
      "chunk_max_chars": 4000
    }
  }
}
```

자격 증명과 설정을 저장한 후 서버를 시작 또는 재시작한다. Zoom의 webhook URL은 `/api/webhooks/zoom`을 등록한다. 서명 검증과 webhook 수신, RTMS stream의 연결 로그를 확인한 후 테스트 회의를 진행한다.

## 운영 확인

`/api/system/health`의 Zoom gateway 정보와 서버 로그로 연결·최종 수신을 확인한다. 테스트 회의에서는 받아쓰기가 생성되는지, 수신 대상의 inbox에 transcript 메시지가 도착하는지 확인한다. 수집과 Anima의 응답 처리는 별도 단계이므로, inbox에 도착한 후 응답이 느리면 Anima 측의 실행 상태도 확인한다.

수신하지 않는 경우, Webhook가 HTTPS로 도달하는지, URL validation이 성공하는지, 서명용 Secret Token과 Client ID / Secret이 일치하는지, Zoom 측에서 RTMS와 transcript를 사용할 수 있는지 확인한다. 회의 ID가 `meeting_mapping`과 일치하는지, `default_anima`가 지정되었는지도 확인한다.
