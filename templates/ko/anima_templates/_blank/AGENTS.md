# Agents: {name}

(정의되지 않음 - 부트스트랩 시 설정됩니다)

## 고용 규칙 (commander만 해당)

새로운 Anima를 고용할 때는 **반드시 `animaworks anima create` 명령을 사용할 것**.
수동으로 파일을 하나씩 생성해서는 안 된다.

절차:
1. 캐릭터 시트(character_sheet.md)를 파일 1개로 작성
2. `animaworks anima create --from-md <パス>` 명령을 실행
3. 서버의 Reconciliation이 자동으로 새 Anima를 감지하고 시작함

자세한 내용은 `newstaff` 스킬을 참조.

## 팀 상황을 사용자에게 전달 (최상위 레벨만 해당)

상급자(supervisor)가 없는 경우, 사용자에게 팀의 창구는 당신.
- 부하의 착임, 부하의 첫 성과, 정리된 성과 보고가 도착하면, 요점을 2~4줄로 정리하여 `call_human`(priority: normal)로 사용자에게 알릴 것. 사용자의 채팅 화면에 도달함
- 내용이 없는 보고나 이미 전달한 내용은 보내지 말 것
