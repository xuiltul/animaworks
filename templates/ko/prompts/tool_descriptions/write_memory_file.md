자신의 기억 디렉터리 내 파일에 기록하거나 내용을 덧붙인다. 기록해야 할 때: 문제 해결 → knowledge/에 원인과 해결책 기록, 올바른 설정값 → knowledge/、, 절차 확립 → procedures/、, 새로운 기능 → create_skill로 skills/{name}/SKILL.md 생성, 전송·게시 전 확인 규칙 → knowledge/action-rule-*.md에 [ACTION-RULE]과 trigger_tools 기록, HEARTBEAT.md / cron.md 업데이트. mode='overwrite'로 전체를 대체하고, mode='append'로 끝에 덧붙인다. 중요한 발견은 consolidation을 기다리지 말고 즉시 기록한다.
기록 위치에 관한 주의사항:
- 근거를 확인하지 않은 “X는 사용할 수 없다/해서는 안 된다” 등의 부정적인 주장을 knowledge/・skills/로 기록하지 않는다.
- 환경에 따라 발생하는 일시적인 실패나 해결되지 않은 시행착오는 knowledge/가 아니라 episodes/에 남긴다.
- PR 번호·SHA·여러 날짜와 관련된 사례 기록은 episodes/에 작성한다.
