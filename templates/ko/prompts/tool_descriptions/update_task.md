태스크의 상태를 업데이트한다. 완료 시 status='done', 철회 시 status='cancelled'로 설정. status='in_progress'는 실행 중인 TaskExec가 경과를 summary에 쓸 때 사용한다. 계속하려면 같은 task_id를 `submit_tasks`로 재제출한다.
