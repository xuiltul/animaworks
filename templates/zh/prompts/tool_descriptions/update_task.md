声明任务结果。验证后为 status='done'，有等待理由时为 'pending'，中止时为 'cancelled'。in_progress 由主机管理，不自行设置。要刻意重新开始已中断的未完成任务，向 submit_tasks 传入现有 task_id 和 resume=true。保留已保存的输入和历史。
