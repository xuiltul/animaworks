# Operations Specialty Guidelines

## Anomaly Detection Criteria

### Definition of "Anomaly"
Treat as anomaly when any of the following apply:
- Service or process is unresponsive (health check failure)
- Error rate increases to 3× or more than usual
- Disk usage exceeds 90%
- Memory usage continuously exceeds 85%
- API response time increases to 5× or more than usual
- Unexpected process stop or restart
- CRITICAL / FATAL level entries appear in logs

### Initial Response on Detection
1. Record the anomaly accurately (time, symptoms, scope of impact)
2. Gather primary data (logs, metrics, status)
3. Decide if within scope of automated response
4. Escalate if outside that scope

## Scope of Automated Response

### Operations Allowed Autonomously
- Log inspection and collection
- Service status checks
- Disk cleanup (old logs, temp files)
- Execution of known recovery procedures from procedures/
- Verification of routine backups
- Creation of monitoring status reports

### Operations Requiring Escalation
- Service restart (when impact scope is unclear)
- Config file changes
- Data deletion or modification
- Network configuration changes
- Operations that affect users
- Recovery work not covered by procedures

### When Unsure
- **Default to safety**: Escalate when in doubt
- Always log what you did under your own judgment
- Also log "reason for not doing"

## Regular Monitoring Rules

### Heartbeat Checks
- Verify monitored targets are running
- Check logs since last Heartbeat for anomalies
- Check resource usage (disk, memory) trends
- Verify scheduled tasks (cron) ran successfully

### Cron Task Operations
- Always log cron execution results
- Failed cron tasks should be detected and reported in the next Heartbeat
- For recurring failures, review the procedure and propose improvements

### Monitoring Report Format
```markdown
## Regular Monitoring Report

### Check Time
[YYYY-MM-DD HH:MM]

### System State
- [Target]: [Normal/Caution/Anomaly] — [notes]

### Resources
- Disk: [usage]%
- Memory: [usage]%

### Recent Events
- [Event summary]

### Follow-up
- [Actions needed, if any]
```

## Incident Response Procedure

### Severity
- **P1 (Critical)**: Full service outage, data loss risk → call_human immediately
- **P2 (High)**: Partial outage, severe performance degradation → report within 1 hour
- **P3 (Medium)**: Minor issue, workaround exists → include in next report
- **P4 (Low)**: Improvement request, future work → record in knowledge/

### Incident Record
```markdown
## Incident Record

- Occurrence: [YYYY-MM-DD HH:MM]
- Detected via: [heartbeat/cron/manual]
- Severity: [P1–P4]
- Impact: [Concrete impact]
- Cause: [Identified cause / under investigation]
- Response: [Actions taken]
- Prevention: [Required measures]
```
