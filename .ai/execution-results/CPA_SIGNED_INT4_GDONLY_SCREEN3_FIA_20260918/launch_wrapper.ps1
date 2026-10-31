$repo = (Get-Location).Path
$py = 'C:\miniconda\envs\DLG\python.exe'
$task = 'CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918'
$result = Join-Path $repo ('.ai\execution-results\' + $task)
$wrapper = Join-Path $result 'run_pipeline.py'
$stdout = Join-Path $result 'wrapper.stdout.log'
$stderr = Join-Path $result 'wrapper.stderr.log'
if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force }
if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force }
$p = Start-Process -FilePath $py -ArgumentList @($wrapper) -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$wrapperPid = [int]$p.Id
$wrapperCreatedMs = ([DateTimeOffset]$p.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds()
Start-Sleep -Milliseconds 700
$callback = Join-Path $result 'task_local_completion_callback.py'
$cbout = Join-Path $result 'callback.stdout.log'
$cberr = Join-Path $result 'callback.stderr.log'
if (Test-Path -LiteralPath $cbout) { Remove-Item -LiteralPath $cbout -Force }
if (Test-Path -LiteralPath $cberr) { Remove-Item -LiteralPath $cberr -Force }
$cb = Start-Process -FilePath $py -ArgumentList @($callback, '--wait-pid', "$wrapperPid", '--expected-created-ms', "$wrapperCreatedMs", '--task-id', $task, '--stage', 'real_experiment', '--execution-record', ('.ai/execution-results/' + $task + '/stage_execution.json'), '--executor-thread', '01a0a799-f1ce-7b31-8c46-b4ddef40aeab') -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $cbout -RedirectStandardError $cberr -PassThru
$callbackPid = [int]$cb.Id
$callbackCreatedMs = ([DateTimeOffset]$cb.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds()
Start-Sleep -Seconds 2
$startup = Get-Content -LiteralPath (Join-Path $result 'startup_receipt.json') -Raw | ConvertFrom-Json
$completionStatus = Join-Path $repo ('.ai\relay-runs\completions\' + $task + '\real_experiment\status.json')
if (-not (Test-Path -LiteralPath $completionStatus)) { throw 'task-local callback did not write startup status' }
$env:CPA_WRAPPER_PID = "$wrapperPid"
$env:CPA_WRAPPER_CREATED_MS = "$wrapperCreatedMs"
$env:CPA_CALLBACK_PID = "$callbackPid"
$env:CPA_CALLBACK_CREATED_MS = "$callbackCreatedMs"
$env:CPA_STARTUP_STATUS = "$($startup.status)"
$env:CPA_STARTUP_STAGE = "$($startup.stage)"
$recordScript = @'
import json, os
from datetime import datetime, timezone
from pathlib import Path
root=Path.cwd(); task='CPA_SIGNED_INT4_GDONLY_SCREEN3_FIA_20260918'
result=root/'.ai/execution-results'/task; schedule=root/'.ai/relay-runs/schedules'/task
wrapper_pid=int(os.environ['CPA_WRAPPER_PID']); callback_pid=int(os.environ['CPA_CALLBACK_PID'])
launch={'wrapper_pid':wrapper_pid,'wrapper_created_ms':int(os.environ['CPA_WRAPPER_CREATED_MS']),'callback_pid':callback_pid,'callback_created_ms':int(os.environ['CPA_CALLBACK_CREATED_MS']),'startup_status':os.environ['CPA_STARTUP_STATUS'],'startup_stage':os.environ['CPA_STARTUP_STAGE'],'launched_at':datetime.now(timezone.utc).isoformat(),'startup_receipt':str(result/'startup_receipt.json'),'stage_execution':str(result/'stage_execution.json'),'callback_status':str(root/'.ai/relay-runs/completions'/task/'real_experiment/status.json')}
for name in ('claim.json','pending_schedule.json','activation_receipt.json'):
 p=result/name; x=json.loads(p.read_text(encoding='utf-8')); x.update({'status':'RUNNING','research_launched':True,'research_process_started':True,'launch_receipt':launch,'callback_registered':True,'updated_at':launch['launched_at']}); p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
p=schedule/'status.json'; x=json.loads(p.read_text(encoding='utf-8')); x.update({'status':'RUNNING','launch':'STARTED','research_process_started':True,'wrapper_pid':wrapper_pid,'wrapper_created_ms':launch['wrapper_created_ms'],'callback_pid':callback_pid,'callback_created_ms':launch['callback_created_ms'],'callback_registered':True,'startup_receipt':launch['startup_receipt'],'callback_status':launch['callback_status'],'updated_at':launch['launched_at']}); p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(result/'launch_receipt.json').write_text(json.dumps(launch,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'wrapper_pid':wrapper_pid,'wrapper_created_ms':launch['wrapper_created_ms'],'callback_pid':callback_pid,'callback_created_ms':launch['callback_created_ms'],'startup_status':launch['startup_status'],'startup_stage':launch['startup_stage'],'callback_status_exists':True},ensure_ascii=False))
'@
$recordScript | & $py -
