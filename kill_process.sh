ps aux | grep train.py | grep -v grep | awk '{print $2}'
ps aux | grep test.py | grep -v grep | awk '{print $2}'

kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')
kill $(ps aux | grep test.py | grep -v grep | awk '{print $2}')
kill $(ps aux | grep pytorch180a0 | grep -v grep | awk '{print $2}')
