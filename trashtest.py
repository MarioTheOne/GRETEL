from flufl.lock import Lock
from datetime import timedelta
import time


lock = Lock('idunno.lck',lifetime=None)

with lock:
  print("I am IN!")
  
  for i in range(10):
    print(lock.is_locked)
    print("I am consuming "+str(i))
    time.sleep(1)



