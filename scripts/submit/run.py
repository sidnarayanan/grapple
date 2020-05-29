#!/usr/bin/env python2.7

import sys 
from job_management import JobQueue, logger

if __name__ == '__main__':
    jq = JobQueue(sys.argv[1:])
    jq.run()
