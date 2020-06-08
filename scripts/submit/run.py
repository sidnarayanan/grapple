#!/usr/bin/env python2.7

import time
from re import sub
import cPickle as pickle
from os import getenv, getuid, system, path, environ
import os
import logging
import shutil
import sys

try:
    import htcondor
    import classad
except ImportError:
    import imp
    local_path = ['/usr/lib64/python2.6/site-packages/']
    def _get_module(name):
        found = imp.find_module(name,local_path)
        return imp.load_module(name,*found)
    classad = _get_module('classad') 
    htcondor = _get_module('htcondor') 

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('submit')
logger.setLevel(logging.DEBUG)


#############################################################
# HTCondor interface for job submission and tracking
#############################################################

### global configuration ###
job_status = {
    1:'idle',
    2:'running',
    3:'removed',
    4:'completed',
    5:'held',
    6:'transferring output',
    7:'suspended',
}
job_status_rev = {v:k for k,v in job_status.iteritems()}


def environ_to_condor():
    s = ''
    for k,v in environ.iteritems():
        if 'SUBMIT' in k or 'USER' in k:
            s += '%s=%s '%(k,v)
    return s


def mkdir(d, clean=False):
    if clean and os.path.exists(d):
        shutil.rmtree(d)
    if not os.path.exists(d):
        os.makedirs(d)


class JobQueue(object):
    available_machines = {m+'.mit.edu':True for m in 
                    [#'t3desk014',
                     't3btch041',
                     't3btch042',
                     't3btch043'
                     ]
                    }
    def __init__(self, confs):
        self.validate_env()
        self.setup_schedd()
        self.prepare(confs)
        time.sleep(5)
        self.running_clusters = {}

    def prepare(self, confs):
        workdir = self.workdir 
        mkdir(workdir, clean=True)
        system('cp -r ${SUBMIT_CODEDIR} ${SUBMIT_WORKDIR}/grapple/')
        runner = '''
#!/bin/bash

conf_id=$1

echo CONF $conf_id

cd %s/grapple/ 
source ./setup.sh
python3 scripts/training/papu/train_pu.py -c ../${conf_id}/conf.yaml
#python3 scripts/training/test.py -c ../${conf_id}/conf.yaml
'''%workdir
        with open(workdir + '/runner.sh', 'w') as frunner:
            frunner.write(runner)

        self.args = []
        for i, conf in enumerate(confs):
            workdir_i = workdir + '/' + str(i)
            mkdir(workdir_i)
            system('cp %s %s/conf.yaml'%(conf, workdir_i))
            self.args.append(i)

    def _submit_task(self, job_id, target):
        cluster_ad = classad.ClassAd()
        proc_ad = classad.ClassAd()
        props = self.base_job_properties.copy()
        props['Environment'] = environ_to_condor()
        for k,v in props.items():
            cluster_ad[k] = v 
            #     classad.ExprTree(
        reqs = '%s && %s'%(
                    self.base_reqs,
                    'TARGET.Machine == "%s"'%target
                )
        cluster_ad['Requirements'] = classad.ExprTree(reqs)

        workdir_i = '%s/%i/'%(self.workdir, job_id)
        proc_ad['UserLog'] = '%s/log'%workdir_i 
        proc_ad['Out'] = '%s/out'%workdir_i
        proc_ad['Err'] = '%s/err'%workdir_i
        proc_ad['Arguments'] = str(job_id)

        cluster_id = self.schedd.submitMany(
                    cluster_ad, [(proc_ad, 1)]
                )
        logger.info('Submitted job %i to cluster %i on %s.'%(job_id, cluster_id, target))
        self.running_clusters[cluster_id] = (target, job_id) 
        self.available_machines[target] = False 

    def _poll(self, cluster_id):
        try:
            results = list(self.schedd.query('ClusterId =?= %i'%cluster_id))[0]
            status = job_status[results['JobStatus']]
        except IndexError:
            status = 'completed'
        target, job_id = self.running_clusters[cluster_id]
        if status in ('running', 'idle'):
            logger.info('Job %i still running on %s'%(job_id, target))
            return 
        # kill the job
        if status == 'held':
            ret = self.schedd.act(
                    htcondor.JobAction.Remove,
                    ['%s.%s'%(cluster_id, [results['ProcId']])]
                )
        del self.running_clusters[cluster_id]
        self.available_machines[target] = True 
        if status == 'held':
            logger.warning('Job %i has failed on %s. Re-queuing.'%(job_id, target))
            self.args.append(job_id)
        else:
            logger.info('Job %i has completed on %s.'%(job_id, target))

    def run(self):
        while len(self.args) > 0 or len(self.running_clusters) > 0:
            for cluster_id in list(self.running_clusters.keys()):
                self._poll(cluster_id)
            
            targets = [host for host,avail in self.available_machines.items() if avail]
            for target in targets:
                if not self.args:
                    break
                job_id = self.args.pop(0)
                self._submit_task(job_id, target)

            time.sleep(15)
        logger.info('All jobs are completed.')

    @staticmethod
    def validate_env():
        missing = [env for env in 
                    ['SUBMIT_WORKDIR', 'SUBMIT_CODEDIR']
                    if env not in environ]
        if missing:
            msg = 'Please define: ' + ', '.join(missing)
            logger.error(msg)
            raise RuntimeError(msg)

    def setup_schedd(self):
        if int(environ.get('SUBMIT_URGENT', 0)):
            acct_grp_t3 = 'group_t3mit.urgent'
        else:
            acct_grp_t3 = 'group_t3mit'
        self.base_reqs = 'UidDomain == "mit.edu" && Arch == "X86_64"'
        self.workdir = getenv('SUBMIT_WORKDIR')
        self.base_job_properties = {
            "Iwd" : self.workdir,
            "Cmd" : "%s/runner.sh"%self.workdir,
            "WhenToTransferOutput" : "ON_EXIT",
            "ShouldTransferFiles" : "YES",
            # "Requirements" :
            #     classad.ExprTree(
            #         ),
            # "REQUIRED_OS" : "rhel6",
            "AcctGroup" : acct_grp_t3,
            "AccountingGroup" : '%s.%s'%(acct_grp_t3, getenv('USER')),
            "OnExitHold" : 
                classad.ExprTree(
                    "( ExitBySignal == true ) || ( ExitCode != 0 )"),
            "In" : "/dev/null",
            "HAS_GPU": '1',
        }

        self.schedd_server = getenv('HOSTNAME')
        self.query_owner = getenv('USER')

        self.coll = htcondor.Collector()
        self.schedd = htcondor.Schedd(
                self.coll.locate(
                    htcondor.DaemonTypes.Schedd,
                    self.schedd_server
                )
            )
                                 
if __name__ == '__main__':
    jq = JobQueue(sys.argv[1:])
    jq.run()
