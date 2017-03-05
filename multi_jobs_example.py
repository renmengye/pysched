#!/usr/bin/env python
"""
Author: Mengye Ren (mren@cs.toronto.edu)

Example of using job scheduler of multiple jobs. This is useful for a
hyper-parameter search with limited resources, and each job maybe of different
length, so scheduling is crucial to keep the maximum utility of the resources.
Callbacks can be used to intialize new jobs whose input is dependent on the
output of the previous programs (data dependency), for example in choosing the
best hyperparameters.

Usage:
  job_request = [JobRequest({cmd line exec for param}) for param in params]
  job_pool = pipeline.add_job_pool(job_request, callback=None)
  job_pool.wait()
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from job import JobRequest, JobScheduler, Pipeline
from slurm import SlurmJobRunnerFactory
from local import LocalJobRunnerFactory


def init_pipeline(max_num_jobs, scheduler):
  if scheduler == "slurm":
    sched = JobScheduler(
        "scheduler", SlurmJobRunnerFactory(), max_num_jobs=max_num_jobs)
  elif scheduler == "local":
    sched = JobScheduler(
        "scheduler", LocalJobRunnerFactory(), max_num_jobs=max_num_jobs)
  p = Pipeline().add_stage(sched)
  p.start()
  return p


def run_multi_jobs(pipeline, callback=None):
  param_list = range(10, 15)
  job_list = []
  for param in param_list:
    # A dummy job is to sleep from 1 to 9 seconds
    job = pipeline.add_job(
        JobRequest(
            # ["sleep", str(param)],
            ["echo", "Hello world {}".format(param)],
            num_gpu=0,
            num_cpu=1,
            job_id=param,
            stdout_file="job_{}.txt".format(param)))
    job_list.append(job)
  return pipeline.add_job_pool(job_list, callback=callback)


def main():
  pipeline = init_pipeline(max_num_jobs=3, scheduler="local")
  run_multi_jobs(pipeline, callback=lambda results: pipeline.finalize())
  pipeline.wait()


if __name__ == "__main__":
  main()
