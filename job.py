"""
Author: Mengye Ren (mren@cs.toronto.edu)

Basic utilities of job scheduling.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import numpy as np
import os
import subprocess
import sys
import threading
import time
import traceback
import Queue

import logger
log = logger.get()


class JobRequest(object):
  """Job request object."""

  def __init__(self, cmd_args, num_gpu=1, num_cpu=2, job_id=None):
    self.cmd_args = cmd_args
    self.num_gpu = num_gpu
    self.num_cpu = num_cpu
    self.job_id = job_id


class JobResults(object):
  """Job results object."""

  def __init__(self, success, job_id=None):
    self.job_id = job_id
    self.success = success


class TimeSeriesResults(object):
  """Time-series results object."""

  def __init__(self, filename, data):
    self.filename = filename
    self.data = data


class CommandDispatcher(object):

  def get_pipes(self, stdout_file):
    return None

  def get_env(self):
    return os.environ

  def finalize(self):
    pass

  def get_exec_command(self, args, stdoutfile=None):
    raise Exception("Not implemented.")

  def dispatch(self, args, stdout_file=None):
    if stdout_file is not None:
      pipe = self.get_pipes(stdout_file)
    else:
      pipe = None
    args = self.get_exec_command(args, stdout_file=stdout_file)
    env = self.get_env()
    job = subprocess.Popen(args, stdout=pipe, stderr=pipe, env=env)
    return job


class JobRunner(threading.Thread):

  def __init__(self,
               request,
               dispatcher,
               stdout_file=None,
               result_queue=None,
               resource_queue=None):
    """
    Args:
    """
    super(JobRunner, self).__init__()
    self.daemon = True
    self._request = request
    self._dispatcher = dispatcher
    self._results = None
    self._result_queue = result_queue
    self._resource_queue = resource_queue
    self._stdout_file = stdout_file
    pass

  @property
  def request(self):
    return self._request

  @property
  def dispatcher(self):
    return self._dispatcher

  @property
  def result_queue(self):
    return self._result_queue

  @property
  def resource_queue(self):
    return self._resource_queue

  @property
  def results(self):
    return self._results

  @property
  def stdout_file(self):
    return self._stdout_file

  def launch(self):
    """Launch a new job."""
    log.info("Job \"{}\" launched".format(self.request.job_id))
    return self.dispatcher.dispatch(
        self.request.cmd_args, stdout_file=self.stdout_file)

  def finalize(self):
    pass

  def run(self):
    job = self.launch()
    code = job.wait()
    success = code == 0
    results = JobResults(success, job_id=self.request.job_id)
    if not success:
      log.error("Job \"{}\" failed".format(self.request.job_id))
    else:
      log.info("Job \"{}\" finished".format(self.request.job_id), verbose=2)
    if self.result_queue is not None:
      self.result_queue.put(results)
    self._results = results
    self.finalize()
    if self.resource_queue is not None:
      self.resource_queue.get()
    pass


class Pipeline(threading.Thread):

  def __init__(self):
    super(Pipeline, self).__init__()
    self._stages = []
    self._source = None
    self._is_started = False
    self._is_finished = False
    self.daemon = True
    self._results = []
    self._job_table = {}
    self._job_pools = []
    self.add_source(JobSource("source"))

  @property
  def stages(self):
    return self._stages

  @property
  def source(self):
    return self._source

  @property
  def job_pools(self):
    return self._job_pools

  @property
  def job_table(self):
    return self._job_table

  def add_source(self, source):
    if len(self.stages) > 0:
      raise Exception("Source must be the first")
    self.stages.append(source)
    self._source = source
    return self

  def finalize_requests(self):
    self.source.finalize()

  def add_stage(self, stage):
    """Chain the current stage with the next stage."""
    if len(self.stages) == 0:
      raise Exception("Must add source first")
    last_stage = self.stages[-1]
    stage.set_input_queue(last_stage.output_queue)
    self.stages.append(stage)
    return self

  def add_job(self, job_request, callback=None):
    self.source.add(job_request)
    job = Job(job_request, callback=callback)
    self.job_table[job_request.job_id] = job
    job.start()
    return job

  def add_job_pool(self, jobs, callback=None):
    jp = JobPool(jobs, callback=callback)
    self.job_pools.append(jp)
    jp.start()
    return jp

  @property
  def results(self):
    return self._results

  @property
  def is_started(self):
    return self._is_started

  @property
  def is_finished(self):
    return self._is_finished

  def wait(self):
    while not self.is_finished:
      time.sleep(10)  # Check back every 10 seconds.

  def finalize(self):
    self._is_finished = True
    self.finalize_requests()
    self.join()

  def run(self):
    self._is_started = True
    for ss in self.stages:
      ss.start()
    try:
      while True:
        item = self.stages[-1].output_queue.get()
        if item is None:
          break
        else:
          job = self.job_table[item.job_id]
          job.stop(item)
    except Exception as e:
      log.error("An exception occurred.")
      log.error(e)
      exc_type, exc_value, exc_traceback = sys.exc_info()
      log.error("*** print_tb:")
      traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
      log.error("*** print_exception:")
      traceback.print_exception(
          exc_type, exc_value, exc_traceback, limit=10, file=sys.stdout)
    for jj in self.job_table.values():
      jj.join()
      self.results.append(jj.result)
    for ss in self.stages:
      ss.join()
    for jp in self.job_pools:
      jp.join()
    pass


class PipelineStage(threading.Thread):

  def __init__(self, name):
    super(PipelineStage, self).__init__()
    self._output_queue = Queue.Queue()
    self._input_queue = None
    self._name = name
    self.daemon = True
    pass

  @property
  def input_queue(self):
    return self._input_queue

  @property
  def output_queue(self):
    return self._output_queue

  @property
  def name(self):
    return self._name

  def set_input_queue(self, iq):
    self._input_queue = iq

  def finalize(self):
    self.output_queue.put(None)

  def run(self):
    try:
      while True:
        inp = self.input_queue.get()
        self.input_queue.task_done()
        if inp is not None:
          self.do(inp)
        else:
          break
    except Exception as e:
      log.error("An exception occurred.")
      log.error(e)
      exc_type, exc_value, exc_traceback = sys.exc_info()
      log.error("*** print_tb:")
      traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
      log.error("*** print_exception:")
      traceback.print_exception(
          exc_type, exc_value, exc_traceback, limit=10, file=sys.stdout)
    self.finalize()

  def do(self, inp):
    raise Exception("Not implemented")


class Job(threading.Thread):

  def __init__(self, request, callback=None):
    super(Job, self).__init__()
    self.daemon = True
    self._result = None
    self._request = request
    self._stop = threading.Event()
    self._callback = callback

  @property
  def result(self):
    return self._result

  @property
  def request(self):
    return self._request

  @property
  def callback(self):
    return self._callback

  def stop(self, result):
    self._result = result
    self._stop.set()

  def stopped(self):
    return self._stop.isSet()

  def set_result(self, val):
    self._result = val

  def run(self):
    while not self.stopped():
      time.sleep(1)
    if self.callback is not None:
      self.callback(self.result)


class JobPool(threading.Thread):

  def __init__(self, jobs, callback=None):
    super(JobPool, self).__init__()
    self._jobs = jobs
    self._callback = callback
    self.daemon = True

  @property
  def jobs(self):
    return self._jobs

  @property
  def callback(self):
    return self._callback

  def run(self):
    results = []
    for jj in self.jobs:
      jj.join()
      results.append(jj.result)
    if self.callback is not None:
      self.callback(results)


class JobSource(PipelineStage):

  def __init__(self, name):
    super(JobSource, self).__init__(name)

  def add(self, request):
    self.output_queue.put(request)

  def run(self):
    # self.finalize()
    pass


class JobScheduler(PipelineStage):

  def __init__(self, name, job_runner_factory, max_num_jobs=4):
    super(JobScheduler, self).__init__(name)
    self._resource_queue = Queue.Queue(maxsize=max_num_jobs)
    self._factory = job_runner_factory
    self._runners = []
    pass

  @property
  def factory(self):
    return self._factory

  @property
  def resource_queue(self):
    return self._resource_queue

  @property
  def runners(self):
    return self._runners

  def do(self, inp):
    self.resource_queue.put(1)
    runner = self.factory.create(
        inp, result_queue=self.output_queue, resource_queue=self.resource_queue)
    runner.start()
    self.runners.append(runner)

  def finalize(self):
    for rr in self.runners:
      rr.join()
    super(JobScheduler, self).finalize()


_CONTEXT = None


class PipelineContext(object):

  def __init__(self,
               pipeline,
               machine=None,
               debug=False,
               report_file=None,
               record_file=None):
    self._pipeline = pipeline
    self._machine = machine
    self._debug = debug
    self._report_file = report_file
    self._record_file = record_file
    self._previous = None

  @property
  def machine(self):
    return self._machine

  @property
  def pipeline(self):
    return self._pipeline

  @property
  def debug(self):
    return self._debug

  @property
  def report_file(self):
    return self._report_file

  @property
  def record_file(self):
    return self._record_file

  def __enter__(self):
    global _CONTEXT
    self._previous = _CONTEXT
    _CONTEXT = self

  def __exit__(self, type, value, traceback):
    global _CONTEXT
    _CONTEXT = self._previous


def get_context():
  global _CONTEXT
  if _CONTEXT is None:
    raise Exception("Pipeline context not initialized yet.")
  return _CONTEXT
