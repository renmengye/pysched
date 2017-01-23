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

  def __init__(self, executable, config, environ):
    self.executable = executable
    self.config = config
    self.environ = environ


class JobResults(object):
  """Job results object."""

  def __init__(self, config, environ, success):
    self.config = config
    self.environ = environ
    self.success = success


class TimeSeriesResults(object):
  """Time-series results object."""

  def __init__(self, filename, data):
    self.filename = filename
    self.data = data


class TimeSeriesEntry(object):
  """Time-series entry."""

  def __init__(self, step, time, value):
    self.step = step
    self.time = time
    self.value = value


class TimeSeriesParser(object):
  """Time-series data parser."""

  def parse(self, file, label):
    with open(file, "r") as f:
      lines = [ll.strip("\n") for ll in f.readlines()]
    col = lines[0].split(",")
    col_dict = dict((col[ii], ii) for ii in range(len(col)))
    step_idx = col_dict["step"]
    time_idx = col_dict["time"]
    value_idx = col_dict[label]
    data = []
    for ll in lines[1:]:
      parts = ll.split(",")
      step = int(parts[step_idx])
      time = datetime.datetime.strptime(parts[time_idx], "%Y-%m-%dT%H:%M:%S.%f")
      val = float(parts[value_idx])
      data.append(TimeSeriesEntry(step, time, val))
    return TimeSeriesResults(file, data)


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


class LocalCommandDispatcher(CommandDispatcher):

  def __init__(self, gpu_id=0):
    self._gpu_id = gpu_id

  @property
  def gpu_id(self):
    return self._gpu_id

  def get_pipes(self, stdout_file):
    """Launch a new job."""
    stdout = open(stdout_file, "w")
    return stdout

  def finalize(self):
    self.stdout.close()

  def get_env(self):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
    return env

  def get_exec_command(self, args, stdout_file=None):
    return args


class JobRunner(threading.Thread):

  def __init__(self,
               request,
               dispatcher,
               result_queue=None,
               resource_queue=None):
    """
    Args:
        config: Job model config object.
        environ: Job environment object.
        machine: Machine name.
    """
    super(JobRunner, self).__init__()
    self.daemon = True
    self._request = request
    self._dispatcher = dispatcher
    self._results = None
    self._result_queue = result_queue
    self._resource_queue = resource_queue
    self.config_file = "../tmp/{}.conf.json".format(self.environ.exp_id)
    self.env_file = "../tmp/{}.env.json".format(self.environ.exp_id)
    self.stdout_file = "../tmp/{}.log".format(self.environ.exp_id)
    pass

  @property
  def request(self):
    return self._request

  @property
  def dispatcher(self):
    return self._dispatcher

  @property
  def config(self):
    return self._request.config

  @property
  def environ(self):
    return self._request.environ

  @property
  def result_queue(self):
    return self._result_queue

  @property
  def resource_queue(self):
    return self._resource_queue

  @property
  def results(self):
    return self._results

  def launch(self):
    """Launch a new job."""
    if not os.path.exists("../tmp"):
      os.makedirs("../tmp")
    with open(self.config_file, "w") as f:
      f.write(self.config.to_json())

    with open(self.env_file, "w") as f:
      f.write(self.environ.to_json())

    log.info("Job \"{}\" launched".format(self.environ.exp_id), verbose=2)
    return self.dispatcher.dispatch(
        [
            self.request.executable, "--config", self.config_file, "--env",
            self.env_file
        ],
        stdout_file=self.stdout_file)

  def finalize(self):
    pass

  def run(self):
    start_time = time.time()
    log.info("Job \"{}\" ({}) started.".format(self.environ.exp_id,
                                               self.environ.description))
    job = self.launch()
    code = job.wait()
    success = code == 0
    results = JobResults(self.config, self.environ, success)
    if not success:
      log.error("Job \"{}\" ({}) failed".format(self.environ.exp_id,
                                                self.environ.description))
    if self.result_queue is not None:
      self.result_queue.put(results)
    self._results = results
    self.finalize()
    if self.resource_queue is not None:
      self.resource_queue.get()
    end_time = time.time()
    log.info("Job \"{}\" ({}) finished in {}.".format(
        self.environ.exp_id, self.environ.description,
        TimeElapsedFormatter().format(int(end_time - start_time))))
    pass


class LocalJobRunnerFactory(object):
  """Creates a new LocalJobRunner."""

  def create(self, request, result_queue, resource_queue):
    return JobRunner(request,
                     LocalCommandDispatcher(request.environ.gpu), result_queue,
                     resource_queue)


class LocalCommandDispatcherFactory(object):
  """Creates a new LocalJobRunner."""

  def create(self, num_gpu=0, num_cpu=2):
    return LocalCommandDispatcher(num_gpu)


class TimeElapsedFormatter(object):

  def format(self, num_sec):
    num_day = 0
    num_hr = 0
    num_min = 0
    ss = []
    if num_sec >= 60:
      num_min = num_sec // 60
      num_sec = num_sec % 60
    if num_min >= 60:
      num_hr = num_min // 60
      num_min = num_min % 60
    if num_hr >= 24:
      num_day = num_hr // 24
      num_hr = num_hr % 24
    if num_day > 0:
      ss.append("{:d}d".format(num_day))
    if num_hr > 0 or num_day > 0:
      ss.append("{:d}h".format(num_hr))
    if num_min > 0 or num_hr > 0 or num_day > 0:
      ss.append("{:d}m".format(num_min))
    ss.append("{:d}s".format(num_sec))
    return " ".join(ss)


class Pipeline(threading.Thread):

  def __init__(self):
    super(Pipeline, self).__init__()
    self._stages = []
    self._source = None
    self._is_started = False
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
    self.job_table[job_request.environ.exp_id] = job
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

  def wait_for_timeout(self, timeout=400000):
    # Temporary hack.
    # Just wait for a timeout.
    if self.is_started:
      time.sleep(timeout)
      self.finalize_requests()
      self.join()
    pass

  def run(self):
    self._is_started = True
    start_time = time.time()
    for ss in self.stages:
      ss.start()
    try:
      while True:
        item = self.stages[-1].output_queue.get()
        if item is None:
          break
        else:
          job = self.job_table[item.environ.exp_id]
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
    end_time = time.time()
    log.info("Pipeline finished in {}.".format(TimeElapsedFormatter().format(
        int(end_time - start_time))))
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
    start_time = time.time()
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
    end_time = time.time()
    log.info("Pipeline stage \"{}\" finished in {}.".format(
        self.name, TimeElapsedFormatter().format(int(end_time - start_time))))

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
