"""
Author: Mengye Ren (mren@cs.toronto.edu)

Logics and interfaces to dispatch jobs locally.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

from job import JobRunner, CommandDispatcher
import logger

log = logger.get()


class LocalCommandDispatcher(CommandDispatcher):

  def __init__(self, gpu_id=0):
    self._gpu_id = gpu_id

  @property
  def gpu_id(self):
    return self._gpu_id

  def get_pipes(self, stdout_file):
    """Launch a new job."""
    if stdout_file is None:
      return None
    else:
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


class LocalJobRunnerFactory(object):
  """Creates a new LocalJobRunner."""

  def create(self, request, result_queue, resource_queue, stdout_file=None):
    return JobRunner(
        request,
        LocalCommandDispatcher(),
        result_queue=result_queue,
        resource_queue=resource_queue,
        stdout_file=stdout_file)


class LocalCommandDispatcherFactory(object):
  """Creates a new LocalJobRunner."""

  def create(self, num_gpu=0, num_cpu=2):
    return LocalCommandDispatcher(num_gpu)
