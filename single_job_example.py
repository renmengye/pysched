"""
Example of using job scheduler of a single job. This is useful for evaluating
networks during training, where the evaluation only needs to happen every few
hours, and we can release the GPU resource while it is not running. The program
comes with two type of dispatcher, one on local, and the other on Slurm. The
Slurm job scheduler will launch the job remotely on one of the GPU servers to
be dynamically chosen at the the time when the job is scheduled. Dispatcher
returns a waitable object to wait until the job finishes.

Usage:
  dispatcher = dispatch_factory.create(num_gpu=1, num_cpu=2)
  job = dispatcher.dispatch(["list", "of", "command", "argument"])
  code = job.wait()
"""

#!/usr/bin/python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import time
import traceback

from slurm import SlurmCommandDispatcherFactory
from job import LocalCommandDispatcherFactory
import logger
log = logger.get()

LOCAL = True  # Whether use local dispatcher or slurm dispatcher.
MIN_INTERVAL = 5  # How much time in second to relaunch another job.
CMD = ["sleep", "10"]  # The command to be launched.

if LOCAL:
  dispatch_factory = LocalCommandDispatcherFactory()
else:
  dispatch_factory = SlurmCommandDispatcherFactory()

if FLAGS.id is None:
  raise Exception("You need to specify model ID.")
while True:
  try:
    start_time = time.time()
    dispatcher = dispatch_factory.create(num_gpu=1, num_cpu=2)
    job = dispatcher.dispatch(CMD)
    code = job.wait()
    if code != 0:
      log.error("Job failed")
  except Exception as e:
    log.error("An exception occurred.")
    log.error(e)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    log.error("*** print_tb:")
    traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
    log.error("*** print_exception:")
    traceback.print_exception(
        exc_type, exc_value, exc_traceback, limit=10, file=sys.stdout)
  end_time = time.time()
  elapsed = end_time - start_time
  # If the time elapsed is smaller than the minimum interval, keep waiting.
  if elapsed < MIN_INTERVAL:
    time.sleep(MIN_INTERVAL - elapsed)
