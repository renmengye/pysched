"""
Author: Mengye Ren (mren@cs.toronto.edu)

Logics and interfaces to dispatch jobs on slurm.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import re
import socket
import subprocess

from job import JobRunner, CommandDispatcher
import logger

log = logger.get()


class SlurmCommandDispatcher(CommandDispatcher):
  """Launching remote jobs through slurm scheduler."""

  def __init__(self, num_gpu=1, num_cpu=2, machine=None):
    super(SlurmCommandDispatcher, self).__init__()
    self._num_gpu = num_gpu
    self._machine = machine
    self._num_cpu = num_cpu

  @property
  def num_gpu(self):
    return self._num_gpu

  @property
  def machine(self):
    return self._machine

  @property
  def num_cpu(self):
    return self._num_cpu

  def get_env(self):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/pkgs/cuda-8.0/lib64:/pkgs/cudnn-7.5:" + \
        env["LD_LIBRARY_PATH"]
    env["PYTHONPATH"] = "/pkgs/tensorflow-gpu-cuda8-11Oct2016:" + \
        env["PYTHONPATH"]
    env["CUDA_HOME"] = "/pkgs/cuda-8.0"
    return env

  def get_exec_command(self, args, stdout_file=None):
    if self.machine is None:
      machinestr = []
    else:
      machinestr = ["-w", self.machine]
    if stdout_file is None:
      stdoutstr = []
    else:
      stdoutstr = ["-o", stdout_file]
    cmd_args = ["srun", "--gres=gpu:{}".format(self.num_gpu)] + machinestr + [
        "-c", str(self.num_cpu), "-l"
    ] + stdoutstr + ["-p", "gpuc"] + args
    return cmd_args


class MachineSpec(object):

  def __init__(self, name, priority, num_gpu):
    self.name = name
    self.priority = priority
    self.num_gpu = num_gpu


class SlurmCommandDispatcherFactory(object):
  """Creates a new SlurmJobRunner."""

  def __init__(self, slurm_config_file=None):
    """
    Args:
      priority_config: A dictionary specifies the priority and number of GPU of
      each machine.
    """
    if slurm_config_file is None:
      slurm_config_file = "slurm_config.json"
    self.slurm_config = json.load(open(priority_config_file, "r"))

  def select_machine(self):
    machine_state_dict = {}
    slurm_info = subprocess.check_output(["sinfo"])
    lines = slurm_info.split("\n")[1:]
    for ll in lines:
      tokens = ll.split(" ")
      tokens = filter(lambda t: t != "", tokens)
      if len(tokens) == 0:
        continue
      state = tokens[4]
      if state not in machine_state_dict:
        machine_state_dict[state] = []
      machines = tokens[5]
      machine_list = machine_state_dict[state]
      rr = "(([\w]+)(\[(\d+)((,\d+)|(-\d+))*\])?)"
      matches = re.findall(rr, machines)
      for pp in matches:
        prefix = pp[1]
        suffix = pp[2]
        if suffix != "":
          for ss in suffix.strip("[]").split(","):
            if "-" in ss:
              ppparts = ss.split("-")
              start = int(ppparts[0])
              end = int(ppparts[1]) + 1
              for ii in range(start, end):
                machine_list.append(prefix + str(ii))
            else:
              machine_list.append(prefix + ss)
        else:
          machine_list.append(prefix)

    slurm_queue = subprocess.check_output(["squeue"])
    machine_count = {}
    lines = slurm_queue.split("\n")[1:]

    for ll in lines:
      tokens = ll.split(" ")
      tokens = filter(lambda t: t != "", tokens)
      if len(tokens) == 0:
        continue
      job_id = tokens[0]
      job_info = subprocess.check_output(["scontrol", "show", "job", job_id])
      m2 = re.search("NodeList=(\w+)", job_info)
      if m2 is None:
        continue
      machine = m2.group(1)
      m = re.search("Gres=gpu:(\d)", job_info)
      num_gpu = int(m.group(1))
      log.info("Machine \"{}\" GPU {}".format(machine, num_gpu))
      if machine not in machine_count:
        machine_count[machine] = num_gpu
      else:
        machine_count[machine] += num_gpu

    hostname = socket.gethostname()
    mm_list = self.slurm_config[hostname]

    count_arr = []
    down_machines = machine_state_dict[
        "down*"] if "down*" in machine_state_dict else []
    log.info("Down machines {}".format(down_machines))
    for jj, mm in enumerate(mm_list):
      if mm["name"] in down_machines:
        mm["priority"] = -10000
      if mm["name"] in machine_count:
        free_gpu = mm["num_gpu"] - machine_count[mm["name"]]
      else:
        free_gpu = mm["num_gpu"]
      count_arr.append((mm, (free_gpu) * mm["priority"]))
    weight_list = np.array([cc[1] for cc in count_arr])
    if weight_list.max() > 0:
      count_idx = np.argmax(weight_list)
      return count_arr[count_idx][0].name
    else:
      return None  # No free GPU available, just wait.

  def create(self, num_gpu, num_cpu, machine=None):
    if machine is None:
      machine = self.select_machine()
    return SlurmCommandDispatcher(
        num_gpu=num_gpu, num_cpu=num_cpu, machine=machine)


class SlurmJobRunnerFactory(object):

  def create(self, request, result_queue, resource_queue, stdout_file=None):
    return JobRunner(
        request,
        SlurmCommandDispatcherFactory().create(
            num_gpu=request.num_gpu, num_cpu=request.num_cpu),
        result_queue=result_queue,
        resource_queue=resource_queue,
        stdout_file=stdout_file)


if __name__ == "__main__":
  print(SlurmCommandDispatcherFactory().select_machine())
