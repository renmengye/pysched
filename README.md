# pysched
Pipeline based scheduler made in Python

## Introduction
This is a python made tool for light-weight pipelines, with simple scheduling
scheduling capability. Sometime we would like to have pipeline to process data
in parallel fashion but it is too intimidating to write one ourself. 

Another headache is to automatically choose remote clusters. Since we don't
know about the schedule condition in the future, it is hard to specify the
machine we would like the program to run ahead of time. If we have too many
jobs pending, this would likely to clog the slurm scheduler.

## Job Parallelization
You have several a hundred sets of hyperparameters to run cross validation, but
you only have limited resources within a large computational cluster. You would
like to parallel the jobs with a maximum factor, and run all the jobs in a
queue. You would also like to define job dependencies so that the
hyperparameter the your next set of experiments can come from the best ones
chosen from the current set of experiments.

Check out multi_job_example.py for this usage scenario.

## Pipeline Automation
You have 100K images to run through conv-net. And after each image is done, you
have more post-processing stages. Instead launching individual jobs, or define 
the stages in a bash script, you would like to run a dynamic pipeline that
constantly process your data in a parallel, continuous fashion, under resource
constraints.

Examples for this usage scenario will come out soon.

## Launching Recurring Jobs
You want to launch an evaluation job every few hours of training. But you would
like the scheduler to automatically choose a cluster to run.

Checkout single_job_example.py for this usage scenario.
