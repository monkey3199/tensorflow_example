import argparse
import json
import logging
import os
import retrying

import tensorflow as tf

def parse_args():
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--sleep_secs",
      default=0,
      type=int,
      help=("Amount of time to sleep at the end"))

  # TODO(jlewi): We ignore unknown arguments because the backend is currently
  # setting some flags to empty values like metadata path.
  args, _ = parser.parse_known_args()
  return args

  @retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000,
                stop_max_delay=60*3*1000)

def run(server, cluster_spec): 
    