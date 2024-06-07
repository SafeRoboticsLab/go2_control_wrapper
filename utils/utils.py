from typing import Optional, Union, Dict
import sys
import os
import argparse
import pickle
import numpy as np
import torch
import yaml


class Range(object):

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __str__(self):
        return '[{0},{1}]'.format(self.start, self.end)


def bool_type(string):
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_obj(obj, filename, protocol=None):
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def load_obj(filename):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


class PrintLogger(object):
    """
  This class redirects print statements to both console and a file.
  """

    def __init__(self, log_file):
        self.terminal = sys.stdout
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_model_index(parent_dir,
                    model_type,
                    step: Optional[int] = None,
                    type="mean_critic",
                    cutoff=0,
                    autocutoff=None):
    """_summary_

  Args:
      parent_dir (str): dir of where the result is
      model_type (str): type of model ("highest", "safest", "worst", "manual")
      step (int, optional): step to load in manual
      type (str, optional): model type to read index from [mean_critic, adv_critic, dstb, ctrl]. Defaults to "mean_critic".
      cutoff (int, optional): Model cutoff, will not consider any model index that's lower than this value. Defaults to None.
      autocutoff (float, optional): Auto calculate where to cutoff by taking percentage of the horizon. Defaults to None

  Raises:
      ValueError: _description_

  Returns:
      _type_: _description_
  """
    print("WARNING: using model index stored in {}".format(type))
    model_dir = os.path.join(parent_dir, "model", type)
    chosen_run_iter = None
    print("\tModel type: {}".format(model_type))

    if model_type == "highest":
        model_list = os.listdir(model_dir)
        highest_number = sorted(
            [int(a.split("-")[1].split(".")[0]) for a in model_list])[-1]
        print("\t\tHighest training number: {}".format(highest_number))
        chosen_run_iter = highest_number

    elif model_type == "safest" or model_type == "worst":
        # get the run with the best result from train.pkl
        train_log_path = os.path.join(parent_dir, "train.pkl")
        with open(train_log_path, "rb") as log:
            train_log = pickle.load(log)
        data = np.array(sorted(train_log["pq_top_k"]))
        if autocutoff is not None:
            print("\t\t\tAuto cutting off with {} ratio".format(autocutoff))
            cutoff = max(data[:, 1]) * autocutoff
        data = data[data[:, 1] > cutoff]
        print("\t\t\tTaking {} of max value {}".format(autocutoff,
                                                       max(data[:, 1])))

        if model_type == "safest":
            safe_rate, best_iteration = data[-1]
            print("\t\tBest training iteration: {}, safe rate: {}".format(
                best_iteration, safe_rate))
        else:
            safe_rate, best_iteration = data[0]
            print("\t\tWorst training iteration: {}, safe rate: {}".format(
                best_iteration, safe_rate))
        chosen_run_iter = best_iteration

    elif model_type == "manual":
        model_list = os.listdir(model_dir)
        iter_list = [int(a.split("-")[1].split(".")[0]) for a in model_list]
        if step in iter_list:
            print("\t\tManual pick a model iteration: {}".format(step))
            chosen_run_iter = step
        else:
            raise ValueError(
                "Cannot find iteration {} in list of runs".format(step))

    assert chosen_run_iter is not None, "Something is wrong, cannot find the chosen run, check evaluation config"

    return int(chosen_run_iter), os.path.join(parent_dir, "model")


def combine_obsrv(obsrv: torch.Tensor,
                  action: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Combines observation and action into a single tensor.

  Args:
      obsrv (torch.Tensor): observation tensor.
      action (Union[np.ndarray, torch.Tensor]): actions that are observed.

  Returns:
      torch.Tensor: combined observation.
  """
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(action).to(obsrv)
    return torch.cat([obsrv, action], dim=-1)


class Struct:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


KEYS = [
    'environment', 'training', 'arch_performance', 'arch_backup',
    'update_performance', 'update_backup'
]


def dict2object(dictionary, key):
    return Struct(**dictionary[key])


def load_config(file_path):
    with open(file_path) as f:
        data: Dict = yaml.safe_load(f)
    config_dict = {}
    for key, value in data.items():
        config_dict[key] = Struct(**value)
    return config_dict


def dump_config(file_path, objects, keys=KEYS):
    data = {}
    for key, object in zip(keys, objects):
        data[key] = object.__dict__
    with open(file_path, "w") as f:
        yaml.dump(data, f)
