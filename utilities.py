import time
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import logging
import sys

# setup logger
def logger(log_dir, need_time=True, need_stdout=False):
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y-%I:%M:%S')
    if need_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        log.addHandler(ch)
    if need_time:
        fh.setFormatter(formatter)
        if need_stdout:
            ch.setFormatter(formatter)
    log.addHandler(fh)
    return log

def timeSince(since):
    s = int(time.time() - since)
    m = math.floor(s / 60)
    s %= 60
    h = math.floor(m / 60)
    m %= 60
    return '%dh %dm %ds' %(h, m, s)

def plot_learning_curve(lines, shapes, colors, labels,
                        save_path, title='', logy=False,
                        ms=1., linewidth=1.,
                        xlabel=None, ylabel=None,
                        ylim=None, yticks=None, xlim=None, xticks=None):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if xlim or xticks: plt.xticks(xticks); plt.xlim(xlim)
    if ylim or yticks: plt.yticks(yticks); plt.ylim(ylim);
    plt.grid(linestyle='dotted')

    for idx, line in enumerate(lines):
        if not logy:
            plt.plot(np.arange(len(line)), line, shapes[idx], color=colors[idx], label=labels[idx],
                 linewidth=linewidth, ms=ms)
        else:
            plt.semilogy(np.arange(len(line)), line, shapes[idx], color=colors[idx], label=labels[idx],
                 linewidth=linewidth, ms=ms)

    plt.legend(loc="upper right")
    plt.savefig(save_path, dpi=900)

def plot_learning_curve_two_side(lines, shapes, colors, labels, positions,
                                 save_path, title='',
                                 ms=1., linewidth=1.,
                                 xlabel=None, ylabel_1=None, ylabel_2=None,
                                 ylim_1=None, yticks_1=None, ylim_2=None, yticks_2=None,
                                 xlim=None, xticks=None):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    plt.title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_1); ax2.set_ylabel(ylabel_2)
    if xlim or xticks: ax1.set_xticks(xticks); ax1.set_xlim(xlim)
    if ylim_1 or yticks_1: ax1.set_yticks(yticks_1); ax1.set_ylim(ylim_1)
    if ylim_2 or yticks_2: ax2.set_yticks(yticks_2); ax2.set_ylim(ylim_2)
    ax1.grid(linestyle='dotted')


    for idx, line in enumerate(lines):
        if positions[idx]==0:
            ax1.plot(np.arange(len(line)), line, shapes[idx], color=colors[idx], label=labels[idx],
                     linewidth=linewidth, ms=ms)
        else:
            ax2.semilogy(np.arange(len(line)), line, shapes[idx], color=colors[idx], label=labels[idx],
                     linewidth=linewidth, ms=ms)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    plt.savefig(save_path, dpi=900)

