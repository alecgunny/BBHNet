import itertools
from collections import defaultdict
from typing import Optional

import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC

from aframe.utils import x_per_y


class TimeSlideAUROC(Metric):
    def __init__(
        self,
        max_fpr: float,
        stride: float,
        pool_length: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric = BinaryAUROC(max_fpr)
        pool_size = int(pool_length / stride)
        pool_stride = int(pool_size // 2)
        self.pool = torch.nn.MaxPool1d(pool_size, pool_stride, ceil_mode=True)

        self.add_state("shifts", default=[])
        self.add_state("background", default=[])
        self.add_state("foreground", default=[])

    def update(
        self, shift: int, background: torch.Tensor, foreground: torch.Tensor
    ) -> None:
        self.shifts.append(torch.Tensor([shift]))
        self.background.append(background)
        self.foreground.append(foreground)

    def compute(self):
        foreground, background = [], defaultdict(list)
        for i, bg, fg in zip(self.shifts, self.background, self.foreground):
            foreground.append(fg)
            background[i.item()].append(bg)
        foreground = torch.cat(foreground)

        pooled_background = []
        for bg in background.values():
            bg = torch.cat(bg).view(1, 1, -1)
            bg = self.pool(bg).view(-1)
            pooled_background.append(bg)
        background = torch.cat(pooled_background)

        # concatenate these with view-averaged foreground
        # predictions to constitute our predicted outputs
        y_pred = torch.cat([background, foreground])

        # now create ground-truth labels
        y = torch.zeros_like(y_pred)
        y[len(background):] = 1

        # shuffle the prediction and target arrays up
        # front so that constant-output models don't
        # accidently come out perfect
        idx = torch.randperm(len(y_pred))
        y_pred = y_pred[idx]
        y = y[idx]
        return self.metric(y_pred, y)


class TimeSlide(torch.utils.data.IterableDataset):
    def __init__(
        self,
        timeseries: torch.Tensor,
        shift_size: int,
        kernel_size: float,
        stride_size: float,
        batch_size: int,
        start: int = 0,
        stop: int = -1,
    ) -> None:
        self.timeseries = timeseries
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.batch_size = batch_size
        self.shift_size = shift_size
        self.start = start
        self.stop = stop if stop != -1 else timeseries.size(-1)
        self.step_size = stride_size * (batch_size - 1) + kernel_size

        shifts = [i * abs(shift_size) for i in range(len(timeseries))]
        if shift_size < 0:
            shifts.reverse()
        self.shifts = shifts

    @property
    def max_shift(self):
        num_channels = len(self.timeseries)
        return abs(self.shift_size) * (num_channels - 1)

    @property
    def num_steps(self):
        size = self.stop - self.start - self.max_shift - self.kernel_size
        return size // self.stride_size + 1

    def new_bounds(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None
    ):
        return TimeSlide(
            self.timeseries,
            self.shift_size,
            self.kernel_size,
            self.stride_size,
            self.batch_size,
            self.start if start is None else start,
            stop or self.stop
        )

    def __len__(self):
        return x_per_y(self.num_steps, self.batch_size)

    def __iter__(self):
        for i in range(len(self)):
            start = self.start + i * self.batch_size * self.stride_size
            X = []
            for j, offset in enumerate(self.shifts):
                offset = start + offset
                stop = min(self.stop, offset + self.step_size)
                x = self.timeseries[j, offset:stop]

                if len(x) < self.step_size:
                    size = len(x) - self.kernel_size
                    strides = int(size // self.stride_size)
                    stop = strides * self.stride_size + self.kernel_size
                    x = x[:stop]
                X.append(x)

            minlen = min([len(x) for x in X])
            X = [x[:minlen] for x in X]
            yield torch.stack(X)


def get_timeslides(
    timeseries: torch.Tensor,
    livetime: float,
    sample_rate: float,
    sample_length: float,
    stride: float,
    batch_size: int
):
    duration = timeseries.size(-1) / sample_rate
    duration -= sample_length

    kernel_size = int(sample_length * sample_rate)
    stride_size = int(stride * sample_rate)

    # create alternating positive and negative
    # shifts until we have enough livetime
    shifts = zip(itertools.count(1, 1), itertools.count(-1, -1))
    shifts = itertools.chain.from_iterable(shifts)
    shifts = itertools.takewhile(lambda _: livetime > 0, shifts)
    timeslides = []
    for shift in shifts:
        timeslide = TimeSlide(
            timeseries,
            int(shift * sample_rate),
            kernel_size,
            stride_size,
            batch_size,
        )
        dur = duration - timeslide.max_shift / sample_rate
        livetime -= dur
        timeslides.append(timeslide)

    # chop off any excess time from the last timeslide
    if livetime < 0:
        remainder = dur + livetime + sample_length
        stop = int(remainder * sample_rate)
        timeslides[-1].stop = stop

    # check if we're running distributed, and if not
    # run all the timeslides on the current device
    try:
        world_size = torch.distributed.get_world_size()
    except RuntimeError:
        return timeslides
    if world_size == 1:
        return timeslides

    # if we're running with more than one device,
    # break up the timeslides such that we do
    # roughly an equal number of steps per device
    total_steps = sum([i.num_steps for i in timeslides])
    steps_per_dev = x_per_y(total_steps, world_size)

    # for each timeslide, check if adding the full
    # timeslide to the current device would go over
    # the target number of steps
    timeslides_per_dev = [[]]
    it = iter(timeslides)
    timeslide = next(it)
    while True:
        current_steps = sum([t.num_steps for t in timeslides_per_dev[-1]])
        if current_steps + timeslide.num_steps > steps_per_dev:
            # stop the current timeslide at the index
            # that gives us the desired number of steps
            num_steps = steps_per_dev - current_steps + 1
            stop = timeslide.start + kernel_size + num_steps * stride_size
            new = timeslide.new_bounds(stop=stop)

            # add this truncated timeslide to our current list,
            # then create a new list to start adding to
            timeslides_per_dev[-1].append(new)
            timeslides_per_dev.append([])

            # start a new truncated timeslide one stride
            # after the last step of the previous one
            start = stop - stride_size - kernel_size
            timeslide = timeslide.new_bounds(start=start)
        else:
            # if this timeslide won't put us over, add the
            # whole thing as is and try to move on to the next
            timeslides_per_dev[-1].append(timeslide)
            try:
                timeslide = next(it)
            except StopIteration:
                break

    # retrieve just the timeslides we need for this device
    global_rank = torch.distributed.get_rank()
    return timeslides_per_dev[global_rank]
