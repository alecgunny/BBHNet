import torch


class TimeSlideDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        timeseries: torch.Tensor,
        sample_rate: float,
        kernel_length: float,
        stride: float,
        batch_size: int,
        livetime: float,
        shift: float,
    ) -> None:
        super().__init__()
        self.timeseries = timeseries
        self.sample_rate = sample_rate
        self.kernel_size = int(kernel_length * sample_rate)
        self.stride_size = int(stride * sample_rate)
        self.batch_size = batch_size
        self.shift_size = int(shift * sample_rate)
        self.livetime_size = int(livetime * sample_rate)

    def steps_for_shift(self, shift: int):
        """Compute the number of kernels that will be taken per shift"""
        num_channels, size = self.timeseries.shape
        shift = abs(shift)  # doesn't matter which direction
        max_shift = shift * (num_channels - 1)

        size -= max_shift + self.kernel_size
        return size // self.stride_size + 1

    def get_step_size(self, batch_size):
        return self.stride_size * (batch_size - 1) + self.kernel_size

    def iter_timeslides(self):
        num_channels = len(self.timeseries)
        T = 0
        i = 1
        while True:
            shift = i * self.shift_size
            num_steps = self.steps_for_shift(shift)
            num_batches, remainder = divmod(num_steps, self.batch_size)
            if remainder:
                num_batches += 1

            shift_idx = [i * abs(shift) for i in range(num_channels)]
            if shift < 0:
                shift_idx.reverse()

            for j in range(num_batches):
                if (j + 1) == num_batches:
                    step_size = self.get_step_size(remainder)
                else:
                    step_size = self.get_step_size(self.batch_size)

                start = j * self.batch_size * self.stride_size
                background = []
                for k, offset in enumerate(shift_idx):
                    offset = start + offset
                    x = self.timeseries[k, offset : offset + step_size]
                    background.append(x)
                yield torch.stack(background)

                T += self.stride_size * self.batch_size
                if T >= self.livetime_size:
                    break
            else:
                # The loop didn't break, so we properly
                # exhausted that shift and we're ready
                # to move on to the next. Do the positive
                # and negative shifts for each shift value
                i *= -1
                if i > 0:
                    i += 1
                continue
            break

    def __iter__(self):
        return self.iter_timeslides()


class ZippedDataset(torch.utils.data.IterableDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        return zip(*self.datasets)
