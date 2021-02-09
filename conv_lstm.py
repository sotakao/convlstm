"""adopted from https://github.com/CIA-Oceanix/DinAE_4DVarNN_torch/blob/master/torch_4DVarNN_dinAE.py"""

import torch

Tensor = torch.Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvLSTMCell(torch.nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        kernel_size: int = 3,
        peephole_connection: bool = False) -> Tensor:
        super().__init__()
        # input_size and hidden_size refers to the number of channels in the input and hidden states
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.convolution = torch.nn.Conv2d(
            input_size + hidden_size, 4 * hidden_size,
            kernel_size = self.kernel_size,
            stride = 1,
            padding = self.padding)
        self._Wc = None

    def set_cell_weights(self, input_shape):
        if self._Wc is None:
            weights = [torch.randn(input_shape) for _ in range(3)]
            self._Wc = [torch.nn.Parameter(w).to(device) for w in weights]
        return self._Wc

    def forward(self, input_, prev_state):
        """
        Args:
            input_: tensor of size [batch, input_channel, height, width]
            prev_state: tensor of size [batch, hidden_channel, height, width]
        """
        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # apply convolution to input x_t and hidden state h_{t-1}
        stacked_inputs = torch.cat((input_, prev_hidden), dim=1)
        gates = self.convolution(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, forget_gate, out_gate, cell_gate = gates.chunk(4, dim=1)

        # set peephole connection if True
        if self.peephole_connection is True:
            Wc = self.set_cell_weights(spatial_size)
            in_gate += Wc[0] * prev_cell
            forget_gate += Wc[1] * prev_cell
            out_gate += Wc[2] * cell_gate

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (forget_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

