#   Copyright 2018 Floris Laporte
#   MIT License

#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the Software
#   without restriction, including without limitation the rights to use, copy, modify,
#   merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following
#   conditions:

#   The above copyright notice and this permission notice shall be included in all copies
#   or substantial portions of the Software.

#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#   PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
#   THE USE OR OTHER DEALINGS IN THE SOFTWARE.

""" An Efficient Unitary Neural Network implementation for PyTorch
based on https://arxiv.org/abs/1612.05231
"""


# Metadata
# --------------------------------------------------------------------------------

name = "torch_eunn"
__author__ = "Floris Laporte"
__version__ = "0.3.0"
__all__ = ["cmm", "eunn", "ModReLU", "EUNN", "EURNN"]


# Imports
# --------------------------------------------------------------------------------

import torch
from math import pi


# Helper functions
# --------------------------------------------------------------------------------


def cmm(x, y):
    """ Complex elementwise multiplication between two torch tensors
        Args:
            x (torch.tensor): A (n+1)-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (d_0, d_1, ..., d_{n-1}, 2)
            y (torch.tensor): A (n+1)-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (d_0, d_1, ..., d_{n-1}, 2)
        Returns:
            torch.tensor: A (n+1)-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (d_0, d_1, ..., d_{n-1}, 2)
    """
    result = torch.stack(
        [
            x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
            x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0],
        ],
        -1,
    )
    return result


def _cmm_b(x, y):
    """ Backward pass for complex elementwise multiplication between two torch tensors """
    result = torch.stack(
        [
            x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1],
            x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0],
        ],
        -1,
    )
    return result


def _permute(x):
    """ Pairwise permutation of tensor elements along the feature dimension """
    return torch.stack([x[:, 1::2], x[:, ::2]], 2).view(*x.shape)


# PyTorch Functions
# --------------------------------------------------------------------------------


def eunn(angles, x):
    """ Perform the action of a unitary matrix U on x.
    The unitary matrix U is represented by the 'angles' tensor.
    Args:
        angles (torch.tensor): the 2-dimensional torch float tensor with the
            angles of the unitary operation. This tensor must have
            shape (hidden_size, capacity), with 'hidden_size' an even number.
        x (torch.tensor): the 3-dimensional torch float tensor on which to
            operate, with the real and imaginary part stored in the last
            dimension of the tensor; i.e. with shape (batch_size, hidden_size, 2)
    Note:
        The following convention for the unitary representation of a single
        mixing unit was chosen:
        .. math::
            M = \begin{pmatrix}
            e^{i\phi} \cos{\theta} & -e^{i\phi}\sin{\theta} \\
            \sin{\theta} & \cos{\theta}
            \end{pmatrix}
    Note:
        To know which unitary matrix the angles represent, you should operate
        on the (complex) identity matrix I::
            I = torch.stack([torch.eye(hidden_size), torch.zeros((hidden_size, hidden_size))], -1)
    """
    return _EUNN.apply(angles, x)


class _EUNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles, x):
        if angles.ndim != 2 or angles.shape[0] % 2:
            raise ValueError(
                "angles should be a 2-dimensional tensor with "
                "shape (hidden_size, capacity), "
                "with 'hidden_size' an even number."
            )
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(
                "x should be a 3-dimensional torch float tensor "
                "with the real and imaginary part stored in the last "
                "dimension of the tensor; i.e. with "
                "shape (batch_size, hidden_size, 2)"
            )
        if x.shape[1] != angles.shape[0]:
            raise ValueError(
                f"the hidden size of 'x' ('{x.shape[1]}') does not "
                f"correspond to the hidden size of 'angles' ('{angles.shape[0]}')"
            )

        # relevant dimensions
        b = x.shape[0]
        m, c = angles.shape

        # phis and thetas
        phi = angles[::2]
        theta = angles[1::2]

        # calculate the sin and cos of rotation angles
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        zeros = torch.zeros_like(cos_theta)

        # fmt: off
        # calculate the rotation vectors with shape = (c, 1, m, 2)
        diag = torch.stack([
            torch.stack([cos_phi * cos_theta, cos_theta], 1).view(-1, c),
            torch.stack([sin_phi * cos_theta, zeros], 1).view(-1, c),
        ], -1)[None].permute(2, 0, 1, 3)
        offdiag = torch.stack([
            torch.stack([-cos_phi * sin_theta, sin_theta], 1).view(-1, c),
            torch.stack([-sin_phi * sin_theta, zeros], 1).view(-1, c),
        ], -1)[None].permute(2, 0, 1, 3)
        # fmt: on

        # loop over sublayers
        xs = torch.zeros((c, *x.shape), dtype=x.dtype, device=x.device)
        for i, (d, o) in enumerate(zip(diag, offdiag)):
            xs[i] = x
            x = cmm(d, x) + cmm(o, _permute(x))
            x = torch.roll(x, 2 * (i % 2) - 1, 1)

        # fmt: off
        ctx.save_for_backward(angles, cos_phi, sin_phi, cos_theta, sin_theta, zeros, diag, offdiag, xs)
        # fmt: on
        return x

    @staticmethod
    def backward(ctx, dL_dx):
        # fmt: off
        angles, cos_phi, sin_phi, cos_theta, sin_theta, zeros, diag, offdiag, xs = ctx.saved_tensors
        # fmt: on

        dL_ddiag = torch.zeros_like(diag)
        dL_doffdiag = torch.zeros_like(offdiag)

        for i in reversed(range(dL_ddiag.shape[0])):
            x = xs[i]
            dL_dx = torch.roll(dL_dx, 2 * ((i + 1) % 2) - 1, 1)
            dL_ddiag[i] = _cmm_b(x, dL_dx).sum(0, keepdims=True)
            dL_doffdiag[i] = _cmm_b(_permute(x), dL_dx).sum(0, keepdims=True)
            dL_dx = _cmm_b(diag[i], dL_dx) + _permute(_cmm_b(offdiag[i], dL_dx))

        dL_ddiag1_r = dL_ddiag[:, 0, ::2, 0].T
        dL_ddiag2_r = dL_ddiag[:, 0, 1::2, 0].T
        dL_ddiag1_i = dL_ddiag[:, 0, ::2, 1].T
        dL_doffdiag1_r = dL_doffdiag[:, 0, ::2, 0].T
        dL_doffdiag2_r = dL_doffdiag[:, 0, 1::2, 0].T
        dL_doffdiag1_i = dL_doffdiag[:, 0, ::2, 1].T

        # fmt: off
        dL_dphi = -dL_ddiag1_r * sin_phi * cos_theta + dL_ddiag1_i * cos_phi * cos_theta
        dL_dphi += dL_doffdiag1_r * sin_phi * sin_theta - dL_doffdiag1_i * cos_phi * sin_theta
        dL_dtheta = -dL_ddiag1_r * cos_phi * sin_theta - dL_ddiag1_i * sin_phi * sin_theta
        dL_dtheta += -dL_ddiag2_r * sin_theta
        dL_dtheta += -dL_doffdiag1_r * cos_phi * cos_theta - dL_doffdiag1_i * sin_phi * cos_theta
        dL_dtheta += dL_doffdiag2_r * cos_theta
        # fmt: on

        dL_dangles = torch.stack([dL_dphi, dL_dtheta], 1).view(*angles.shape)

        return dL_dangles, dL_dx


# PyTorch Modules
# --------------------------------------------------------------------------------


class ModReLU(torch.nn.Module):
    """ A modular ReLU activation function for complex-valued tensors """

    def __init__(self, hidden_size):
        """ ModReLU
        Args:
            hidden_size (int): the number of features of the input/output tensors.
        """
        super(ModReLU, self).__init__()
        self.hidden_size = hidden_size
        self.bias = torch.nn.Parameter(torch.rand(1, hidden_size))
        self.relu = torch.nn.ReLU()

    def forward(self, x, eps=1e-5):
        """ ModReLU forward
        Args:
            x (torch.tensor): A 3-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (batch_size, hidden_size, 2)
            eps (optional, float): A small number added to the norm of the
                complex tensor for numerical stability (default=1e-5).
        Returns:
            torch.tensor: A 3-dimensional torch float tensor with the real and
                imaginary part stored in the last dimension of the tensor; i.e.
                with shape (batch_size, hidden_size, 2)
        """
        x_re, x_im = x[..., 0], x[..., 1]
        norm = torch.sqrt(x_re ** 2 + x_im ** 2) + eps
        phase_re, phase_im = x_re / norm, x_im / norm
        activated_norm = self.relu(norm + self.bias)
        modrelu = torch.stack(
            [activated_norm * phase_re, activated_norm * phase_im], -1
        )
        return modrelu


class EUNN(torch.nn.Module):
    """ Efficient Unitary Neural Network layer
    This layer works similarly as a torch.nn.Linear layer. The difference in this case
    is however that the action of this layer can be represented by a unitary matrix.
    This EUNN is loosely based on the tunable version of the EUNN proposed in
    https://arxiv.org/abs/1612.05231. However, the last diagonal matrix of phases
    was removed in favor of periodic boundary conditions. This makes the algorithm
    considerably faster and more stable.
    """

    def __init__(self, hidden_size, capacity=None):
        """ Efficient Unitary Neural Network layer
        Args:
            hidden_size (int): the number of features of the input/output tensors.
                This number should be even.
            capacity (int): 0 < capacity <= hidden_size. This number represents the
                number of layers containing unitary rotations. The higher the capacity,
                the more of the unitary matrix space can be filled. This obviously
                introduces a speed penalty. In recurrent neural networks, a small
                capacity is usually preferred.
        """
        if hidden_size % 2 != 0:
            raise ValueError("EUNN `hidden_size` should be even")
        if capacity is None:
            capacity = hidden_size

        self.hidden_size = int(round(hidden_size))
        self.capacity = int(round(capacity))

        super(EUNN, self).__init__()

        self.angles = torch.nn.Parameter(
            2 * pi * torch.rand(self.hidden_size, self.capacity)
        )

    def forward(self, x):
        r""" forward pass through the layer
        Args:
            x (torch.tensor): A 3-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the
                tensor; i.e. with shape (batch_size, hidden_size, 2)
        Returns:
            torch.tensor: A 3-dimensional torch float tensor with the real and
                imaginary part stored in the last dimension of the tensor; i.e.
                with shape (batch_size, hidden_size, 2)
        Note:
            The following convention for the unitary representation of a single
            mixing unit was chosen:
            .. math::
                M = \begin{pmatrix}
                e^{i\phi} \cos{\theta} & -e^{i\phi}\sin{\theta} \\
                \sin{\theta} & \cos{\theta}
                \end{pmatrix}
        """

        # get and validate shape of input tensor:
        b, m, ri = x.shape
        c = self.capacity
        if m != self.hidden_size:
            raise ValueError(
                "Input tensor for EUNN layer has hidden_size of %i, "
                "but the EUNN layer expects a hidden_size of %i" % (m, self.hidden_size)
            )
        return eunn(self.angles, x)


class EURNN(torch.nn.Module):
    """ Efficient Unitary Recurrent Neural Network unit
    This recurrent cell works similarly as a torch.nn.RNN layer. The difference
    in this case is however that the action of the internal weight matrix is
    represented by a unitary matrix.
    """

    def __init__(
        self, input_size, hidden_size, capacity=2, output_type="real", batch_first=False
    ):
        """  Efficient Unitary Recurrent Neural Network unit
        Args:
            input_size (int): the size of the input vector
            hidden_size (int): the size of the internal unitary matrix.
            capacity (int): 0 < capacity <= hidden_size. This number represents
                the number of layers containing unitary rotations. The higher
                the capacity, the more of the unitary matrix space can be
                filled. This obviously introduces a speed penalty. In recurrent
                neural networks, a small capacity is usually preferred.
            output_type (str): how to return the output. Allowed values are
                'complex', real', 'imag', 'abs'.
            batch_first (bool): if the first dimension of the input vector is
                the batch or the sequence.
        """
        super(EURNN, self).__init__()
        self.hidden_size = int(hidden_size)
        self.batch_first = int(batch_first)
        self.output_function = {
            "real": lambda x: x[..., 0],
            "imag": lambda x: x[..., 1],
            "complex": lambda x: x,
            "abs": lambda x: torch.sqrt(torch.sum(x ** 2, -1)),
        }[output_type]
        self.output_type = output_type

        self.input_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_layer = EUNN(hidden_size, capacity=capacity)
        self.modrelu = ModReLU(hidden_size)

    def forward(self, input, state=None):
        """ forward pass through the RNN
        Args:
            input (torch.tensor): A 4-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the tensor;
                i.e. with shape
                    (batch_size, sequence_length, feature_size, 2) if self.batch_first==True
                    (sequence_length, batch_size, feature_size, 2) if self.batch_first==False
        Returns:
            torch.tensor: A 4-dimensional torch float tensor with the
                real and imaginary part stored in the last dimension of the tensor;
                i.e. with shape
                    (batch_size, sequence_length, feature_size, 2) if self.batch_first==True
                    (sequence_length, batch_size, feature_size, 2) if self.batch_first==False
        """

        # apply the input layer to the input up front
        input_shape = input.shape
        input = self.input_layer(input.view(-1, input_shape[-1])).view(
            *(input_shape[:-1] + (-1,))
        )

        # make input complex by adding an all-zero dimension
        input = torch.stack([input, torch.zeros_like(input)], -1)

        # unstack the input
        if self.batch_first:
            input = torch.unbind(input, 1)
        else:
            input = torch.unbind(input, 0)

        # if no internal state is given, create one
        if state is None:
            with torch.no_grad():
                state = torch.zeros_like(input[0])

        # recurrent loop
        output = []
        for inp in input:
            state = self.modrelu(inp + self.hidden_layer(state))
            output.append(self.output_function(state))  # take real part

        # stack output
        if self.batch_first:
            output = torch.stack(output, 1)
        else:
            output = torch.stack(output, 0)

        # return output and internal state
        return output, 