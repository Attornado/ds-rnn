"""DeltaRNN cells: temporally-sparse GRU/LSTM (Neil et al., 2017)."""

from dsrnn.nn.deltarnn.deltagru import DeltaGRU
from dsrnn.nn.deltarnn.deltalstm import DeltaLSTM

__all__ = ["DeltaGRU", "DeltaLSTM"]
