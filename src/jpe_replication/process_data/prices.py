# Convert counts data for regression 
import numpy as np 
import pandas as pd 
import os 
from jpe_replication.process_data.helpers import read_file
from jpe_replication.process_data.jpe_specific_format_tools.format_tools import (
    translate_state_indices, 
    translate_decision_indices
)
