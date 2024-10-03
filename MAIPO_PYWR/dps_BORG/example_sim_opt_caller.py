'''
Purpose: To initiate the optimization process with BORG, which will iteratively call the Pywr simulation model.
# documentation: https://waterprogramming.wordpress.com/2017/03/06/using-borg-in-parallel-and-serial-with-a-python-wrapper/
'''

import example_sim_opt  # Import main optimization module that uses borg python wrapper

# Module within example
example_sim_opt.Optimization()

# ALMOST WORKING! NEED borg.dll, NOT libborg.so FOR WINDOWS
