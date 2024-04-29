# Also known as DGM = DAG (directed acyclic graph)

#  (Sleep) ->  (Happyness) <- (Weather) <- (Seasons) : This means Happyness is  from sleep and weather where weather 

# (Sleep) : Bernoulli distribution -> (Good : 0, Bad : 1) -> Encoded to {0,1}
# (Happyness) : Bernoulli distribution -> (Good : 0, Bad : 1) -> Encoded to {0,1}
# (Weather) : Bernoulli distribution -> (Good : 0, Bad : 1) -> Encoded to {0,1}
# (Seasons) : Categorical 

import numpy as np
import tensorflow_probability as tfp 

tfd = tfp.distributions

tfd.Bernoulli()

