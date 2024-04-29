import tensorflow_probability as tfp
tfd = tfp.distributions

class Default:
    def default(self,p):
        return tfd.Bernoulli(probs=p)