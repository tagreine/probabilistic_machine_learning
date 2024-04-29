from distributions.default_distribution import Default

class Model:

    def default_model(self,p):
        default_distribution = Default()
        return default_distribution.default(p)