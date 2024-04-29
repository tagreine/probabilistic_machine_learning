from distributions.default_distribution import Default
from models.default_model import Model

dd = Default()
dm = Model() 

def app():
    pass

if __name__ == "__main__":
    print(dm.default_model(p=0.2))
