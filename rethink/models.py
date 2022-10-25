import tqdm
import torch
from torch import tensor as tt 
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
import pyro.infer.autoguide
from pyro.infer.autoguide import AutoMultivariateNormal, AutoDiagonalNormal, init_to_mean,  AutoLaplaceApproximation
from pyro.optim import Adam
import pandas as pd

class Inputs(object):
    """
    Class function to load in pandas dataframe and make it usable for pyro
    """
    def __init__(self, df, cols_dict = {}):
        assert type(df) is pd.DataFrame, 'provide pandas dataframe'
        self.raw_df = df
        self.df = pd.DataFrame(index = self.raw_df.index )
        self._key_dict = cols_dict
        self.params = {}
        for ef_col in self._key_dict.keys():
            self.params[f"{ef_col}_mean"] = self.raw_df[self._key_dict[ef_col]].mean()
            self.params[f"{ef_col}_std"] = self.raw_df[self._key_dict[ef_col]].values.std()
            self.df[ef_col] = self._standardize( ef_col )

    def get_val(self, var, category = False):
        if category:
            return( tt(self.df.loc[:,var].values).long() )
        else:
            return( tt(self.df.loc[:,var].values).double() )
            
    def _standardize(self, name):
        return (self.raw_df[self._key_dict[name]].values - self.params[f"{name}_mean"])/self.params[f"{name}_std"]
    
    def _unstandardize(self, name):
        return self.params[f"{name}_mean"] + self.params[f"{name}_std"] * self.df[self._key_dict[name]].values


class RegressionBase:
    def __init__(self, X, Y, name = ''):
        self.name = name
        self.X = X
        self.Y = Y
        # if categoricals is None:
        #     categoricals = []
        # for col in set(df.columns) - set(categoricals):
        #     setattr(self, col, tt(df[col].values).double())
        # for col in categoricals:
        #     setattr(self, col, tt(df[col].values).long())
            
    def __call__(self, x=None):
        """
        Simple regression model for X and Y
        """
        a = pyro.sample("a", pyro.distributions.Normal(*tt((0., 0.2))))
        bX = pyro.sample("bX", pyro.distributions.Normal(*tt((0., 0.5))))
        sigma = pyro.sample("sigma", pyro.distributions.Exponential(tt(1.)))
        if x is None:
            mu = pyro.deterministic("mu", a + bX * self.X)
            with pyro.plate("data"):
                pyro.sample("Y", pyro.distributions.Normal(mu, sigma), obs=self.Y)
        else:
            mu = pyro.deterministic("mu", a + bX * x)
            return(pyro.sample('Y', pyro.distributions.Normal(mu, sigma) ) )
        
    def train(self, num_steps, lr=1e-2, restart=True, autoguide=None, use_tqdm=False):
        if restart:
            pyro.clear_param_store()
            if autoguide is None:
                autoguide = AutoMultivariateNormal
            else:
                autoguide = getattr(pyro.infer.autoguide, autoguide)
            self.guide = autoguide(self, init_loc_fn=init_to_mean)
        svi = SVI(self, guide=self.guide, optim=Adam({"lr": lr}), loss=Trace_ELBO())
        loss = []
        if use_tqdm:
            iterator = tqdm.notebook.tnrange(num_steps)
        else:
            iterator = range(num_steps)
        for _ in iterator:
            loss.append(svi.step())
        return loss

    def sample_prior(self, num_samples = 1000, sites=None):
        return {
            k: v.detach().numpy()
            for k, v in Predictive(
                self,
                {},
                return_sites=sites,
                num_samples=num_samples
            )().items()
        }

