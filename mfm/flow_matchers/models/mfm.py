import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, pad_t_like_x
from torch.func import jvp  



class MetricFlowMatcher(ConditionalFlowMatcher):
    def __init__(
        self, geopath_net: torch.nn.Module = None, alpha: float = 1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.geopath_net = geopath_net
        if self.alpha != 0:
            assert (
                geopath_net is not None
            ), "GeoPath model must be provided if alpha != 0"

    def gamma(self, t, t_min, t_max):
        return (
            1.0
            - ((t - t_min) / (t_max - t_min)) ** 2
            - ((t_max - t) / (t_max - t_min)) ** 2
        )

    def d_gamma(self, t, t_min, t_max):
        return 2 * (-2 * t + t_max + t_min) / (t_max - t_min) ** 2

    def compute_mu_t(self, x0, x1, t, t_min, t_max):

        with torch.enable_grad():
            t = pad_t_like_x(t, x0)
            if self.alpha == 0:
                return (t_max - t) / (t_max - t_min) * x0 + (t - t_min) / (
                    t_max - t_min
                ) * x1
            self.geopath_net_output = self.geopath_net(x0, x1, t)
            if self.geopath_net.time_geopath:
                self.doutput_dt = self.doutput_dt_fun(self.geopath_net, x0, x1, t)
        return (
            (t_max - t) / (t_max - t_min) * x0
            + (t - t_min) / (t_max - t_min) * x1
            + self.gamma(t, t_min, t_max) * self.geopath_net_output
        )

    def sample_xt(self, x0, x1, t, epsilon, t_min, t_max):
        mu_t = self.compute_mu_t(x0, x1, t, t_min, t_max)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def sample_location_and_conditional_flow(
        self,
        x0,
        x1,
        t_min,
        t_max,
        training_geopath_net=False,
        midpoint_only=False,
        t=None,
    ):

        self.training_geopath_net = training_geopath_net
        with torch.enable_grad():
            if t is None:
                t = torch.rand(x0.shape[0], requires_grad=True)
            t = t.type_as(x0)
            t = t * (t_max - t_min) + t_min
            if midpoint_only:
                t = (t_max + t_min) / 2 * torch.ones_like(t).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps, t_min, t_max)
        ut = self.compute_conditional_flow(x0, x1, t, xt, t_min, t_max)

        return t, xt, ut

    def compute_conditional_flow(self, x0, x1, t, xt, t_min, t_max):
        del xt
        t = pad_t_like_x(t, x0)
        if self.alpha == 0:
            return (x1 - x0) / (t_max - t_min)
        return (
            (x1 - x0) / (t_max - t_min)
            + self.d_gamma(t, t_min, t_max) * self.geopath_net_output
            + (
                self.gamma(t, t_min, t_max) * self.doutput_dt
                if self.geopath_net.time_geopath
                else 0
            )
        )
    
    @staticmethod
    def doutput_dt_fun(model, x0, x1, t_raw):
        def f(tt):
            t_padded = pad_t_like_x(tt, x0)        
            return model(x0, x1, t_padded)

        _, dydt = jvp(f, (t_raw,), (torch.ones_like(t_raw),))
        return dydt.squeeze(-1)      
