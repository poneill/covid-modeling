from collections import namedtuple
import numpy as np
from scipy.special import gammaln
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from tqdm import tqdm
from eda import us_data
T = len(us_data['confirmed'])

log = np.log
exp = np.exp

N = US_POP = 327 * 10**6
underreporting_factors = np.linspace(1, 10, 1000)
doubling_times = np.linspace(2, 7, 1000)

VAR_NAMES = ['s', 'i', 'c', 'ru', 'rc', 'd']
PARAM_NAMES = ['beta', 'delta', 'gamma_u', 'gamma_c', 'mu']
# Param assumptions
incubation_period = 14
recovery_period = 21
fatality_rate = 0.02
R0 = 2.2


iota = 1 / incubation_period
rho = 1 / recovery_period
delta = rho * (fatality_rate) / (1 - fatality_rate)
epsilon = R0 * (rho + delta)

def log_fac(x):
    return gammaln(x + 1)

def sir_deriv(arr, params):
    assert(np.isclose(np.sum(arr), 1))
    s, i, c, ru, rc, d = arr
    beta, delta, gamma_u, gamma_c, mu = params
    ds =  - beta * s * i
    di =  beta * s * i - gamma_u * i - delta * i
    dc = delta * i - (mu + gamma_c) * c
    dru = gamma_u * i
    drc = gamma_c * c
    dd = mu * c
    darr = np.array([ds, di, dc, dru, drc, dd])
    assert(np.isclose(np.sum(darr), 0))
    return darr


def solve_sir(x0, params):
    f = lambda t, x: sir_deriv(x, params)
    assert(np.isclose(sum(x0), 1))
    t_span = (0, T)
    sol = solve_ivp(f, t_span, x0, max_step=1, t_eval=range(T))
    return sol

def init_approximation(params):
    beta, delta, gamma_u, gamma_c, mu = params
    ALPHA = beta - (delta + gamma_u)
    ETA = gamma_c + mu
    coeff = delta * I0/(ALPHA + ETA)
    Kc = -coeff  # c should be zero at t=0
    def c(t):
        return coeff * exp(ALPHA * t) + Kc*exp(-ETA*t)
    def z(t):
        return coeff / ALPHA * exp(ALPHA * t) - Kc / ETA * exp(-ETA*t)
    Kz = -mu * z(0)
    def d(t):
        return mu * z(t) + Kz
    Kru = -gamma_c * z(0)
    def rc(t):
        return gamma_c * z(t) + Kru

    return c, d, rc

def bound(x):
    return np.clip(x, 1/N, 1 - 1/N)

def init_approximation_sse(log_params):
    M = 10
    params = exp(log_params)
    ts = np.arange(T)
    _c, _d, _rc = init_approximation(params)
    c = (lambda x: bound(_c(x)))(ts)[:-2] + 1/N
    d = (lambda x: bound(_d(x)))(ts)[:-2] + 1/N
    rc = (lambda x: bound(_rc(x)))(ts)[:-2] + 1/N
    trash = bound(1 - (c + d + rc))
    obs_c = us_data['confirmed'][:-2]
    obs_d = us_data['deaths'][:-2]
    obs_rc = us_data['recovered']
    obs_trash = N - (obs_c + obs_d + obs_rc)
    prefactor = log_fac(N) - (log_fac(obs_c) + log_fac(obs_d) + log_fac(obs_rc) + log_fac(obs_trash))
    #return sum(((log(c(ts) + 1/N) - log(obs_c + 1/N)))**2) + sum(((log(d(ts) + 1/N) - log(obs_d + 1/N)))**2) + sum((log(rc(ts)[:-2] + 1/N) - log(obs_rc + 1/N))**2)
    return sum(prefactor + obs_c * log(c) + obs_d * log(d) + obs_rc * log(rc) + obs_trash * log(trash))


def mh(lf, x, iterations=10000):
    def q(x):
        return x + np.random.normal(0, 0.1, size=len(x0))
    traj = []
    ll = lf(x)
    accepts = 0
    for iteration in range(iterations):
        xp = q(x)
        llp = lf(xp)
        if log(random.random()) < llp - ll:
            x = xp
            ll = llp
            accepts += 1
        if iteration % 100 == 0:
            traj.append((x, ll))
            print(iteration, ll, accepts, accepts / max(iteration, 1))
    return traj

def fit_init_approximation(tol=10**-14):
    x0 = np.random.normal(0, 1, size=len(PARAM_NAMES))
    # x0 = np.array([ 13.26726095,  -7.21161112,  13.26726049,  -6.55617211,
    #    -52.65910809])
    return minimize(init_approximation_sse, x0, method='powell', options={'maxiter': 100000, 'xtol':tol, 'disp':True})

def check_init_approxiation_fit(tol):
    sol = fit_init_approximation(tol)

def plot_log_params(sol, plot_data=True, plot_legend=True, show=True):
    params = exp(log_params)
    c, d, rc = init_approximation(params)
    obs_c = us_data['confirmed'] / N
    obs_d = us_data['deaths'] / N
    obs_rc = us_data['recovered'] / N
    ts = np.arange(T)
    if plot_data:
        plt.plot(obs_c, linestyle=' ', marker='o', label='obs c')
        plt.plot(obs_d, linestyle=' ', marker='o', label='obs d')
        plt.plot(obs_rc, linestyle=' ', marker='o', label='obs rc')
    plt.plot(c(ts), label='est c', color='b')
    plt.plot(d(ts), label='est d', color='orange')
    plt.plot(rc(ts), label='est rc', color='g')
    if plot_legend:
        plt.legend()
    if show:
        plt.show()


def test_init_approximation():
    # VAR_NAMES = ['s', 'i', 'c', 'ru', 'rc', 'd']
    I0 = 1/N
    ic = [1-I0, I0, 0, 0, 0, 0]
    params = np.array([ 0.82,  0.22,  0.34,  2.30, 10.28]) * 3
    sol = solve_sir(ic, params)
def estimate_init_conds():
    confirmed_cases = 13
    underreporting_factor = 10
    initial_cases = confirmed_cases * underreporting_factor
    susceptible_cases = boston_pop - initial_cases
    infected_cases = initial_cases / 3
    exposed_cases = initial_cases - infected_cases
    s = susceptible_cases / boston_pop
    e = exposed_cases / boston_pop
    i = infected_cases / boston_pop
    d = 0
    r = 0

# def plot_sol(sol):
#     ts = sol.t
#     c = sol.y[VAR_NAMES.index('c'), :]
#     i = sol.y[VAR_NAMES.index('i'), :]
#     y = c + i
#     y0, yf = y[0], y[10]
#     t0, tf = ts[0], ts[10]
#     doublings = np.log2(yf / y0)
#     doubling_time = (tf - t0) / doublings
#     print("doubling time:", doubling_time)
#     for i, var_name in enumerate(var_names):
#         plt.plot(sol.y[i, :], label=var_name)
#     plt.legend()
#     plt.show()

def log_likelihood(sol, us_data):
    obs_c = us_data['confirmed']
    obs_rc = us_data['recovered']
    obs_d = us_data['deaths']
    y_c = sol.y[VAR_NAMES.index('c'), :]
    y_rc = sol.y[VAR_NAMES.index('rc'), :]
    y_d = sol.y[VAR_NAMES.index('d'), :]
    y_trash = 1 - (y_c + y_rc + y_d)
    log_prob = 0
    for t in range(T):
        #print(t)
        C, RC, D = obs_c[t], obs_rc[t], obs_d[t]
        TRASH = N - (C + RC + D)
        c, rc, d, trash = y_c[t], y_rc[t], y_d[t], y_trash[t]
        prefactor = log_fac(N) - (log_fac(C) + log_fac(RC) + log_fac(D) + log_fac(TRASH))
        #print(c, rc, d)
        log_prob_t = prefactor + C * log(c) + RC * log(rc) + D * log(d) + TRASH * log(trash)
        #print(prefactor, log_prob_t)
        log_prob += log_prob_t
    return log_prob_t

def log_likelihood2(sol, us_data):
    obs_c = us_data['confirmed']
    obs_rc = us_data['recovered']
    obs_d = us_data['deaths']
    y_c = sol.y[VAR_NAMES.index('c'), :]
    y_rc = sol.y[VAR_NAMES.index('rc'), :]
    y_d = sol.y[VAR_NAMES.index('d'), :]
    y_trash = 1 - (y_c + y_rc + y_d)
    log_prob = 0
    for t in range(T):
        #print(t)
        C, RC, D = obs_c[t], obs_rc[t], obs_d[t]
        TRASH = N - (C + RC + D)
        c, rc, d, trash = y_c[t], y_rc[t], y_d[t], y_trash[t]
        #print(c, rc, d)
        log_prob_t = -((C - c*N)**2 + (RC - rc*N)**2 + (D - (d*N))**2 + (TRASH - trash*N)**2)
        #print(prefactor, log_prob_t)
        log_prob += log_prob_t
    return log_prob_t


def random_hyp():
    ic = np.array([0.99] + [random.random() * 0.01 for _ in range(len(VAR_NAMES) - 1)])
    ic = ic / sum(ic)

    log_thetas = np.random.normal(0, 1, size=len(PARAM_NAMES))
    thetas = exp(log_thetas)
    thetas[5:] /= 10
    return ic, thetas

def mutate_hyp(hyp):
    ic, thetas = hyp
    log_ic = log(ic)
    new_log_ic = log_ic + np.random.normal(0, 0.01, size=len(ic))
    new_ic = exp(new_log_ic)
    new_ic /= sum(new_ic)
    log_thetas = log(thetas)
    new_log_thetas = log_thetas + np.random.normal(0, 0.01, size=len(thetas))
    new_thetas = exp(new_log_thetas)
    return new_ic, new_thetas

def ll_from_hyp(hyp):
    ic, thetas = hyp
    sol = solve_sir(ic, thetas)
    return log_likelihood2(sol, us_data)

def fit_model(generations=10000):
    ll = None
    traj = []
    acceptances = 0
    while ll is None:
        hyp = random_hyp()
        print(hyp)
        prop_ll = ll_from_hyp(hyp)
        if not np.isnan(prop_ll):
            ll = prop_ll
    for t in range(generations):
        hyp_p = mutate_hyp(hyp)
        ll_p = ll_from_hyp(hyp_p)
        if np.log(random.random()) < ll_p - ll:
            acceptances += 1
            hyp = hyp_p
            ll = ll_p
        if t % 100 == 0:
            traj.append((hyp, ll))
            print(t, ll, "ar:", acceptances / (t + 1))
            print(hyp)
    return traj

def ps_from_lls(lls):
    print("min, max:", min(lls), max(lls))
    a = min(lls)
    expa = exp(a)
    ps = [exp(ll - a) for ll in lls]
    return ps

def check_hyp(hyp):
    sol = solve_sir(*hyp)
    plot_sol(sol)
    plt.plot(us_data['confirmed'] / N, label='obs confirmed', marker='o', linestyle=' ')
    plt.plot(us_data['recovered'] / N, label='obs recovered', marker='o', linestyle=' ')
    plt.plot(us_data['deaths'] / N, label='obs deaths', marker='o', linestyle=' ')
    plt.legend()
