"""
Model the name distribution as a function of state and year using
a spatiotemporal Gaussian Process model with multinomial observations.
We use the Polya-gamma augmentation trick to perform fully-Bayesian inference.
"""

import os
import time
import gzip
from zipfile import ZipFile
from urllib.request import urlretrieve
import pickle as pickle
import operator
from collections import namedtuple, defaultdict

import numpy as np
from functools import reduce
import imp
np.random.seed(0)
from scipy.special import gammaln
from scipy.misc import logsumexp

import matplotlib.pyplot as plt
from hips.plotting.layout import create_figure, create_axis_at_location
from hips.plotting.sausage import sausage_plot
import brewer2mpl
colors = brewer2mpl.get_map("Set1", "Qualitative",  9).mpl_colors
goodcolors = np.array([0,1,2,4,6,7,8])
colors = np.array(colors)[goodcolors]

from GPy.kern import RBF, Matern52

from pgmult.utils import pi_to_psi, psi_to_pi, ln_pi_to_psi
import pgmult.gp
imp.reload(pgmult.gp)

import pgmult.distributions
imp.reload(pgmult.distributions)


#############
#  loading  #
#############

def load_data(
        start_train_year=1900, end_train_year=2010, end_year=2013,
        continental=False, DC=True, N_names=None, train_state=None,
        downsample=None):

    # load data, data array is (years x states x names)
    data, years, states, names = download_data(N_names, end_year)

    # get the latitude and longitude
    loc_file = os.path.join('data','names','state_latlon.csv')
    with open(loc_file) as infile:
        latlon = {state:(float(lat),float(lon)) for
                  state, lat, lon in
                  [line.strip().split(',') for line in infile]}

    # flatten out the first two dimensions
    flat_data, flat_years, flat_states, lat, lon = all_flat \
        = flatten(data, years, states, names, latlon)

    # split train and test
    train_inds, test_inds = get_train_test_ind(
        flat_years, flat_states, train_state, continental, DC)
    train_data, train_years, train_states, train_lat, train_lon = \
        [arr[train_inds] for arr in all_flat]
    test_data, test_years, test_states, test_lat, test_lon = \
        [arr[test_inds] for arr in all_flat]

    # downsample
    if downsample is not None:
        print("Downsampling data to ", downsample, " names per year/state")
        assert isinstance(downsample, int) and downsample > 0
        from pgmult.utils import downsample_data
        downsample_train_data = downsample_data(train_data, downsample)
        downsample_test_data = downsample_data(test_data, downsample)
    else:
        downsample_train_data = downsample_test_data = None

    # put into a Dataset namedtuple
    Dataset = namedtuple("Dataset", ["K", "data", "years", "lat", "lon", "states", "names"])

    train = Dataset(N_names, train_data, train_years, train_lat, train_lon, train_states, names)
    test  = Dataset(N_names, test_data,  test_years,  test_lat,  test_lon,  test_states,  names)

    downsample_train = Dataset(
        N_names, downsample_train_data, train_years, train_lat, train_lon, train_states, names)
    downsample_test  = Dataset(
        N_names, downsample_test_data,  test_years,  test_lat, test_lon,  test_states, names)

    return downsample_train, downsample_test, train, test


def download_data(N_names, end_year):
    url = 'http://www.ssa.gov/oact/babynames/state/namesbystate.zip'
    datafile = os.path.join("data", "names", "namesbystate.zip")

    if not os.path.exists(datafile):
        print('Downloading census data for the first time...')
        urlretrieve(url, datafile)
        print('...done!')
    print('Loading census data from zip file...')
    alldata = parse_names_files(ZipFile(datafile), N_names, end_year)
    print('...done!')

    return alldata


def parse_names_files(zfile, N_names, end_year):
    def parse_state(string):
        data = defaultdict(lambda: defaultdict(int))
        for line in string.strip().split('\r\n'):
            state, gender, year, name, count = line.split(',')
            if gender.lower() == 'm' and int(year) <= end_year:
                data[name.lower()][int(year)] += int(count)
        return data

    def get_years(dct):
        sum = lambda lst: reduce(operator.or_, lst)
        return sorted(sum(set(dd.keys()) for d in list(dct.values()) for dd in list(d.values())))

    def get_top_names(dct, N_names):
        counts = defaultdict(int)
        for statedict in list(dct.values()):
            for name, yeardict in statedict.items():
                counts[name] += sum(yeardict.values())  # sum over years
        return sorted(list(counts.keys()), key=counts.__getitem__, reverse=True)[:N_names]

    def get_data_array(dct, years, states, names):
        counts = np.zeros((len(years), len(states), len(names)))
        for year_idx, year in enumerate(years):
            for state_idx, state in enumerate(states):
                for name_idx, name in enumerate(names):
                    counts[year_idx, state_idx, name_idx] = \
                        dct[state][name][year]
        return counts

    # dct[state][name][year] = count
    keys = [key for key in zfile.namelist() if key.endswith('.TXT')]
    dct = {key[:2]: parse_state(zfile.read(key)) for key in keys}

    years = get_years(dct)
    states = sorted(dct.keys())
    names = get_top_names(dct, N_names)
    data = get_data_array(dct, years, states, names)

    return data, years, states, names


def flatten(data, years, states, names, latlon):
    flat_data = data.reshape(-1, len(names))
    flat_years = np.repeat(years, len(states))
    flat_states = np.tile(states, len(years))
    lat, lon = np.array([latlon[st] for st in flat_states]).T
    return flat_data, flat_years, flat_states, lat, lon


def get_train_test_ind(flat_years, flat_states, train_state, continental, DC):
    train_inds = (flat_years >= start_train_year) & (flat_years < end_train_year)
    test_inds  = (flat_years >= end_train_year)

    if train_state is not None:
        train_inds &= (flat_states == train_state)
        test_inds &= (flat_states == train_state)

    if continental:
        not_AK = np.array([st.upper() != "AK" for st in flat_states])
        train_inds &= not_AK
        test_inds  &= not_AK

        not_HI = np.array([st.upper() != "HI" for st in flat_states])
        train_inds &= not_HI
        test_inds  &= not_HI

    if not DC:
        not_DC = np.array([st.upper() != "DC" for st in flat_states])
        train_inds &= not_DC
        test_inds  &= not_DC

    return train_inds, test_inds


#############
#  fitting  #
#############

def get_inputs(data):
    return np.hstack((data.years[:,None], data.lon[:,None], data.lat[:,None]))


def fit_gp_multinomial_model(model, test, pi_train=None, N_samples=100, run=1):
    if pi_train is not None:
        if isinstance(model, pgmult.gp.LogisticNormalGP):
            model.data_list[0]["psi"] = ln_pi_to_psi(pi_train) - model.mu
        elif isinstance(model, pgmult.gp.MultinomialGP):
            model.data_list[0]["psi"] = pi_to_psi(pi_train) - model.mu
            model.resample_omega()
    else:
        model.initialize_from_data()

    ### Inference
    results_base = os.path.join("results", "names", "run%03d" % run, "results")
    results_file = results_base + ".pkl.gz"
    if os.path.exists(results_file):
        with gzip.open(results_file, "r") as f:
            samples, lls, pred_lls, timestamps = pickle.load(f)

    else:
        Z_test = get_inputs(test)
        lls = [model.log_likelihood()]
        samples = [model.copy_sample()]
        pred_ll, pred_pi = model.predictive_log_likelihood(Z_test, test.data)
        pred_lls = [pred_ll]
        pred_pis = [pred_pi]
        times = [0]

        # Print initial values
        print("Initial LL: ", lls[0])
        print("Initial Pred LL: ", pred_lls[0])


        for itr in range(N_samples):
            print("Iteration ", itr)
            tic = time.time()
            model.resample_model(verbose=True)
            times.append(time.time()-tic)

            samples.append(model.copy_sample())
            lls.append(model.log_likelihood())
            pred_ll, pred_pi = model.predictive_log_likelihood(get_inputs(test), test.data)
            pred_lls.append(pred_ll)
            pred_pis.append(pred_pi)

            print("Log likelihood: ", lls[-1])
            print("Pred Log likelihood: ", pred_ll)

            # Save this sample
            # with gzip.open(results_file + ".itr%03d.pkl.gz" % itr, "w") as f:
            #     pickle.dump(model, f, protocol=-1)

        lls = np.array(lls)
        pred_lls = np.array(pred_lls)
        timestamps = np.cumsum(times)

    return samples, lls, pred_lls, pred_pis, timestamps


#############
#  metrics  #
#############

def compute_pred_likelihood(model, samples, test):
    Z_pred = get_inputs(test)

    preds = []
    for sample in samples:
        model.set_sample(sample)
        preds.append(model.predict(Z_pred, full_output=True)[1])

    psi_pred_mean = np.mean(preds, axis=0)

    if isinstance(model, pgmult.gp.MultinomialGP):
        pi_pred_mean = np.array([psi_to_pi(psi) for psi in psi_pred_mean])
    elif isinstance(model, pgmult.gp.LogisticNormalGP):
        from pgmult.internals.utils import ln_psi_to_pi
        pi_pred_mean = np.array([ln_psi_to_pi(psi) for psi in psi_pred_mean])
    else:
        raise NotImplementedError

    pll_gp = gammaln(test.data.sum(axis=1)+1).sum() - gammaln(test.data+1).sum()
    pll_gp += np.nansum(test.data * np.log(pi_pred_mean))
    return pll_gp

def compute_static_pred_ll(static_model, test):
    pll = 0
    years = np.unique(test.years)
    X_test = test.data
    for year in years:
        pll += static_model.predictive_log_likelihood(X_test[test.years==year])
    return pll

def compute_fraction_top_names(test, test_pis, top=10, bottom=10):
    # Use the average pi to compute the predicted top 10 names
    test_pi_mean = np.mean(test_pis, axis=0)

    # Sort names in decreasing order of predicted probability
    pred_perm = np.argsort(-test_pi_mean, axis=1)
    data_perm = np.argsort(-test.data, axis=1)

    # Compute number of top names predicted per year (averaged over states)
    top_scores = np.array([len(np.intersect1d(pp[:top],dp[:top]))
                           for pp,dp in zip(pred_perm, data_perm)])

    bottom_scores = np.array([len(np.intersect1d(pp[-bottom:],dp[-bottom:]))
                              for pp,dp in zip(pred_perm, data_perm)])

    results = np.zeros((len(np.unique(test.years)), 4))
    for i,yr in enumerate(sorted(np.unique(test.years))):
        results[i,0] = top_scores[test.years==yr].mean(0)
        results[i,1] = top_scores[test.years==yr].std(0)

        results[i,2] = bottom_scores[test.years==yr].mean(0)
        results[i,3] = bottom_scores[test.years==yr].std(0)

        print("Top: %d:  %.2f +- %.2f" % (yr, results[i,0], results[i,1]))
        print("Bot: %d:  %.2f +- %.2f" % (yr, results[i,2], results[i,3]))

    return results


##############
#  plotting  #
##############

def plot_census_results(train, samples, test, test_pis):
    # Extract samp[les
    train_mus = np.array([s[0] for s in samples])
    train_psis = np.array([s[1][0][0] for s in samples])
    # omegas = np.array([s[1][0][1] for s in samples])

    # Adjust psis by the mean and compute the inferred pis
    train_psis += train_mus[0][None,None,:]
    train_pis = np.array([psi_to_pi(psi_sample) for psi_sample in train_psis])
    train_pi_mean = np.mean(train_pis, axis=0)
    train_pi_std = np.std(train_pis, axis=0)

    # Compute test pi mean and std
    test_pi_mean = np.mean(test_pis, axis=0)
    test_pi_std = np.std(test_pis, axis=0)

    # Compute empirical probabilities
    train_pi_emp = train.data / train.data.sum(axis=1)[:,None]
    test_pi_emp = test.data / test.data.sum(axis=1)[:,None]


    # Plot the temporal trajectories for a few names
    names = ["Scott", "Matthew", "Ethan"]
    states = ["NY", "TX", "WA"]
    linestyles = ["-", "--", ":"]

    fig = create_figure(figsize=(3., 3))
    ax1 = create_axis_at_location(fig, 0.6, 0.5, 2.25, 1.75)
    for name, color in zip(names, colors):
        for state, linestyle in zip(states, linestyles):
            train_state_inds = (train.states == state)
            train_name_ind = np.array(train.names) == name.lower()
            train_years = train.years[train.states == state]
            train_mean_name = train_pi_mean[train_state_inds, train_name_ind]
            train_std_name = train_pi_std[train_state_inds, train_name_ind]

            test_state_inds = (test.states == state)
            test_name_ind = np.array(test.names) == name.lower()
            test_years = test.years[test.states == state]
            test_mean_name = test_pi_mean[test_state_inds, test_name_ind]
            test_std_name = test_pi_std[test_state_inds, test_name_ind]

            years = np.concatenate((train_years, test_years))
            mean_name = np.concatenate((train_mean_name, test_mean_name))
            std_name = np.concatenate((train_std_name, test_std_name))

            # Sausage plot
            sausage_plot(years, mean_name, std_name,
                         color=color, alpha=0.5)

            # Plot inferred mean
            plt.plot(years, mean_name,
                     color=color, label="%s, %s" % (name, state),
                     ls=linestyle, lw=2)

            # Plot empirical probabilities
            plt.plot(train.years[train_state_inds],
                     train_pi_emp[train_state_inds, train_name_ind],
                     color=color,
                     ls="", marker="x", markersize=4)

            plt.plot(test.years[test_state_inds],
                     test_pi_emp[test_state_inds, test_name_ind],
                     color=color,
                     ls="", marker="x", markersize=4)

    # Plot a vertical line to divide train and test
    ylim = plt.gca().get_ylim()
    plt.plot((test.years.min()-0.5) * np.ones(2), ylim, ':k', lw=0.5)
    plt.ylim(ylim)

    # plt.legend(loc="outside right")
    plt.legend(bbox_to_anchor=(0., 1.05, 1., .105), loc=3,
               ncol=len(names), mode="expand", borderaxespad=0.,
               fontsize="x-small")

    plt.xlabel("Year")
    plt.xlim(train.years.min(), test.years.max()+0.1)
    plt.ylabel("Probability")

    # plt.tight_layout()
    fig.savefig("census_gp_rates.pdf")

    plt.show()
    plt.pause(0.1)

def plot_spatial_distribution(train, samples, name="Ethan", year=2000):
    # Extract data from samples
    # Extract samp[les
    mus = np.array([s[0] for s in samples])
    psis = np.array([s[1][0][0] for s in samples])
    # omegas = np.array([s[1][0][1] for s in samples])

    # Adjust psis by the mean and compute the inferred pis
    psis += mus[0][None,None,:]
    pis = np.array([psi_to_pi(psi_sample) for psi_sample in psis])

    # Extract single name, year data
    data = pis[-1, train.years==year, train.names == name.lower()]
    lons = train.lon[train.years == year]
    lats = train.lat[train.years == year]

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, aspect="equal")

    from mpl_toolkits.basemap import Basemap
    m = Basemap(width=6000000, height=3500000,
                resolution='l',projection='stere',
                lat_ts=50,lat_0=40,lon_0=-100.,
                ax=ax)
    land_color  = [.98, .98, .98]
    water_color = [.75, .75, .75]
    # water_color = [1., 1., 1.]
    m.fillcontinents(color=land_color, lake_color=water_color)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    m.drawmapboundary(fill_color=water_color)

    # Convert data lons and data lats to map coordinates
    dx, dy = m(lons, lats)

    # Interpolate at a grid of points
    glons, glats = m.makegrid(100, 100)
    gx, gy = m(glons, glats)
    M = gx.size

    # Interpolate
    from scipy.interpolate import griddata
    gdata = griddata(np.hstack((dx[:,None], dy[:,None])),
                     data,
                     np.hstack((gx.reshape((M,1)), gy.reshape((M,1)))),
                     method="cubic")
    gdata = gdata.reshape(gx.shape)

    # Plot the contour
    cs = ax.contour(gx, gy, gdata, 15, cmap="Reds", linewidth=2)
    plt.title("%s (%d)" % (name, year))

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(cs, cax=cax)
    cbar.set_label("Probability", labelpad=10)

    plt.subplots_adjust(left=0.05, bottom=0.1, top=0.9, right=0.85)
    fig.savefig("%s_%d_geo.pdf" % (name.lower(), year))

    return fig, ax, m

def plot_pred_ll_vs_time(static_pll, emp_pll,
                         sbm_gp_timestamps, sbm_gp_plls, sbm_avg_pll,
                         lnm_gp_timestamps, lnm_gp_plls, lnm_avg_pll,
                         results_dir=".",
                         begin_test_year=2012, end_test_year=2013,
                         norm=1, baseline=0, burnin=0
                         ):
    # Plot predictive log likelihoods over time
    fig = plt.figure(figsize=(0.38*5.25, 0.45*5.25))
    fig.set_tight_layout(True)

    min_time = lnm_gp_timestamps[burnin]
    max_time = sbm_gp_timestamps[-1]
    # plt.plot(pred_lls, lw=2, color=colors[0], label="LSB GP")
    plt.plot(sbm_gp_timestamps[burnin:], (sbm_gp_plls[burnin:] - baseline) / norm, lw=2, color=colors[0], label="SBM GP")
    plt.plot([min_time, max_time], (sbm_avg_pll - baseline) / norm* np.ones(2), ls='--', color=colors[0])

    plt.plot(lnm_gp_timestamps[burnin:], (lnm_gp_plls[burnin:] - baseline) / norm, lw=2, color=colors[1], label="LNM GP")
    plt.plot([min_time, max_time], (lnm_avg_pll - baseline) / norm * np.ones(2), ls='--', color=colors[1])

    plt.plot([0, max_time], (emp_pll - baseline) / norm * np.ones(2), lw=2, color=colors[2], label="Raw GP")
    # plt.plot([0, max_time], (static_pll - baseline) / norm * np.ones(2), lw=2, color=colors[3], label="2011 Values")

    plt.xscale("log")
    plt.xlim(min_time, max_time)

    # plt.plot(sbm_gp_plls, lw=2, color=colors[3], label="LNM GP")
    # plt.plot([0, N_samples], sbm_avg_pll * np.ones(2), ls='--', color=colors[3])

    plt.legend(loc="lower right", prop={"size": 8})

    plt.xlabel("Time [sec] (log scale)")
    plt.xscale("log")

    plt.ylabel("Pred. LL [nats/name]")
    # plt.ylim(0,0.35)
    # plt.title("Census (%d-%d)" % (begin_test_year, end_test_year))

    # plt.ion()
    plt.show()
    # plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "census_pred_ll_vs_time.pdf"))


############
#  script  #
############

if __name__ == "__main__":
    run = 7
    start_train_year = 1960
    end_train_year = 2012
    N_names = 100
    state = None

    # Set the downsampling level
    downsample = 50

    train, test, full_train, full_test = \
        load_data(start_train_year=start_train_year,
                  end_train_year=end_train_year,
                  N_names=N_names,
                  continental=True, DC=False,
                  train_state=state,
                  downsample=downsample)

    # Make a kernel
    k_year   = Matern52(input_dim=1, active_dims=[0], variance=10, lengthscale=30)
    k_latlon = RBF(input_dim=2, active_dims=[1,2], variance=0.1, lengthscale=5)
    kernel = k_year * k_latlon


    from pgmult.internals.utils import mkdir
    results_dir = os.path.join("results", "names", "run%03d" % run)
    mkdir(results_dir)
    results_file = os.path.join("results", "names", "run%03d" % run, "census_results.pkl.gz")

    # Fit a static model to the last year of training data
    static_model = pgmult.distributions.IndependentMultinomialsModel(train.data[train.years==end_train_year-1])
    static_pll = compute_static_pred_ll(static_model, full_test)
    print("Static (%d) PLL: %f" % (end_train_year-1, static_pll))

    # Fit a standard GP to the raw probabilities
    # THIS OPTIMIZES THE KERNEL PARAMETERS FOR PIs!
    results_file = os.path.join(results_dir, "emp_gp_results.pkl.gz")
    if os.path.exists(results_file):
        with gzip.open(results_file) as f:
            emp_gp_model, emp_pll, emp_ppi = pickle.load(f)
    else:
        emp_gp_model = pgmult.gp.EmpiricalStickBreakingGPModel(train.K, kernel, D=3, alpha=0.1)
        emp_gp_model.add_data(get_inputs(train), train.data, optimize_hypers=True)
        emp_pll, emp_ppi = \
            emp_gp_model.predictive_log_likelihood(get_inputs(full_test), full_test.data)

        with gzip.open(results_file, "w") as f:
            pickle.dump((emp_gp_model, emp_pll, emp_ppi), f, protocol=-1)
    print("Empirical GP PLL: ", emp_pll)

    # Get the pi_train of the empirical model
    emp_pi_train = emp_gp_model.predict(get_inputs(train))[0]

    # Make the Multinomial GP model and add the data
    results_file = os.path.join(results_dir, "sbm_gp_results.pkl.gz")
    N_samples = 20
    burnin = N_samples // 2
    if os.path.exists(results_file):
        with gzip.open(results_file) as f:
            sbm_gp_samples, sbm_gp_plls, sbm_gp_ppis, sbm_gp_timestamps = pickle.load(f)
    else:
        gp_model = pgmult.gp.MultinomialGP(train.K, emp_gp_model.kernel, D=3)
        gp_model.add_data(get_inputs(train), train.data)

        sbm_gp_samples, _, sbm_gp_plls, sbm_gp_ppis, sbm_gp_timestamps = \
            fit_gp_multinomial_model(gp_model, full_test, pi_train=emp_pi_train, N_samples=N_samples)

        with gzip.open(results_file, "w") as f:
            pickle.dump((sbm_gp_samples[-1:], sbm_gp_plls, sbm_gp_ppis, sbm_gp_timestamps), f, protocol=-1)

    sbm_avg_pll = logsumexp(sbm_gp_plls[burnin:]) - np.log(N_samples - burnin)
    print("Stick Breaking Multinomial GP PLL: ", sbm_avg_pll)

    # Also compute predictive likelihood of the *average* predicted pi
    # pll_gp = compute_pred_likelihood(gp_model, samples, test)

    # Now fit a logistic normal GP model with elliptical slice sampling
    N_samples = 200
    results_file = os.path.join(results_dir, "lnm_gp_results.pkl.gz")
    ln_gp_model = pgmult.gp.LogisticNormalGP(train.K, emp_gp_model.kernel, D=3)
    ln_gp_model.add_data(get_inputs(train), train.data)
    if os.path.exists(results_file):
        with gzip.open(results_file) as f:
            lnm_gp_samples, lnm_gp_plls, lnm_gp_ppis, lnm_gp_timestamps = pickle.load(f)
    else:
        lnm_gp_samples, _, lnm_gp_plls, lnm_gp_ppis, lnm_gp_timestamps = \
            fit_gp_multinomial_model(ln_gp_model, full_test, pi_train=emp_pi_train, N_samples=N_samples)

        with gzip.open(results_file, "w") as f:
            pickle.dump((lnm_gp_samples[-1:], lnm_gp_plls, lnm_gp_ppis, lnm_gp_timestamps), f, protocol=-1)

    lnm_avg_pll = logsumexp(lnm_gp_plls[burnin:]) - np.log(N_samples-burnin)
    print("Logistic Normal GP PLL: ", lnm_avg_pll)

    # Also compute the PLL from average of predictive samples,
    # effectively marginalizing over Z_train using Monte Carlo
    # lnm_marg_pll_gp = compute_pred_likelihood(ln_gp_model, lnm_gp_samples[burnin:], full_test)
    # print "Logistic Normal GP marginal PLL: ", lnm_marg_pll_gp


    # Compute fraction of top names predicted
    print("Static predictions: ")
    compute_fraction_top_names(full_test, [np.vstack((static_model.pi, static_model.pi))])
    print("Raw GP")
    compute_fraction_top_names(full_test, [emp_ppi])
    print("SBM GP")
    compute_fraction_top_names(full_test, sbm_gp_ppis[burnin:])
    print("LNM GP")
    compute_fraction_top_names(full_test, lnm_gp_ppis[burnin:])

    plot_pred_ll_vs_time(static_pll, emp_pll,
                         sbm_gp_timestamps, sbm_gp_plls, sbm_avg_pll,
                         lnm_gp_timestamps, lnm_gp_plls, lnm_avg_pll,
                         results_dir=results_dir,
                         norm=full_test.data.sum(),
                         baseline=static_pll)

    # Plot inferred probabilities over time
    # plot_census_results(full_train, sbm_gp_samples[burnin:], full_test, sbm_gp_ppis)
    #
    # # Plot the inferred spatial distribution
    # plot_spatial_distribution(full_train, sbm_gp_samples)
