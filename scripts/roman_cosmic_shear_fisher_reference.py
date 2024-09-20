
# this script takes in a config file and outputs 1) covariance 2) Fisher matrix
# usage "python roman_cosmic_shear_fisher.py [yml file]"

import os
import jax
import jax.numpy as np
import jax_cosmo as jc
import numpy as onp
import scipy
import yaml
import os
import argparse

# parse arguments configfile, experiment and redshiftbin
parser = argparse.ArgumentParser()
parser.add_argument('configfile', type=str, help='Config yaml file')
args = parser.parse_args()

#make sure that the configfile exists
if os.path.isfile(args.configfile):
    print("reading config file: %s"%args.configfile)
else:
    print("Config file doesn't exist")
    sys.exit(1)

loader = yaml.SafeLoader
with open(args.configfile) as f:
    config0 = [ c for c in yaml.load_all(f.read(), loader) ]

config = config0[0]
config['config_file'] = args.configfile

nbin = config['nofz']['nbin']
nofz_dir = config['nofz']['nofz_dir']
nofz_file = config['nofz']['nofz_file']
reference = config['nofz']['reference']
neff = config['survey']['neff']
ell_min = config['analysis']['ell_min']
ell_max = config['analysis']['ell_max']

ell = np.logspace(np.log10(ell_min), np.log10(ell_max))
fsky = config['survey']['area']/(4*np.pi*180**2/np.pi**2)

sigma_sigma8 = config['priors']['sigma_sigma8']
sigma_omega_c = config['priors']['sigma_omega_c']
sigma_omega_b = config['priors']['sigma_omega_b']
sigma_h = config['priors']['sigma_h']
sigma_ns = config['priors']['sigma_ns']
sigma_w0 = config['priors']['sigma_w0']
sigma_wa = config['priors']['sigma_wa']

sigma_A = config['priors']['sigma_A']
sigma_eta = config['priors']['sigma_eta']

sigma_m_opt = config['priors']['sigma_m_opt']
sigma_z_opt = config['nofz']['sigma_z_opt'] # will need to change this to read in multiple values
                 
outfile = config['outfile']

# read in the n(z)
Z = []
NZ = []

if reference==True:
    
    for i in range(nbin):
        z = onp.loadtxt(nofz_dir+'WFIRST_zdist_sources_bin'+str(i)+'.txt')[:,0]
        nz = onp.loadtxt(nofz_dir+'WFIRST_zdist_sources_bin'+str(i)+'.txt')[:,1]
        Z.append(z)
        NZ.append(nz)

else:
    sompz = np.load(nofz_dir+nofz_file+'.yaml_som_z.npy')

    for i in range(nbin):
        z = sompz[:,0]
        nz = sompz[:,i+1]
        Z.append(z)
        NZ.append(nz)   
    
Z = np.array(Z)
NZ = np.array(NZ)

# Effective number of sources
neff_s = np.ones(nbin)*neff
nzs_s = [jc.redshift.kde_nz(Z[i], NZ[i], bw=0.01, gals_per_arcmin2=neff_s[i]) for i in range(nbin)]


# Define a function to go to and from a 1d parameter vector
def get_params_vec(cosmo, m, dz, ia):
   
    return np.array([ 
        # Cosmological parameters
        cosmo.sigma8, cosmo.Omega_c, cosmo.Omega_b,
        cosmo.h, cosmo.n_s, cosmo.w0, cosmo.wa] + m + dz + ia).flatten()
    
def unpack_params_vec(params):
    # Retrieve cosmology
    cosmo = jc.Cosmology(sigma8=params[0], Omega_c=params[1], Omega_b=params[2],
                         h=params[3], n_s=params[4], w0=params[5],
                         Omega_k=0., wa=params[6])
    A = params[7+2*nbin]
    eta = params[7+2*nbin+1]
    return cosmo, [params[7+i] for i in range(nbin)], [params[7+nbin+i] for i in range(nbin)], [A, eta]

# Mean data vector
@jax.jit
def mu(params):
    # First unpack parameter vector
    cosmo, m, dz, (A, eta) = unpack_params_vec(params) 

    # Build source nz with redshift systematic bias
    nzs_s_sys = [jc.redshift.systematic_shift(nzi, dzi) 
                for nzi, dzi in zip(nzs_s, dz)]

    # Define IA model, z0 is fixed
    b_ia = jc.bias.des_y1_ia_bias(A, eta, 0.62)

    # Define the lensing and number counts probe
    probes = [jc.probes.WeakLensing(nzs_s_sys, ia_bias=b_ia, multiplicative_bias=m)]
    cl = jc.angular_cl.angular_cl(cosmo, ell, probes)

    return cl

## Building a Fisher matrix, we just need the covarianc
@jax.jit
def cov(params):
    
    cl_signal = mu(params)
    
    # First unpack parameter vector
    cosmo, m, dz, (A, eta) = unpack_params_vec(params) 
    
    # Build source nz with redshift systematic bias
    nzs_s_sys = [jc.redshift.systematic_shift(nzi, dzi) 
                for nzi, dzi in zip(nzs_s, dz)]
    
    # Define IA model, z0 is fixed
    b_ia = jc.bias.des_y1_ia_bias(A, eta, 0.62)
    
    # Define the lensing and number counts probe
    probes = [jc.probes.WeakLensing(nzs_s_sys, ia_bias=b_ia, multiplicative_bias=m, sigmae=0.26)]
    
    cl_noise = jc.angular_cl.noise_cl(ell, probes)
    
    cov = jc.angular_cl.gaussian_cl_covariance(ell, probes, cl_signal, cl_noise, f_sky=fsky, sparse=False)
    
    return cov

def symmetrized_matrix(U):
    u"""Return a new matrix like `U`, but with upper-triangle elements copied to lower-triangle ones."""
    M = U.copy()
    inds = onp.triu_indices_from(M,k=1)
    M[(inds[1], inds[0])] = M[inds]
    return M

def symmetric_positive_definite_inverse(M):
    u"""Compute the inverse of a symmetric positive definite matrix `M`.

    A :class:`ValueError` will be thrown if the computation cannot be
    completed.

    """
    import scipy.linalg
    U,status = scipy.linalg.lapack.dpotrf(M)
    if status != 0:
        raise ValueError("Non-symmetric positive definite matrix")
    M,status = scipy.linalg.lapack.dpotri(U)
    if status != 0:
        raise ValueError("Error in Cholesky factorization")
    M = symmetrized_matrix(M)
    return M
                               
# define cosmology
                               
fid_cosmo = jc.Cosmology(sigma8=0.831, Omega_c=0.3156-0.0492, Omega_b=0.0492,
    h=0.6727, n_s=0.9645, w0=-1., Omega_k=0., wa=0.)
fid_params = get_params_vec(fid_cosmo, np.zeros(nbin).tolist(), np.zeros(nbin).tolist(), [1.0, 0.])

# get DV and covariance
cl_gg = mu(fid_params)
C = cov(fid_params)
jacobian = jax.jit(jax.jacfwd(lambda p: mu(p).flatten()))
j = jacobian(fid_params)

j64 = onp.array(j).astype(onp.float64)
C64 = onp.array(C).astype(onp.float64)

# And we get the fisher matrix from the jacobian and covariance
CC = symmetric_positive_definite_inverse(C64)
F = onp.einsum('ia,ij,jb->ab', j64, CC, j64)
F = 0.5*(F + F.T)

d = onp.zeros(len(fid_params))

d[0] = 1./(sigma_sigma8)**2
d[1] = 1./(sigma_omega_c)**2
d[2] = 1./(sigma_omega_b)**2
d[3] = 1./(sigma_h)**2
d[4] = 1./(sigma_ns)**2
d[5] = 1./(sigma_w0)**2
d[6] = 1./(sigma_wa)**2

d[7:7+nbin] = 1./(sigma_m_opt)**2
d[7+nbin:7+2*nbin] = 1./(sigma_z_opt)**2

d[7+2*nbin] = 1./(sigma_A)**2
d[7+2*nbin+1] = 1./(sigma_eta)**2

F_noprior = F.copy()
F_wprior = F + np.diag(d)

cov_estimate = symmetric_positive_definite_inverse(F)
np.savez(outfile, F_nopriors=F_noprior, F_wprior=F_wprior, cov=cov_estimate)
