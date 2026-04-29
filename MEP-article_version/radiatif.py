"""
Radiative code "Herbert & Paillard" (2011-2013)
described in:
  Herbert C., Paillard D., Dubrulle B. J. Clim. 2013
      supplementary material : https://dx.doi.org/10.1175/JCLI-D-13-00060.s1

D. Paillard november 2017
didier.paillard@lsce.ipsl.fr


D. Paillard september 2020 : with the implementation of derivatives

"""

import numpy as np
import mpmath as fp  # used for polylog functions
import profile_bis as prf  # attention : profile is also a public library
import physics as phy
import constants as cst

# irSigma = 5.6704e-8  # = sigma
irSigma = cst.irSigma

############################
#
#
#   the LW budget is computed as bLW(i) = Sum[ Lij sigma Tj^4 , j=0 to nLevel], i.e. a linear function of (sigma T^4)
#       where Lij functions of Ti (i=0 to nLevel), CO2 and H2O concentrations in the atmosphere (i=0 to nLevel-1)
#
# when considering only the temperature dependence in T^4 as a first approximation (i.e. assuming L constant),
# we have an approximation of the gradient: (dbLWi/dTj) = 4 Lij sigma Tj^3
#
#############################

irBand_beta = 1.66  # = 1/mu
irBand_cte = 1.43878  # = 100 (h c)/k

irBand_size = 23
irBand_start = [0, 40, 160, 280, 380, 500, 600, 667, 720, 800, 837, 900, 1000, 1200, 1350, 1450, 1550, 1650, 1750, 1850,
                1950, 2050, 2200]
# irBand_end = [40, 160, 280, 380, 500, 600, 667, 720, 800, 837, 900, 1000, 1200, 1350, 1450, 1550, 1650, 1750, 1850,
# 1950, 2050, 2200, 1000000]
irBand_co2_k = [0, 0, 0, 0, 0, 0, 0, 653.8, 653.8, 653.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
irBand_co2_a = [0, 0, 0, 0, 0, 0, 0, 0.129, 0.129, 0.129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
irBand_h2o_k = [579.75, 7210.3, 6024.8, 1614.1, 139.03, 21.64, 2.919, 2.919, 0.3856, 0.0715, 0.0715, 0.0209, 0, 12.65,
                134.4, 632.9, 331.2, 434.1, 136., 35.65, 9.015, 1.529, 0]
irBand_h2o_a = [0.093, 0.182, 0.094, 0.081, 0.08, 0.068, 0.06, 0.06, 0.059, 0.067, 0.067, 0.051, 0, 0.089, 0.23, 0.32,
                0.296, 0.452, 0.359, 0.165, 0.104, 0.116, 0]


#   Planck function : b(x) = (15/pi^4) x^3 / (exp(x)-1)
#
def fp_planck_b(x):
    if x == 0:
        return 0
    return (15 / fp.pi ** 4) * (x ** 3) / (fp.exp(x) - 1)


def planck_b(x):
    return float(fp_planck_b(x))


# primitive of Planck function b : b1(x) = 1 + (15/pi^4) ( x^3 Log(1-exp(-x)) - 3 x^2 PolyLog[2,exp(-x)] - 6 x
# PolyLog[3,exp(-x)] - 6 PolyLog[4,exp(-x)] )
#
def fp_planck_b1(x):
    if x == 0:
        return 0
    return 1 + (15 / fp.pi ** 4) * (
            (x ** 3) * fp.log(1 - fp.exp(-x))
            - 3 * (x ** 2) * fp.polylog(2, fp.exp(-x))
            - 6 * x * fp.polylog(3, fp.exp(-x))
            - 6 * fp.polylog(4, fp.exp(-x)))


def planck_b1(x):
    return abs(fp_planck_b1(x))


#   Planck values for each bands at given temperature T
#
# def planck_vector(T):               #       returns 24 values [0,23]
#    a = (irBand_cte/T)
#    pb = np.empty([irBand_size+1])
#    for i in range(irBand_size):
#        pb[i] = planck_b( a*irBand_start[i] )
#    pb[irBand_size] = 0
#    return pb


#   integrals of Planck functions for each bands at given temperature T
#
def planck_integ_vector(T):  # returns 23 values [0,22]
    a = (irBand_cte / T)
    ib = [planck_b1(a * irBand_start[i]) for i in range(irBand_size)]
    pb = np.empty([irBand_size], dtype=np.float)
    for i in range(irBand_size - 1):
        pb[i] = ib[i + 1] - ib[i]
    pb[irBand_size - 1] = 1. - ib[irBand_size - 1]
    return pb


def planck_Dinteg_vector(T):  # returns 23 values [0,22]
    a = (irBand_cte / T)
    asT = (-a / T)
    ib = [irBand_start[i] * planck_b(a * irBand_start[i]) for i in range(irBand_size)]
    pb = np.empty([irBand_size], dtype=np.float)
    for i in range(irBand_size - 1):
        pb[i] = asT * (ib[i + 1] - ib[i])
    pb[irBand_size - 1] = -asT * ib[irBand_size - 1]
    return pb


def planck_integ_matrix(vT):  # returns tensor[nLevel+1,irBand_size]
    nLevPlus1 = vT.size
    ip = np.empty([nLevPlus1, irBand_size])
    for i in range(nLevPlus1):
        ip[i, :] = planck_integ_vector(vT[i])
    return ip


def planck_Dinteg_matrix(vT):  # returns tensor[nLevel+1,irBand_size]
    nLevPlus1 = vT.size
    ip = np.empty([nLevPlus1, irBand_size])
    for i in range(nLevPlus1):
        ip[i, :] = planck_Dinteg_vector(vT[i])
    return ip


#   IR absorption for CO2 and H2O
#
def tauC(k, x):
    if irBand_co2_k[k] > 0:
        c = irBand_co2_k[k] * x * irBand_beta
        # return c/fp.sqrt(1+c/irBand_co2_a[k])
        return c / np.sqrt(1 + c / irBand_co2_a[k])
    else:
        return 0


def dtauC_dC(k, x):
    if irBand_co2_k[k] > 0:
        a = irBand_co2_k[k] * irBand_beta
        b = irBand_co2_a[k]
        # return (a/2)*(2*b+a*x)/(b+a*x)/fp.sqrt(1+a*x/b)

        # print("a,b, z = ",a," ",b," ", (a/2)*(2*b+a*x)/(b+a*x)/np.sqrt(1+a*x/b))
        return (a / 2) * (2 * b + a * x) / (b + a * x) / np.sqrt(1 + a * x / b)
    else:
        return 0


# dtauC_dC_vec = np.vectorize(dtauC_dC)

def tauH(k, x):
    if irBand_h2o_k[k] > 0:
        c = irBand_h2o_k[k] * x * irBand_beta
        # return c/fp.sqrt(1+c/irBand_h2o_a[k])
        return c / np.sqrt(1 + c / irBand_h2o_a[k])
    else:
        return 0


def dtauH_dH(k, x):
    if irBand_h2o_k[k] > 0:
        a = irBand_h2o_k[k] * irBand_beta
        b = irBand_h2o_a[k]
        # return (a/2)*(2*b+a*x)/(b+a*x)/fp.sqrt(1+a*x/b)
        return (a / 2) * (2 * b + a * x) / (b + a * x) / np.sqrt(1 + a * x / b)
    else:
        return 0


def exptauC(k, x):
    # return fp.exp( -tauC(k,x) )
    return np.exp(-tauC(k, x))


def exptauH_with_continuum(k, x):
    if 9 <= k <= 12:
        # return fp.exp( -tauH( k, x ) - 0.1*x*irBand_beta )
        return np.exp(-tauH(k, x) - 0.1 * x * irBand_beta)
    else:
        # return fp.exp( -tauH(k,x) )
        return np.exp(-tauH(k, x))


def exptau_vector(uC, uH):  # returns 23 values [0,22]
    return [exptauC(k, uC) * exptauH_with_continuum(k, uH) for k in range(irBand_size)]


def dexptau_dc_vector(uC, uH):  # returns 23 values [0,22]
    v = np.empty([irBand_size])
    for k in range(irBand_size):
        """
        print("k = ",k)
        print("uC = ",uC)
        print("uH = ",uH)
        print(exptauC(k,uC)*exptauH_with_continuum(k,uH))
        """
        # print("dtauCv = ",dtauC_dC(k,uC))
        v[k] = -dtauC_dC(k, uC) * exptauC(k, uC) * exptauH_with_continuum(k, uH)
    # print("v = ",v)
    # print("uC = ",uC)
    # print("uH = ",uH)
    return v


def dexptau_dh_vector(uC, uH):  # returns 23 values [0,22]
    v = np.empty([irBand_size])
    for k in range(irBand_size):
        if 9 <= k <= 12:
            v[k] = (-dtauH_dH(k, uH) - 0.1 * irBand_beta) * exptauC(k, uC) * exptauH_with_continuum(k, uH)
        else:
            v[k] = (-dtauH_dH(k, uH)) * exptauC(k, uC) * exptauH_with_continuum(k, uH)
    return v


#   IR absorption tensor for each band, between each levels
#
def cumul_exptau_tensor(vC, vH, f,
                        empty_val):  # input vC,vH vectors [0,nLevel-1]    returns tau[nLevel,nLevel,irBand_size] ATTENTION tau[i,j,k] undefined for j>i
    nLevel = vC.size
    tau = np.full([nLevel, nLevel, irBand_size], empty_val)
    for i in range(nLevel):  # i is [0,nLevel-1]
        sumC = 0.
        sumH = 0.
        for j in range(i, -1, -1):  # j is from i down to 0, -> [0,nLevel-1]
            sumC += vC[j]
            sumH += vH[j]
            # print("sumC = ",sumC)
            # print("sumH = ",sumH)
            tau[i, j, :] = f(sumC, sumH)
    return tau


def exptau_tensor(vC,
                  vH):  # input vC,vH vectors [0,nLevel-1]    returns tau[nLevel,nLevel,irBand_size] ATTENTION tau[i,j,k] undefined for j>i
    return cumul_exptau_tensor(vC, vH, exptau_vector, 1.0)


"""    nLevel = vC.size
    tau = np.ones([nLevel,nLevel,irBand_size])
    for i in range(nLevel):             #i is [0,nLevel-1]
        sumC = 0.
        sumH = 0.
        for j in range(i,-1,-1):        #j is from i down to 0, -> [0,nLevel-1]
            sumC += vC[j]
            sumH += vH[j]
            tau[i,j,:] = exptau_vector(sumC,sumH)
    return tau
"""


def dexptau_dc_tensor(vC, vH):
    return cumul_exptau_tensor(vC, vH, dexptau_dc_vector, 0.)


def dexptau_dh_tensor(vC, vH):
    return cumul_exptau_tensor(vC, vH, dexptau_dh_vector, 0.)


#   NEF (net exchange formulation) matrices C,D,B,L
#
def nef_C_mat(ip, tau):
    nLevPlus1 = ip.shape[0]  # = nLevel+1
    c = np.empty([nLevPlus1, nLevPlus1])
    for i in range(nLevPlus1):  # i = [0,nLevel]
        c[i, i] = 1.
    for i in range(1, nLevPlus1):  # i = [1,nLevel]
        for j in range(i):  # j = [0,i-1] ie [0,0] to [0,nLevel-1]
            c[i, j] = np.dot(tau[i - 1, j, :], ip[i, :])
    for i in range(nLevPlus1 - 1):  # i = [0,nLevel-1]
        for j in range(i + 1, nLevPlus1):  # j = [i+1,nLevel] ie [1,nLevel] to [nLevel,nLevel]
            c[i, j] = np.dot(tau[j - 1, i, :], ip[i, :])
    return c


def nef_D_mat(ip, tau, nC):
    nLevPlus1 = ip.shape[0]
    d = np.empty([nLevPlus1, nLevPlus1])
    for i in range(1, nLevPlus1):  # i = [1,nLevel]
        for j in range(i - 1):  # j = [0,i-2] ie [0,-1] to [0,nLevel-2]
            d[i, j] = np.dot(tau[i - 2, j, :], ip[i, :])
    for i in range(1, nLevPlus1):  # i = [1,nLevel]
        for j in range(i + 1, nLevPlus1):  # j = [i+1,nLevel]
            d[i, j] = np.dot(tau[j - 1, i - 1, :], ip[i, :])
    for j in range(nLevPlus1):  # j = [0,nLevel]
        d[0, j] = 0.
    for i in range(1, nLevPlus1):  # i = [1,nLevel]
        d[i, i - 1] = 1.
    for i in range(1, nLevPlus1):  # i = [1,nLevel]
        d[i, i] = nC[i, i - 1]
    return d


def nef_B_mat(nC, nD):
    nLevPlus1 = nC.shape[0]
    b = np.empty([nLevPlus1, nLevPlus1 + 1])
    for i in range(nLevPlus1):  # i = [0,nLevel]
        b[i, i] = 0.
    for i in range(1, nLevPlus1):  # i = [1,nLevel]
        for j in range(1, nLevPlus1):  # j = [1,nLevel]
            if i != j:
                b[i, j] = nC[i, j] + nD[i, j - 1] - nC[i, j - 1] - nD[i, j]
    for i in range(1, nLevPlus1):  # i = [1,nLevel]
        b[i, 0] = nC[i, 0] - nD[i, 0]
        b[0, i] = nC[0, i] - nC[0, i - 1]
    for i in range(1, nLevPlus1):  # i = [1,nLevel]
        b[i, nLevPlus1] = nD[i, nLevPlus1 - 1] - nC[i, nLevPlus1 - 1]
    b[0, nLevPlus1] = -nC[0, nLevPlus1 - 1]
    return b


def nef_L_mat(nB):
    nLevPlus1 = nB.shape[0]
    l = np.empty([nLevPlus1, nLevPlus1])
    for i in range(nLevPlus1):  # i = [0,nLevel]
        for j in range(nLevPlus1):  # j = [0,nLevel]
            l[i, j] = -nB[j, i]
    for i in range(nLevPlus1):  # i = [0,nLevel]
        s = 0.
        for k in range(nLevPlus1 + 1):  # k = [0,nLevel+1]
            s += nB[i, k]
        l[i, i] += s
    return l


def new_nef_B_mat(ip, tau, kc=0):
    nLevPlus1 = ip.shape[0]
    # print("ip = ", ip)
    # print("tau = ", tau)
    d = 0
    if kc == 0:
        d = 1
    noBlock = (kc <= 0)
    mc = np.zeros([nLevPlus1, nLevPlus1])
    for i in range(nLevPlus1):  # i = [0,nLevel]
        mc[i, i] = d  # j==i
        for j in range(i):  # j<i
            if noBlock or (i >= kc >= j + 1):
                mc[i, j] = np.dot(tau[i - 1, j], ip[i])
        for j in range(i + 1, nLevPlus1):  # j>i
            if noBlock or (j >= kc >= i + 1):
                mc[i, j] = np.dot(tau[j - 1, i], ip[i])

    # print("mc = ", mc)
    # print("mc: zero ?= ", nef_C_mat(ip,tau)-mc)

    md = np.zeros([nLevPlus1, nLevPlus1])
    for j in range(nLevPlus1):  # i = 0
        md[0, j] = 0
    for i in range(1, nLevPlus1):  # i = [1,nLevel]
        md[i, i - 1] = d  # j==i-1
        md[i, i] = mc[i, i - 1]  # j==i
        for j in range(i - 1):  # j<i-1
            if noBlock or (i - 1 >= kc >= j + 1):
                md[i, j] = np.dot(tau[i - 2, j], ip[i])
        for j in range(i + 1, nLevPlus1):  # j>i
            if noBlock or (j >= kc >= i):
                md[i, j] = np.dot(tau[j - 1, i - 1], ip[i])
    # print("md = ", md)
    # print("md: zero ?= ", nef_D_mat(ip,tau,mc)-md)

    nc = np.empty([nLevPlus1, nLevPlus1 + 1])
    for i in range(nLevPlus1):  # i = [0,nLevel]
        nc[i, 0] = mc[i, 0]
        nc[i, i] = 0
        nc[i, nLevPlus1] = -mc[i, nLevPlus1 - 1]
        for j in range(1, nLevPlus1):
            if i != j:
                nc[i, j] = mc[i, j] - mc[i, j - 1]

    nd = np.empty([nLevPlus1, nLevPlus1 + 1])
    for i in range(nLevPlus1):  # i = [0,nLevel]
        nd[i, 0] = md[i, 0]
        nd[i, i] = 0
        nd[i, nLevPlus1] = -md[i, nLevPlus1 - 1]
        for j in range(1, nLevPlus1):
            if i != j:
                nd[i, j] = md[i, j] - md[i, j - 1]

    return nc - nd


def dLW_dc(Tv, co2v, h2ov):  # derivative with respect to co2
    nLevPlus1 = Tv.shape[0]
    planckM = planck_integ_matrix(Tv)
    dexp_dc = dexptau_dc_tensor(co2v, h2ov)
    sigT4 = np.empty([nLevPlus1])
    for i in range(nLevPlus1):  # i = [0,nLevel]
        sigT4[i] = irSigma * Tv[i] ** 4
    dlw_dc = np.empty([nLevPlus1 - 1, nLevPlus1])
    for kc in range(nLevPlus1 - 1):
        nB_C = new_nef_B_mat(planckM, dexp_dc, kc + 1)
        dlw_dc[kc] = np.dot(nef_L_mat(nB_C), sigT4)
    return dlw_dc


def dLW_dh(Tv, co2v, h2ov):  # derivative with respect to h2O
    nLevPlus1 = Tv.shape[0]
    planckM = planck_integ_matrix(Tv)
    dexp_dh = dexptau_dh_tensor(co2v, h2ov)
    sigT4 = np.empty([nLevPlus1])
    for i in range(nLevPlus1):  # i = [0,nLevel]
        sigT4[i] = irSigma * Tv[i] ** 4
    dlw_dh = np.empty([nLevPlus1 - 1, nLevPlus1])
    for kc in range(nLevPlus1 - 1):
        nB_C = new_nef_B_mat(planckM, dexp_dh, kc + 1)
        dlw_dh[kc] = np.dot(nef_L_mat(nB_C), sigT4)
    return dlw_dh


#  final LW budget in W.m-2

def nefLW(Tv, co2v, h2ov):
    planckM = planck_integ_matrix(Tv)
    tauT = exptau_tensor(co2v, h2ov)

    # nefC = nef_C_mat(planckM,tauT)
    # nefD = nef_D_mat(planckM,tauT,nefC)
    # nefB = nef_B_mat(nefC,nefD)
    # return nef_L_mat(nefB)
    #
    #   nefB2 = new_nef_B_mat(planckM,tauT)
    #   print("nefB zero ?= ", nefB-nefB2)
    return nef_L_mat(new_nef_B_mat(planckM, tauT))


def bLW(Tv, co2v, h2ov):
    nefL = nefLW(Tv, co2v, h2ov)
    nLevPlus1 = Tv.shape[0]
    sigT4 = np.empty([nLevPlus1])
    for i in range(nLevPlus1):  # i = [0,nLevel]
        sigT4[i] = irSigma * Tv[i] ** 4
    return np.dot(nefL, sigT4)


#######################
#
#   SW model
#
#######################


def AxDown(v, c, f):
    n = v.shape[0]
    AxD = np.empty([n + 1])
    s = 0.
    for i in range(n - 1, -1, -1):  # i = n-1 down to 0
        s += v[i]
        AxD[i] = f(s * c)  # vdown(i) = Sum{ v(k), k>=i }*c	;	 AxD(i) = Awv( vdown(i) )
    AxD[n] = s * c  # AxD[-1] = AxD(n) = Sum{ v(k) }*c
    return AxD


def AxUp(v, c, z, f):
    n = v.shape[0]
    AxU = np.empty([n + 1])
    s = 0.
    AxU[0] = f(z)
    for i in range(n):  # i = 0 to n-1
        s += v[i]
        AxU[i + 1] = f(z + s * c)  # vup(i) = Sum{ h2O(k), k<=i }*c	;	 AxU(i) = Awv( z + vup(i) )
    return AxU


# remark:
#   dAxDown_dv( v, c, f ) ... starting from  AxDown( v, c, df )
#   dAxUp_dv( v, c, z, f ) ... AxUp( v, c, z, df )


#  ozone parameterization

def Aoz(x):
    return (0.02118 * x) / (1 + 0.042 * x + 0.000323 * x * x) + (1.082 * x) / pow((1 + 138.6 * x), 0.805) + (
            0.0658 * x) / (1 + (103.6 * x) * (103.6 * x) * (103.6 * x))


def Aozone(u, cosz):
    return Aoz(35. * u / np.sqrt(1224. * cosz * cosz + 1.))


def Ra(cosz):
    return 0.219 / (1 + 0.816 * cosz)


Rastar = 0.144


def Mozone(cosz):
    return 35. / np.sqrt(1224. * cosz * cosz + 1.)


Mbar = 1.9


def alboz(alb, cosz):
    return Ra(cosz) + (1 - Ra(cosz)) * alb * (1 - Rastar) / (1 - alb * Rastar)


def dalboz_da(alb, cosz):
    z = 1 / (1 - alb * Rastar)
    return (1 - Ra(cosz)) * (1 - Rastar) * z * z


def swscat(O3, cosz):
    return 0.353 + (0.647 - Rr(cosz) - Aozone(O3, cosz))


def sabar(O3, alb, cosz):
    return 0.353 + (swscat(O3, cosz) - 0.353) / (1 - Rrstar * alb)


def dsabar_da(O3, alb, cosz):
    z = 1 / (1 - Rrstar * alb)
    return Rrstar * (swscat(O3, cosz) - 0.353) * z * z


def sw_sum(a, v, c_down, c_up, f, sab=None):
    n = v.shape[0]
    sw = np.empty([n + 1])
    aDown = AxDown(v, c_down, f)
    aUp = AxUp(v, c_up, aDown[-1], f)
    aDown[-1] = 0.
    for i in range(n):  # i = 0 to n-1
        # sk = aDown[i]-aDown[i+1]
        # sks = aUp[i+1]-aUp[i]
        sw[i + 1] = ((aDown[i] - aDown[i + 1]) + a * (aUp[i + 1] - aUp[i]))
    if sab is None:
        sw[0] = 0.
    else:
        sw[0] = (1 - a) * (sab - aDown[0])
    return sw


def dsw_sum_dv(a, v, c_down, c_up, df, sab=None):  # CAREFUL: return a matrix !!
    n = v.shape[0]
    dsw = np.empty([n, n + 1])
    daDown = c_down * AxDown(v, c_down, df)
    daUp = AxUp(v, c_up, c_down * np.sum(v), df)
    # print("daDown = ",daDown)
    # print("daUp = ",daUp)
    c_plus = c_down + c_up
    for i in range(n):
        if sab is None:
            dsw[i, 0] = 0.
        else:
            dsw[i, 0] = (1 - a) * (-daDown[0])
        for j in range(i):  # j = 0 à i-1
            dsw[i, j + 1] = (daDown[j] - daDown[j + 1]) + a * c_down * (daUp[j + 1] - daUp[j])
        dsw[i, i + 1] = daDown[i] + a * (c_plus * daUp[i + 1] - c_down * daUp[i])
        for j in range(i + 1, n):  # j = i+1 à n
            dsw[i, j + 1] = a * c_plus * (daUp[j + 1] - daUp[j])

    # aDown[-1] = 0.
    # for i in range(n):                  #   i = 0 to n-1
    #    dsw[i+1] = ((aDown[i]-aDown[i+1]) + a*(aUp[i+1]-aUp[i]))

    return dsw


def swO3(eps, albo, o3v, cosZ):
    sw = sw_sum(albo, o3v, Mozone(cosZ), Mbar, Aoz)
    # sw[0] = 0.
    return eps * sw


def swO3_(eps, albo, o3v, cosZ):
    n = o3v.shape[0]
    sw = np.empty([n + 1])
    Moz = Mozone(cosZ)
    aDown = AxDown(o3v, Moz, Aoz)
    # aUp = AxUp( o3v, Mbar, aDown[o3v.shape[0]], Aoz )
    aUp = AxUp(o3v, Mbar, aDown[-1], Aoz)
    # aDown[o3v.shape[0]] = 0.
    aDown[-1] = 0.
    for i in range(n):  # i = 0 to n-1
        sk = aDown[i] - aDown[i + 1]
        sks = aUp[i + 1] - aUp[i]
        sw[i + 1] = eps * (sk + albo * sks)
    sw[0] = 0.
    return sw


#  water vapor parameterization

def Awv(u):
    return 2.9 * u / (pow(1 + 141.5 * u, 0.635) + 5.925 * u)


def dAwv(u):
    numerator = 2.9 + (2.9 * 141.5 * 0.365) * u
    z = 5.925 * u + pow(1 + 141.5 * u, 0.635)
    denominator = pow(1 + 141.5 * u, 0.365) * z * z
    return numerator / denominator


def Rr(cosz):
    return 0.28 / (1 + 6.43 * cosz)


Rrstar = 0.0685


def swH2O(eps, alb, h2ov, cosZ, sab):
    sw = sw_sum(alb, h2ov, 1.0 / cosZ, 5. / 3., Awv, sab)
    # sw[0] = (1-alb) * (sab - adown0)
    return eps * sw


def swH2O_(eps, alb, h2ov, cosZ, sab):
    n = h2ov.shape[0]
    sw = np.empty([n + 1])
    aDown = AxDown(h2ov, 1.0 / cosZ, Awv)
    # aUp = AxUp( h2ov, 5./3., aDown[h2ov.shape[0]], Awv )
    aUp = AxUp(h2ov, 5. / 3., aDown[-1], Awv)
    # aDown[h2ov.shape[0]] = 0.
    aDown[-1] = 0.
    for i in range(n):  # i = 0 to n-1
        sk = aDown[i] - aDown[i + 1]
        sks = aUp[i + 1] - aUp[i]
        sw[i + 1] = eps * (sk + alb * sks)
    sw[0] = eps * (1 - alb) * (sab - aDown[0])
    return sw


#  final SW budget

def bSW(eps_Sover4, alb, cosZ, h2ov, o3v):
    # sumO3 = np.sum( o3v )
    sab = sabar(np.sum(o3v), alb, cosZ)
    # swH = swH2O( eps_Sover4, alb, h2ov, cosZ, sab )
    # swH, adown0 = sw_sum( alb, h2ov, 1.0/cosZ, 5./3., Awv )
    # swH[0] = (1-alb) * (sab - adown0)
    swH = sw_sum(alb, h2ov, 1.0 / cosZ, 5. / 3., Awv, sab)

    albo = alboz(alb, cosZ)
    # swO = swO3( eps_Sover4, albo, o3v, cosZ )
    # swO, z =  sw_sum( albo, o3v, Mozone( cosZ ), Mbar, Aoz )
    # swO[0] = 0.
    swO = sw_sum(albo, o3v, Mozone(cosZ), Mbar, Aoz)

    return eps_Sover4 * (swH + swO)


def dbSW_dh(eps_Sover4, alb, cosZ, h2ov, _o3v):
    # sab = sabar(np.sum( o3v ),alb,cosZ)
    # swH = swH2O( eps_Sover4, alb, h2ov, cosZ, sab )
    dswH = dsw_sum_dv(alb, h2ov, 1.0 / cosZ, 5. / 3., dAwv, 0.)  # sab is not used, but variable is not None !!
    # dswH[0] = (1-alb) * (- dadown0)

    return eps_Sover4 * dswH


"""
## and the derivatives
def dbSW_da( eps_Sover4, alb, cosZ, h2ov, o3v ):
    sab = sabar(np.sum( o3v ),alb,cosZ)
    aDown = AxDown( h2ov, 1.0/cosZ, Awv )
    aUp = AxUp( h2ov, 5./3., aDown[-1], Awv )
    aUpO = AxUp( o3v, Mbar, aDown[-1], Aoz )
    
    #albo = alboz(alb,cosZ)
    #swO = swO3( eps_Sover4, albo, o3v, cosZ )
    return #swH+swO
"""

#######################
#
#   NEW CODE
#
#######################


Sover4 = cst.Sover4
Tref = cst.Tref


class Radiation:
    def __init__(self, n, profileIndex=1, linear=False, useRelativeH=True, albedo=0.1, cosZ=0.25, eps=1.00013951364302,
                 co2ppm=280, useScaling='full'):
        self.nb_levels = n  # number of atmospheric levels + the ground (= nb_atmo + 1)
        if linear:  # linear around the reference temperature profile
            self.useRH = False
            if useScaling == 'full':
                self.useScaling = 'fixed'  # = 'full', 'fixed', 'no'
        else:
            self.useRH = useRelativeH
            self.useScaling = useScaling  # = 'full', 'fixed', 'no'
        self.prof = prf.Profile(profileIndex, n - 1, p0=1013.25, useRH=self.useRH, useScalingC=useScaling,
                                useScalingHL=useScaling, useScalingHS=useScaling)
        # self.prof = prf.Profile(profileIndex,n-1,p0=1013.25,useRH=self.useRH,useScalingC=useScaling,
        # useScalingHL='full',useScalingHS=useScaling) self.prof = prf.Profile(profileIndex,n-1,p0=1013.25,
        # useRH=self.useRH,useScalingC=useScaling,useScalingHL=useScaling,useScalingHS='full')
        self.alb = albedo
        self.cosZ = cosZ
        self.eps = eps
        self.co2ppm = co2ppm
        if linear:
            self.linear = False
            self.linearR, self.linearR0, b = self.ddx_bilanR(Tref / self.prof.temperatureRef)
        self.linear = linear

    def bilanSW_in_Wm2(self, _T_kelvin):
        # print("h2oSW = ",self.prof.h2o_SW)
        return bSW(self.eps * Sover4, self.alb, self.cosZ, self.prof.h2o_SW, self.prof.o3_SW)

    def bilanLW_in_Wm2(self, T_kelvin):
        # print("h2oLW = ",self.prof.h2o_LW)
        # print("co2LW = ",self.co2ppm*self.prof.co2_LW)
        return bLW(T_kelvin, self.co2ppm * self.prof.co2_LW, self.prof.h2o_LW)

    def bilanR_in_Wm2(self, x):
        tempV = Tref / x
        self.prof.set_temperature(tempV[1:])
        bilanLW = self.bilanLW_in_Wm2(tempV)
        bilanSW = self.bilanSW_in_Wm2(tempV)
        # print("bilanLW = ",bilanLW)
        # print("bilanSW = ",bilanSW)
        return bilanLW + bilanSW

    def bilanR(self, x):
        if self.linear:
            return self.linearR0 + np.dot(self.linearR, x)
        return self.bilanR_in_Wm2(x) / Sover4

    def ddx_bilanR(self, x):
        if self.linear:
            return self.linearR, self.linearR0, self.linearR0 + np.dot(self.linearR, x)
        tempV = Tref / x
        self.prof.set_temperature(tempV[1:])
        nefL = nefLW(tempV, self.co2ppm * self.prof.co2_LW, self.prof.h2o_LW)
        n1 = tempV.shape[0]
        sigT4 = np.empty([n1])
        for i in range(n1):  # i = [0,nLevel]
            sigT4[i] = irSigma * tempV[i] ** 4
        bilanLW = np.dot(nefL, sigT4)
        bilanSW = bSW(self.eps * Sover4, self.alb, self.cosZ, self.prof.h2o_SW, self.prof.o3_SW)
        # linear case: LW = nef.sigT4  et  SW = cte
        #    dLW = nef.4sigT3.(-Tref/x2)
        #    d(T4)/dx = 4T3.(-Tref/x2)  = 4(Tref/x)^3.(-Tref/x2) = -4(Tref)^4/x5
        ddx = -4 * irSigma * (Tref ** 4) * pow(x, -5)
        r = np.dot(nefL, np.diag(ddx))
        # print("r1 = ",r/Sover4)
        if not self.linear:
            dplanckM = planck_Dinteg_matrix(tempV)
            tauT = exptau_tensor(self.co2ppm * self.prof.co2_LW, self.prof.h2o_LW)
            dnefLW = nef_L_mat(new_nef_B_mat(dplanckM, tauT, -1))
            ddxx = sigT4 * (-Tref / (x * x))
            r = r + np.dot(dnefLW, np.diag(ddxx))
            # print("r2 = ",np.dot( dnefLW, np.diag(ddxx) ) )
            # print("r2 = ",r/Sover4)
            if self.prof.use_scaled_co2_LW == 'full':
                dlw_dc = dLW_dc(tempV, self.co2ppm * self.prof.co2_LW, self.prof.h2o_LW)
                # print("dlw_dc = ",dlw_dc)
                # print("dlw_dc = ",dlw_dc/Sover4)
                rc0 = np.dot(np.diag(8 * self.co2ppm * self.prof.co2_LW / x[1:]), dlw_dc)
                # print("cl = ",self.co2ppm*self.prof.co2_LW)
                # print("t = ",x[1:])
                # print("diag = ",8*self.co2ppm*self.prof.co2_LW/x[1:])
                rc = np.transpose(np.vstack([np.zeros([n1]), rc0]))
                # print("rc = ",rc/Sover4)
                # print("rc = ",rc)
                r = r + rc
                # print("r3 = ",r/Sover4)

            if self.prof.use_scaled_h2o_LW == 'full' or self.useRH:
                dlw_dh = dLW_dh(tempV, self.co2ppm * self.prof.co2_LW, self.prof.h2o_LW)
                dh_dx = np.zeros([n1 - 1])
                if self.prof.use_scaled_h2o_LW == 'full':
                    dh_dx += 0.45 * self.prof.h2o_LW / x[1:]
                if self.useRH:
                    dh_dx += self.prof.h2o_LW * (-Tref / x[1:] / x[1:]) * phy.dlogew_dT_vec(tempV[1:])
                rc1 = np.dot(np.diag(dh_dx), dlw_dh)
                rc = np.transpose(np.vstack([np.zeros([n1]), rc1]))
                r = r + rc
                # print("dLW = ",r/Sover4)

            if self.prof.use_scaled_h2o_SW == 'full' or self.useRH:
                dsw_dh = dbSW_dh(self.eps * Sover4, self.alb, self.cosZ, self.prof.h2o_SW, self.prof.o3_SW)
                # print("dsw_dh = ",dsw_dh)
                dh_dx = np.zeros([n1 - 1])
                if self.prof.use_scaled_h2o_SW == 'full':
                    dh_dx += 0.45 * self.prof.h2o_SW / x[1:]
                if self.useRH:
                    dh_dx += self.prof.h2o_SW * (-Tref / x[1:] / x[1:]) * phy.dlogew_dT_vec(tempV[1:])
                rc2 = np.dot(np.diag(dh_dx), dsw_dh)
                rc = np.transpose(np.vstack([np.zeros([n1]), rc2]))
                # print("dh_dx = ",dh_dx)
                # print("dh_dT = ",-(x*x/Tref)[1:]*dh_dx)
                # print("dSW = ",rc/Sover4)
                r = r + rc

        r = r / Sover4
        b = (bilanLW + bilanSW) / Sover4
        r0 = b - np.dot(r, x)
        return r, r0, b  # bilan = b = r.x + r0 around x

    def minus_entropy(self, x):
        return np.dot(x, self.bilanR(x))

    def minus_entropy_with_jac(self, x):
        r, r0, b = self.ddx_bilanR(x)
        return np.dot(x, b), np.dot(r + np.transpose(r), x) + r0

    def bilanColumn(self, x):  # sum of bilanR, = 0 for a simple column
        return np.sum(self.bilanR(x))

    def unconstrained_MEP_sol(self):
        ones = np.ones(self.nb_levels)
        x0 = ones
        x = 2 * x0
        while (np.linalg.norm(x - x0) / self.nb_levels) > 1.0e-11:
            r, r0, b = self.ddx_bilanR(x0)
            sr = -np.dot(ones, r)
            m = np.append(r, np.reshape(sr, (1, self.nb_levels)),
                          axis=0)  # np.append(np.append(r, sr, axis=1),np.zeros(n+2),axis=0)
            m = np.append(m, np.zeros((self.nb_levels + 1, 1)), axis=1)
            m = m + np.transpose(m)
            b = np.append(-r0, np.sum(r0))
            sol = np.linalg.solve(m, b)
            x = x0
            x0 = sol[:-1]
            # print("x = ",x0)
        return x0


#     END     #
if False:  # compare python and C++ radiative code
    from radiatif2 import Rad
    import time

    radiaC = Rad(4, 1, 1013.25, 280.0)
    radiaPy = Radiation(5, useRelativeH=True, useScaling='full')

    x0 = np.array([.8, .9, 1., 1.1, 1.2])

    tic = time.perf_counter()
    r, r0, b = radiaPy.ddx_bilanR(x0)
    toc = time.perf_counter()
    print("b = ", b)
    print("r  = ", r)
    print("r0 = ", r0)
    print("computing time = ", toc - tic, " seconds")

    # b = radiaC.bilanR(x0)
    tic = time.perf_counter()
    b, db = radiaC.ddx_bilanR(x0)
    toc = time.perf_counter()
    print("b = ", b)
    print("r  = ", db)
    print("r0 = ", b - np.dot(db, x0))
    print("computing time = ", toc - tic, " seconds")
