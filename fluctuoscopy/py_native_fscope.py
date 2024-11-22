import math
import time
from math import pi, sinh

from scipy.special import digamma, polygamma

c1 = 0.405284734569351085775517852838911 # 4/pi^2
c2 = -1.96351002602142347944097633299876 # Psi(1/2)
c3 = 0.63661977236758134307553505349006 # 2/pi
c4 = 1.2020569031595942853997381615114 # zeta(3)
c5 = 0.202642367284675542887758926419455 # 2/pi^2
c6 = 0.693147180559945309417232121458177 # ln(2)
c7 = 0.2807297417834425849120716073954404 # 1/(2\gamma_E)
c8 = 1.270362845461478170023744211540578999118 # -ln(2)-Psi(1/2)=-c6-c2
c_MTi = 15.503138340149910087738157533551 # pi^3/2 (factor 2 is in the integral)
c_MT = 0.010265982254684335189152783267119  # 1/pi^4
c_DOS = 0.129006137732797956737688210754255 # 4/pi^3
c_CC = 0.00138688196439446972811978351321894  # 4/pi^6/3
s_NC = 0.032251534433199489184422052688564 # 1/pi^3
s_IV = -0.032251534433199489184422052688564 # -1/pi^3
c_NMRi = 31.006276680299820175476315067101 # pi^3
c_NMRi2 = 12.566370614359172953850573533118 # 4pi
c_NMR = 0.037829191583088762117100356213279  # 1/(Zeta[3]*7*\[Pi])
hc20 = 0.69267287375563603674263246549077793763519897 #=pi^2*exp[-gamma_e]/8
INT0 = 10e-10 # integral cutoff at z=0
DX = 10e-6
PX = 10e-4 # integrable pole cut-out
GLx =[
    -0.90617984593866399280,
    -0.53846931010568309104,
    0,
    0.53846931010568309104,
    0.90617984593866399280,
]
GLw = [
    0.23692688505618908751,
    0.47862867049936646804,
    0.56888888888888888889,
    0.47862867049936646804,
    0.23692688505618908751,
]
# coefficients for exp
XA =  [
    2.7182818284590452354,
    1.648721270700128,
    1.284025416687742,
    1.133148453066826,
    1.064494458917859,
    1.031743407499103,
    1.015747708586686,
    1.007843097206448,
    1.003913889338348,
    1.001955033591003,
]
# coefficients for Gamma
XG =  [
    0,
    8.333333333333333e-02,
    -2.777777777777778e-03,
    7.936507936507937e-04,
    -5.952380952380952e-04,
    8.417508417508418e-04,
    -1.917526917526918e-03,
    6.410256410256410e-03,
    -2.955065359477124e-02,
    1.796443723688307e-01,
    -1.39243221690590,
]
# and for real log(gamma)
XGL = [
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5,
]
#coefficients for Psi
XP = [
    0,
    -0.83333333333333333e-01,
    0.83333333333333333e-02,
    -0.39682539682539683e-02,
    0.41666666666666667e-02,
    -0.75757575757575758e-02,
    0.21092796092796093e-01,
    -0.83333333333333333e-01,
    0.4432598039215686,
]
#coefficients for Bernoulli number
XBN = [
    20.718442527395005639466798045864127459653576742641,   #//4 pi exp(1/2)
    2.837877066409345483560659472811235287590943673502,    #//log(2 pi e)
    0.083333333333333333333333333333333333333333333333,    #//1/12
    0.00347222222222222222222222222222222222222222222,     #//1/288
    -0.00268132716049382716049382716049382716049382716,     #//-139/51840
    -0.000229472093621399176954732510288065843621399177,    #//-571/2488320
    0.000784,                                              #//49/(2*31250)
]

#the "first" 32 non-zero Bernoulli numbers (B_2 to B_64)  [B_0 & B_1 are not included]
B32 =  [
    0.16666666666666666666666666666667,
    -0.033333333333333333333333333333333,
    0.023809523809523809523809523809524,
    -0.033333333333333333333333333333333,
    0.075757575757575757575757575757576,
    -0.25311355311355311355311355311355,
    1.1666666666666666666666666666667,
    -7.0921568627450980392156862745098,
    54.971177944862155388471177944862,
    -529.12424242424242424242424242424,
    6192.1231884057971014492753623188,
    -86580.253113553113553113553113553,
    1.4255171666666666666666666666667e6,
    -2.7298231067816091954022988505747e7,
    6.0158087390064236838430386817484e8,
    -1.5116315767092156862745098039216e10,
    4.2961464306116666666666666666667e11,
    -1.3711655205088332772159087948562e13,
    4.8833231897359316666666666666667e14,
    -1.9296579341940068148632668144863e16,
    8.4169304757368261500055370985604e17,
    -4.0338071854059455413076811594203e19,
    2.1150748638081991605601453900709e21,
    -1.2086626522296525934602731193708e23,
    7.5008667460769643668557200757576e24,
    -5.0387781014810689141378930305220e26,
    3.6528776484818123335110430842971e28,
    -2.8498769302450882226269146432911e30,
    2.3865427499683627644645981919219e32,
    -2.1399949257225333665810744765191e34,
    2.0500975723478097569921733095672e36,
    -2.0938005911346378409095185290028e38,
]

def hc2(x):
    Nbs = 32  # <10^-9 error
    if x >= 1.0:
        return 0.0
    if x < 1e-6:
        return hc20
    # Solve log(t) + Psi(1/2 + 2/pi^2 * h/t) - Psi(1/2) = 0 -> bisection
    c = math.log(x) - c2
    h2 = 1 - x
    h1 = hc20 * h2
    for i in range(Nbs):
        hm = 0.5 * (h1 + h2)
        fm = c + digamma(0.5 + c5 * hm / x)
        if fm < 0.0:
            h1 = hm
        else:
            h2 = hm
    return 0.5 * (h1 + h2)

def calcFC(t,h,Tct,Tctp):
    if h < hc2(t):
        return 1, 0, 0, 0, 0, 0, 0
    sAL, sMTsum, sMTint, sDOS, sCC, res = MC_sigma(t, h, Tct, Tctp)
    k = 1 + res
    MC = sAL + sMTsum + sMTint + sDOS + sCC
    return t, h, k, sAL, sMTsum, sMTint, sDOS, sCC, MC

def MC_sigma(t, h, Tctau, Tctauphi):
    if t < 0.0001:
        return 0, 0, 0, 0, 0, 0
    h = max(h, 0.0001) # This is some hack that was missing entirely in the original code, would just return 0

    Z = 5.0
    KZ = 200
    K = 1000

    sAL = 0.0
    sMTint = 0.0
    sMTsum = 0.0
    sDOS = 0.0
    sCC = 0.0

    M = int(1.0 / (t * Tctau) + 0.5)
    gphi = 0.125 * math.pi / (t * Tctauphi)
    res = 0

    for m in range(M, -1, -1):
        rs, sumMT, sumCC = MCksum(m, t, h, K)
        if rs > res:
            res = rs
        iAL, iMT, iDOS, iCC = MCint(m, t, h, Z, KZ)
        sAL += (1.0 + m) * iAL
        sMTint += c_MTi / (gphi + 2 * h / t * (m + 0.5)) * iMT
        sMTsum += sumMT
        sDOS += iDOS
        sCC += (m + 0.5) * sumCC

    sigAL = sAL / math.pi
    sigMTint = c_MT * sMTint * h / t
    sigMTsum = c_MT * sMTsum * h / t
    sigDOS = c_DOS * sDOS * h / t
    sigCC = c_CC * sCC * h * h / (t * t)

    return sigAL, sigMTsum, sigMTint, sigDOS, sigCC, res

def MCksum(n, t, h, kmax):
    MT = 0.0
    CC = 0.0
    res = 0
    if t < 1e-6:
        t = 1e-6
        res = 2
    xn = c1 * (n + 0.5) * h / t
    kM = 2000 - int(2 * xn)
    if kM < 2:
        kM = 2

    # Now do the integration
    Am = math.log(h * (n + 0.5)) - c2 - c6
    zmax = 1.0 / (2 * c1 + t * kM / (h * (n + 0.5)))
    Nz = 25  # Functions are smooth enough and zmax < 1.2
    dz = zmax / Nz
    dzh = 0.5 * dz

    for k in range(Nz):
        for j in range(5):
            z = k * dz + dzh * (1.0 + GLx[j])
            x = 1.0 / (Am - math.log(z))
            CC += GLw[j] * x * z
            MT += GLw[j] * x

    x = -t / (h * (n + 0.5))
    CC = x * x * dzh * CC
    MT = x * dzh * MT

    if kM > 2:  # Need summation
        # Shift k by one, then sum from 2 to kmax
        Am = math.log(t) - c2
        for k in range(kM - 1, 1, -1):
            x = 0.5 * k + xn
            en = Am + digamma(x)
            en2 = polygamma(2, x)
            en3 = polygamma(3, x)
            MT += en2 / en
            CC += en3 / en

    # k=0 term
    x = 0.5 + xn
    en = math.log(t) - c2 + digamma(x)
    en2 = polygamma(2, x)
    en3 = polygamma(3, x)
    MT = 2 * MT + en2 / en
    CC = 2 * CC + en3 / en

    return res, MT, CC

def MCint(n, t, h, zmax, zsteps):
    dz = zmax / zsteps
    dzh = 0.5 * dz
    s_al = 0.0
    s_mt = 0.0
    s_dos = 0.0
    s_cc = 0.0

    for i in range(-zsteps, zsteps):
        for j in range(5):
            z = i * dz + dzh * (1.0 + GLx[j])
            al, mt, dos, cc = MCfunc(n, t, h, z)
            s_al += GLw[j] * al
            s_mt += GLw[j] * mt
            s_dos += GLw[j] * dos
            # s_cc += GLw[j] * cc

    AL = dzh * s_al
    MT = 2 * dzh * s_mt
    DOS = dzh * s_dos
    CC = 0  # 4 * dzh * s_cc

    return AL, MT, DOS, CC

def MCfunc(n, t, h, z):
    if abs(z) < INT0:
        z = INT0 if z >= 0.0 else -INT0

    enr, eni = E_n(n, t, h, z)
    en1r, en1i = E_n(n + 1, t, h, z)

    px, psip = CPSI(0.5 + c1 * h / t * (n + 0.5) + DX, 0.5 * z)
    px, psim = CPSI(0.5 + c1 * h / t * (n + 0.5) - DX, 0.5 * z)

    imen1 = 0.25 * (psip - psim) / DX  # Im E_n'
    imen2 = 0.25 * (psip + psim - eni - eni) / DX / DX  # Im E_n''

    absn = enr * enr + eni * eni
    absn1 = en1r * en1r + en1i * en1i
    D = sinh(pi * z)
    D = 1 / (D * D * absn)

    dr = enr - en1r
    di = eni - en1i
    res = (dr * dr - di * di) * eni * en1i - dr * di * (eni * en1r + en1i * enr)

    AL = res * D / absn1
    MT = eni * eni * D
    DOS = eni * imen1 * D
    CC = 0  # eni * imen2 * D

    return AL, MT, DOS, CC

def CPSI(X, Y):
    X1 = X
    Y1 = Y

    if Y == 0.0 and X == int(X) and X <= 0.0:
        PSR = 1e+200
        PSI = 0.0
    else:
        if X < 0.0:
            X = -X
            Y = -Y
        X0 = X
        if X < 8.0:
            N = 8 - int(X)
            X0 = X + 1.0 * N
        if X0 == 0.0 and Y != 0.0:
            TH = 0.5 * pi
        if X0 != 0.0:
            TH = math.atan(Y / X0)
        Z2 = X0 * X0 + Y * Y
        Z0 = math.sqrt(Z2)
        PSR = math.log(Z0) - 0.5 * X0 / Z2
        PSI = TH + 0.5 * Y / Z2
        for K in range(1, 9):
            PSR = PSR + XP[K] * pow(Z2, -K) * math.cos(2.0 * K * TH)
            PSI = PSI - XP[K] * pow(Z2, -K) * math.sin(2.0 * K * TH)
        if X < 8.0:
            RR = 0.0
            RI = 0.0
            for K in range(1, N + 1):
                RR = RR + (X0 - K) / (math.pow(X0 - K, 2) + Y * Y)
                RI = RI + Y / (math.pow(X0 - K, 2) + Y * Y)
            PSR = PSR - RR
            PSI = PSI + RI
        if X1 < 0.0:
            TN = math.sin(pi * X) / math.cos(pi * X)
            TM = math.tanh(pi * Y)
            CT2 = TN * TN + TM * TM
            PSR = PSR + X / (X * X + Y * Y) + pi * (TN - TN * TM * TM) / CT2
            PSI = PSI - Y / (X * X + Y * Y) - pi * TM * (1.0 + TN * TN) / CT2
            X = X1
            Y = Y1

    return PSR, PSI

def E_n(n, t, h, z):
    re, im = CPSI(0.5 + c1 * h / t * (n + 0.5), 0.5 * z)
    re = math.log(t) + re - c2
    return re, im

if __name__ == "__main__":
    # Check run time and if the results are the same as FSCOPE in shell
    t0 = time.time()
    results = calcFC(1.1,0.01,0.01,0.01)
    print(f"Run time for one point: {time.time()-t0}")
    fscope_truth = [
        1.1,
        0.01,
        1,
        0.5022763931191433,
        -0.21279555421499902,
        0.005515951340560925,
        -0.11883226208615542,
        0.022010791857737896,
        0.19817532001628768,
        6.327609300613403,
    ]
    for i, res in enumerate(results):
        success = abs(res - fscope_truth[i]) < 1e-6
        print(f"Result {i} {"PASS" if success else "FAIL"}: {res} Ground truth: {fscope_truth[i]}")

