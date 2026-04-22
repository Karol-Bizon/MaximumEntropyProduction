"""
Radiative Code "Herbert & Paillard" (2011-2013)
described in:
  Herbert C., Paillard D., Dubrulle B. J. Clim. 2013
      supplementary material : https://dx.doi.org/10.1175/JCLI-D-13-00060.s1

D. Paillard november 2017
didier.paillard@lsce.ipsl.fr
"""

import numpy as np
import physics as phy
import constants as cst


# import radiatif as rad


#################################


########################################
#
#   Standard Profiles McClatchey 1972
#
########################################

class McClatcheyProfileTropical:
    pressure = np.array(
        [1013, 904, 805, 713, 633, 559, 492, 432, 378, 329, 286, 247, 213, 182, 156, 132, 111, 93.7, 78.9, 66.6, 56.5,
         48, 40.9, 35, 30, 25.7, 12.2, 6., 3.05, 1.59, 0.854, 0.0579, 0.0003])
    temperature = np.array(
        [300, 294, 288, 284, 277, 270, 264, 257, 250, 244, 237, 230, 224, 217, 210, 204, 197, 195, 199, 203, 207, 211,
         215, 217, 219, 221, 232, 243, 254, 265, 270, 219, 210])
    density = np.array(
        [1167, 1064., 968.9, 875.6, 795.1, 719.9, 650.1, 585.5, 525.8, 470.8, 420.2, 374, 331.6, 292.9, 257.8, 226,
         197.2, 167.6, 138.2, 114.5, 95.15, 79.38, 66.45, 56.18, 47.63, 40.45, 18.31, 8.6, 4.181, 2.097, 1.101, 0.0921,
         0.0005])
    wvdensity = np.array(
        [19., 13., 9.3, 4.7, 2.2, 1.5, 0.85, 0.47, 0.25, 0.12, 0.05, 0.017, 0.006, 0.0018, 0.001, 0.00076, 0.00064,
         0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011, 0.000043,
         0.000019, 0.0000063, 0.00000014, 0.000000001])
    o3density = np.array(
        [0.000056, 0.000056, 0.000054, 0.000051, 0.000047, 0.000045, 0.000043, 0.000041, 0.000039, 0.000039, 0.000039,
         0.000041, 0.000043, 0.000045, 0.000045, 0.000047, 0.000047, 0.000069, 0.00009, 0.00014, 0.00019, 0.00024,
         0.00028, 0.00032, 0.00034, 0.00034, 0.00024, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086,
         0.000000000043])


class McClatcheyProfileMidLatitudeSummer:
    pressure = np.array(
        [1013, 902, 802, 710, 628, 554, 487, 426, 372, 324, 281, 243, 209, 179, 153, 130, 111, 95, 81.2, 69.5, 59.5, 51,
         43.7, 37.6, 32.2, 27.7, 13.2, 6.52, 3.33, 1.76, 0.951, 0.0671, 0.0003])
    temperature = np.array(
        [294, 290, 285, 279, 273, 267, 261, 255, 248, 242, 235, 229, 222, 216, 216, 216, 216, 216, 216, 217, 218, 219,
         220, 222, 223, 224, 234, 245, 258, 270, 276, 218, 210])
    density = np.array(
        [1191, 1080, 975.7, 884.6, 799.8, 721.1, 648.7, 583, 522.5, 466.9, 415.9, 369.3, 326.9, 288.2, 246.4, 210.4,
         179.7, 153.5, 130.5, 111, 94.53, 80.56, 68.72, 58.67, 50.14, 42.88, 13.22, 6.519, 3.33, 1.757, 0.9512, 0.06706,
         0.0005])
    wvdensity = np.array(
        [14, 9.3, 5.9, 3.3, 1.9, 1.0, 0.61, 0.37, 0.21, 0.12, 0.064, 0.022, 0.006, 0.0018, 0.001, 0.00076, 0.00064,
         0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011, 0.000043,
         0.000019, 0.0000063, 0.00000014, 0.000000001])
    o3density = np.array(
        [0.00006, 0.00006, 0.00006, 0.000062, 0.000064, 0.000066, 0.000069, 0.000075, 0.000079, 0.000086, 0.00009,
         0.00011, 0.00012, 0.00015, 0.00018, 0.00019, 0.00021, 0.00024, 0.00028, 0.00032, 0.00034, 0.00036, 0.00036,
         0.00034, 0.00032, 0.0003, 0.0002, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086, 0.000000000043])


class McClatcheyProfileMidLatitudeWinter:
    pressure = np.array(
        [1018, 897.3, 789.7, 693.8, 608.1, 531.3, 462.7, 401.6, 347.3, 299.2, 256.8, 219.9, 188.2, 161, 137.8, 117.8,
         100.7, 86.1, 73.5, 62.8, 53.7, 45.8, 39.1, 33.4, 28.6, 24.3, 11.1, 5.18, 2.53, 1.29, 0.682, 0.0467, 0.0003])
    temperature = np.array(
        [272.2, 268.7, 265.2, 261.7, 255.7, 249.7, 243.7, 237.7, 231.7, 225.7, 219.7, 219.2, 218.7, 218.2, 217.7, 217.2,
         216.7, 216.2, 215.7, 215.2, 215.2, 215.2, 215.2, 215.2, 215.2, 215.2, 217.4, 227.8, 243.2, 258.5, 265.7, 230.7,
         210.2])
    density = np.array(
        [1301, 1162, 1037, 923, 828.2, 741.1, 661.4, 588.6, 522.2, 461.9, 407.2, 349.6, 299.9, 257.2, 220.6, 189, 182,
         138.8, 118.8, 101.7, 86.9, 74.21, 63.38, 54.15, 46.24, 39.5, 17.83, 7.924, 3.625, 1.741, 0.8954, 0.07051,
         0.0005])
    wvdensity = np.array(
        [3.5, 2.5, 1.8, 1.2, 0.66, 0.38, 0.21, 0.085, 0.035, 0.016, 0.0075, 0.0069, 0.006, 0.0018, 0.001, 0.00076,
         0.00064, 0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011,
         0.000043, 0.000019, 0.0000063, 0.00000014, 0.000000001])
    o3density = np.array(
        [0.00006, 0.000054, 0.000049, 0.000049, 0.000049, 0.000058, 0.000064, 0.000077, 0.00009, 0.00012, 0.00016,
         0.00021, 0.00026, 0.0003, 0.00032, 0.00034, 0.00036, 0.00039, 0.00041, 0.00043, 0.00045, 0.00043, 0.00043,
         0.00039, 0.00036, 0.00034, 0.00019, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086, 0.000000000043])


class McClatcheyProfileSubArcticSummer:
    pressure = np.array(
        [1010, 896, 792.9, 700, 616, 541, 473, 413, 359, 310.7, 267.7, 230, 197.7, 170, 146, 125, 108, 92.8, 79.8, 68.6,
         58.9, 50.7, 43.6, 37.5, 32.27, 27.8, 13.4, 6.61, 3.4, 1.81, 0.987, 0.0707, 0.0003])
    temperature = np.array(
        [287, 282, 276, 271, 266, 260, 253, 246, 239, 232, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225,
         225, 225, 226, 228, 235, 247, 262, 274, 277, 216, 210])
    density = np.array(
        [1220, 1110, 997.1, 898.5, 807.7, 724.4, 651.9, 584.9, 523.1, 466.3, 414.2, 355.9, 305.9, 263, 226, 194.3,
         167.1, 143.6, 123.5, 106.2, 91.28, 78.49, 67.5, 58.05, 49.63, 42.47, 13.38, 6.614, 3.404, 1.817, 0.9868,
         0.07071, 0.0005])
    wvdensity = np.array(
        [9.1, 6, 4.2, 2.7, 1.7, 1, 0.54, 0.29, 0.013, 0.042, 0.015, 0.0094, 0.006, 0.0018, 0.001, 0.00076, 0.00064,
         0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011, 0.000043,
         0.000019, 0.0000063, 0.00000014, 0.000000001])
    o3density = np.array(
        [0.000049, 0.000054, 0.000056, 0.000058, 0.00006, 0.000064, 0.000071, 0.000075, 0.000079, 0.00011, 0.00013,
         0.00018, 0.00021, 0.00026, 0.00028, 0.00032, 0.00034, 0.00039, 0.0004, 0.00041, 0.00039, 0.00036, 0.00032,
         0.0003, 0.00028, 0.00026, 0.00014, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086, 0.000000000043])


class McClatcheyProfileSubArcticWinter:
    pressure = np.array(
        [1013, 887.8, 777.5, 679.8, 593.2, 515.8, 446.7, 385.3, 330.8, 282.9, 241.8, 206.7, 176.6, 151, 129.1, 110.3,
         94.31, 80.58, 68.82, 58.75, 50.14, 42.77, 36.47, 31.09, 26.49, 22.56, 10.2, 4.701, 2.243, 1.113, 0.5719,
         0.04016, 0.0003])
    temperature = np.array(
        [257.1, 259.1, 255.9, 252.7, 247.7, 240.9, 234.1, 227.3, 220.6, 217.2, 217.2, 217.2, 217.2, 217.2, 217.2, 217.2,
         216.6, 216, 215.4, 214.8, 214.1, 213.6, 213, 212.4, 211.8, 211.2, 216., 222.2, 234.7, 247, 259.3, 245.7, 210])
    density = np.array(
        [1372, 1193, 1058, 936.6, 833.9, 745.7, 664.6, 590.4, 522.6, 453.8, 387.9, 331.5, 283.4, 242.2, 207.1, 177,
         151.7, 130, 111.3, 95.29, 81.55, 69.76, 59.66, 51, 43.58, 37.22, 16.45, 7.368, 3.33, 1.569, 0.7682, 0.05695,
         0.0005])
    wvdensity = np.array(
        [1.2, 1.2, 0.94, 0.68, 0.41, 0.2, 0.098, 0.054, 0.011, 0.0084, 0.0055, 0.0038, 0.0026, 0.001, 0.001, 0.00076,
         0.00064, 0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011,
         0.000043, 0.000019, 0.0000063, 0.00000014, 0.000000001])
    o3density = np.array(
        [0.000041, 0.000041, 0.000041, 0.000043, 0.000045, 0.000047, 0.000049, 0.000071, 0.00009, 0.00015, 0.00024,
         0.00032, 0.00043, 0.00047, 0.00049, 0.00056, 0.00062, 0.00062, 0.00062, 0.0006, 0.00056, 0.00051, 0.00047,
         0.00043, 0.00036, 0.00032, 0.00015, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086, 0.000000000043])


class McClatcheyProfileMidLatitude:
    def __init__(self):
        ps = McClatcheyProfileMidLatitudeSummer()
        pw = McClatcheyProfileMidLatitudeWinter()
        self.pressure = 0.5 * (ps.pressure + pw.pressure)
        self.temperature = 0.5 * (ps.temperature + pw.temperature)
        self.density = 0.5 * (ps.density + pw.density)
        self.wvdensity = 0.5 * (ps.wvdensity + pw.wvdensity)
        self.o3density = 0.5 * (ps.o3density + pw.o3density)


class McClatcheyProfileSubArctic:
    def __init__(self):
        ps = McClatcheyProfileSubArcticSummer()
        pw = McClatcheyProfileSubArcticWinter()
        self.pressure = 0.5 * (ps.pressure + pw.pressure)
        self.temperature = 0.5 * (ps.temperature + pw.temperature)
        self.density = 0.5 * (ps.density + pw.density)
        self.wvdensity = 0.5 * (ps.wvdensity + pw.wvdensity)
        self.o3density = 0.5 * (ps.o3density + pw.o3density)


class StdProfile:
    def __init__(self, i):
        if i == 1:
            self.ref_profile = McClatcheyProfileTropical()
        elif i == 2:
            self.ref_profile = McClatcheyProfileMidLatitudeSummer()
        elif i == 3:
            self.ref_profile = McClatcheyProfileMidLatitudeWinter()
        elif i == 4:
            self.ref_profile = McClatcheyProfileSubArcticSummer()
        elif i == 5:
            self.ref_profile = McClatcheyProfileSubArcticWinter()
        elif i == 6:
            self.ref_profile = McClatcheyProfileMidLatitude()
        elif i == 7:
            self.ref_profile = McClatcheyProfileSubArctic()

        # basic profiles tabulated above
        self.pressure = self.ref_profile.pressure
        self.temperature = self.ref_profile.temperature
        self.density = self.ref_profile.density
        self.wvdensity = self.ref_profile.wvdensity
        self.o3density = self.ref_profile.o3density

        # useful profiles deducted from
        c = (10. / cst.g / cst.massVolO3)
        self.o3_SW = c * np.divide(self.o3density, self.density)

        c = (10. / cst.g) * 1.e-6 * 100. * (cst.massMolCO2 / cst.Rgaz)  # (10./g) * (1ppm) * 100 * Mco2/Rgp
        self.co2 = c * self.pressure / np.multiply(self.temperature, self.density)
        # actually "self.co2" is (should be) a constant !!
        #  = (10/cst.g)*(cst.massMolCO2/cst.massMolAir)*1.0e-6
        self.co2_LW = self.co2 * pow(self.pressure / 1013., 1.75) * pow(273. / self.temperature, 8)

        c = (10. / cst.g)
        self.humidity = c * np.divide(self.wvdensity, self.density)
        self.h2o_LW = self.humidity * pow(self.pressure / 1013., 0.75) * pow(273. / self.temperature, 0.45)
        self.h2o_SW = self.humidity * pow(self.pressure / 1013., 1.0) * pow(273. / self.temperature, 0.45)

        c = (cst.massMolAir / cst.massMolH2O)  # Mair/Mh2o
        vew = np.vectorize(phy.ew)
        self.rh = c * np.divide(self.wvdensity, self.density) * np.divide(self.pressure, vew(self.temperature))


##################################################
#
#   interpolation on n levels (equally spaced on pressure)
#
##################################################

def pressureScale(p0, n):  # n atmospheric levels + 1 surface -> n+1 levels, in decreasing order
    p = np.empty([n + 1])
    p[0] = p0
    for i in range(n):
        p[i + 1] = p0 * (1 - (i + 0.5) / n)
    return p


def pressureBounds(p):  # bottom and top of levels - OK ONLY FOR EQUALLY SPACED LEVELS....
    n = p.shape[0]
    pt = np.empty([n])
    pt[0] = p[0]
    for i in range(1, n - 1):
        pt[i] = 0.5 * (p[i] + p[i + 1])
    pt[n - 1] = 0.
    return pt


def integratedLinearInterpolation(x,
                                  y):  # x and y are 1D np.array of same size, x increasing    -> sy = integral( y(x) dx ) on scale x
    n = x.shape[0]
    sy = np.empty([n])
    sy[0] = 0.
    for i in range(1, n):  # i = 1 to n-1
        sy[i] = np.trapz(y[:i + 1], x[:i + 1])
    return sy


def integInterpolation(x, xR, yR):
    integY = integratedLinearInterpolation(xR, yR)
    # return np.diff( np.interp( x, xR, integY ) )      ###     exact on xR scale, not true integral on x scale, but should be sufficient
    n = x.shape[0]
    sy = np.zeros(n)
    ixR = np.searchsorted(xR, x)
    for i in range(n):
        if ixR[i] > 0:
            dx = x[i] - xR[ixR[i] - 1]
            sy[i] = integY[ixR[i] - 1] + dx * yR[ixR[i] - 1]
            if ixR[i] < xR.shape[0]:
                slope = (yR[ixR[i]] - yR[ixR[i] - 1]) / (xR[ixR[i]] - xR[ixR[i] - 1])
                sy[i] += 0.5 * dx * dx * slope
    return np.diff(sy)


class Profile:
    #  n_atm = number of box in the atmosphere
    #  useScaling = 'full', 'fixed', 'no'
    def __init__(self, i, n_atm, p0=1013.25, useRH=False, useScalingC='fixed', useScalingHL='fixed',
                 useScalingHS='fixed'):
        # constant values
        self.ref_profile = StdProfile(i)
        self.pressure = pressureScale(p0, n_atm)
        self.useRelativeH = useRH
        self.use_scaled_co2_LW = useScalingC
        self.use_scaled_h2o_LW = useScalingHL
        self.use_scaled_h2o_SW = useScalingHS

        #      if self.use_scaled_h2o_LW == 'fixed' and self.useRelativeH == True:
        #          raise NameError('h2o LW scaling option "fixed" is not compatible with using relative humidity')
        #      if self.use_scaled_h2o_SW == 'fixed' and self.useRelativeH == True:
        #          raise NameError('h2o SW scaling option "fixed" is not compatible with using relative humidity')

        self.temperatureRef = np.interp(self.pressure, self.ref_profile.pressure[::-1],
                                        self.ref_profile.temperature[::-1])
        self.pressureBounds = pressureBounds(self.pressure)
        # print("ioc2",integratedLinearInterpolation(self.ref_profile.pressure[::-1], self.ref_profile.co2_LW[::-1]))

        self.rh = np.interp(self.pressure, self.ref_profile.pressure[::-1], self.ref_profile.rh[::-1])
        self.o3_SW = integInterpolation(self.pressureBounds[::-1], self.ref_profile.pressure[::-1],
                                        self.ref_profile.o3_SW[::-1])[::-1]

        # self.co2 is a constant (*deltap): interpolation gives only noise...
        # self.co2 = integInterpolation(self.pressureBounds[::-1], self.ref_profile.pressure[::-1], self.ref_profile.co2[::-1])[::-1]
        self.co2 = (p0 / n_atm) * (10 / cst.g) * (cst.massMolCO2 / cst.massMolAir) * 1.0e-6 * np.ones(n_atm)

        # values may change, but initialized here for a fix standard profile
        # if useRH == False:
        self.humidity = integInterpolation(self.pressureBounds[::-1], self.ref_profile.pressure[::-1],
                                           self.ref_profile.humidity[::-1])[::-1]

        # if useScaling == 'fixed':
        self.co2_LW = integInterpolation(self.pressureBounds[::-1], self.ref_profile.pressure[::-1],
                                         self.ref_profile.co2_LW[::-1])[::-1]
        self.h2o_SW = integInterpolation(self.pressureBounds[::-1], self.ref_profile.pressure[::-1],
                                         self.ref_profile.h2o_SW[::-1])[::-1]
        self.h2o_LW = integInterpolation(self.pressureBounds[::-1], self.ref_profile.pressure[::-1],
                                         self.ref_profile.h2o_LW[::-1])[::-1]

        # if useScaling == 'fixed' and useRelativeH
        self.h2o_SW0 = self.h2o_SW / self.humidity
        self.h2o_LW0 = self.h2o_LW / self.humidity

        # if useScaling == 'no':
        if self.use_scaled_h2o_LW == 'no':
            self.h2o_LW = self.humidity
        if self.use_scaled_h2o_SW == 'no':
            self.h2o_SW = self.humidity
        if self.use_scaled_co2_LW == 'no':
            self.co2_LW = self.co2

    def set_H2O(self, vT):
        if self.useRelativeH:
            c = (10. / cst.g) * (cst.massMolH2O / cst.massMolAir)  # (10/9.81)*(18.01524/28.97)
            # vew = np.vectorize(phy.ew)
            dpsp = -np.diff(self.pressureBounds) / self.pressure[1:]
            # self.humidity = c * dpsp * self.rh[1:] * vew(self.temperature[1:])
            self.humidity = c * dpsp * self.rh[1:] * phy.ew_vec(vT)

    def set_scaling(self, vT):
        if self.use_scaled_h2o_LW == 'full':
            self.h2o_LW = self.humidity * pow(self.pressure[1:] / 1013., 0.75) * pow(273. / vT, 0.45)
        else:
            if self.useRelativeH:
                if self.use_scaled_h2o_LW == 'no':  # use updated humidity
                    self.h2o_LW = self.humidity
                if self.use_scaled_h2o_LW == 'fixed':  # use updated humidity with fixed scaling
                    self.h2o_LW = self.humidity * self.h2o_LW0
        if self.use_scaled_h2o_SW == 'full':
            self.h2o_SW = self.humidity * pow(self.pressure[1:] / 1013., 1.0) * pow(273. / vT, 0.45)
        else:
            if self.useRelativeH:
                if self.use_scaled_h2o_SW == 'no':  # use updated humidity
                    self.h2o_SW = self.humidity
                if self.use_scaled_h2o_SW == 'fixed':  # use updated humidity with fixed scaling
                    self.h2o_SW = self.humidity * self.h2o_SW0
        if self.use_scaled_co2_LW == 'full':
            self.co2_LW = self.co2 * pow(self.pressure[1:] / 1013., 1.75) * pow(273. / vT, 8)

    def set_temperature(self, vT):
        self.set_H2O(vT)
        self.set_scaling(vT)


"""            
    #   this scaling is not identical to self.h2o_LW/SW: it depends on input T AND it is computed at the model levels (not the reference ones)
    #       -> necessary for the 'useRH' option
    def scaled_h2o_LW( self, vT ):
        return self.humidity * pow(self.pressure[1:]/1013.,0.75) * pow(273./vT,0.45)
    def scaled_h2o_SW( self, vT ):
        return self.humidity * pow(self.pressure[1:]/1013.,1.0) * pow(273./vT,0.45)
    def scaled_co2_LW( self, vT ):
        return self.co2 * pow(self.pressure[1:]/1013.,1.75) * pow(273./vT,8)
"""
