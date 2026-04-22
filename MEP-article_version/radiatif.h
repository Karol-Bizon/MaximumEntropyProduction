/*
 *  radiatif.h
 *
 *  Created by Didier on 12/02/13.
 *  Copyright 2013-2020 LSCE. All rights reserved.
 *
 *     std = C++11      (for the "stdexpr" keyword)
 */

/// dependencies = { gcem } not required
/// gcem : only for compile time computations
/// if not available:  #undef use_compile_time_table




#ifndef RADIATIF_H
#define RADIATIF_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <immintrin.h>      //  intrinsics:     _mm_cvttsd_si32 ,   _mm_load_sd

//#include "core.h"
#include "physics.h"

/////////////////////////////////////////////////////
//  précalculs des tableaux à la compilation - nécessite c++11 ou c++14 au moins ?
//#define use_compile_time_table

#ifdef use_compile_time_table

#include "gcem/gcem.hpp"

template<int N>
struct Table
{   typedef    double    (*doubleFunc)(double);
    constexpr Table( doubleFunc f, double start, double step ) : values()
    {   for (auto i = 0; i < N; ++i)
            values[i] = f(start + i*step );
    }
    double values[N];
};

#endif  //use_compile_time_table

/////////////////////////////////////////////////////
//      ordre d'appel SW, LW
//  cf radiativeModel::bilan_T()
//      la routine setSW efface le bilan, la routine LW accumule.
#define resetSW     //  #ifdef resetSW : sw[],dswdT[] = ...  //  #else  : sw[],dswdT[] += ...
//#define resetLW     //  #ifdef resetLW : lw[],dlwdT[] = ...  //  #else  : lw[],dlwdT[] += ...




/////////////////////////////////////////////////////

enum	scalingType		{	no_scaling = 0, fixedT_scaling, full_scaling };



///////////////////////////////////////////////
//
//  useful utilities not specific to ratiative
//
///////////////////////////////////////////////

namespace radiatif_util {
    
    typedef	double	(*doubleFunc)(double);
    inline double square( double x )   {   return x*x; };
    
        //  BEWARE !!
        //  v is assumed 1-offest:  v.first = v[1]  and v.last = v[n]
    inline void	hunt( size_t n, const double* v, double x, size_t& jlo )            //	d'apres Numerical Recipes 'hunt'
    {	size_t	jm,jhi,inc;
        bool	ascnd = (n == 1) || (v[n] > v[1]);      //	for n==1, ascnd is assumed true
        
        if (jlo <= 0 || jlo > n)        {	jlo=0;	jhi=n+1;	}
        else {	inc=1;
            if ((x >= v[jlo]) == ascnd)   {
                if (jlo == n) return;
                jhi=(jlo)+1;
                while ((x >= v[jhi]) == ascnd) {
                    jlo=jhi;		inc += inc;		jhi=(jlo)+inc;
                    if (jhi > n)	{		jhi=n+1;	break;		}
                }
            } else {
                if (jlo == 1) {		jlo=0;		return;		}
                jhi=(jlo)--;
                while ((x < v[jlo]) == ascnd) {
                    jhi=(jlo);
                    inc <<= 1;
                    if (inc >= jhi) {
                        jlo=0;
                        break;
                    }
                    else jlo=jhi-inc;
                }
            }
        }		
        while (jhi-jlo != 1) {
            jm=(jhi+jlo) >> 1;
            if (x > v[jm] == ascnd)	jlo = jm;
            else					jhi = jm;
        }
    };
    
        //  returns jlo such that (in the ascending case)  v[jlo] < x <= v[jlo+1]
        //  assumes v is 1-offset:      v[1] to v[n]
        //      jlo = 0  if  x <= v[1]
        //      jlo = n  if  x > v[n]
    inline size_t  bissectbracket( size_t n, const double* v, double value, size_t& klo )
    {
        hunt( n, v, value, klo );                       //	output: 0 <= klo <= n
        //if  ( klo >= n )    return	( v[n] - v[n-1] );
        //if  ( klo == 0 )    return	( v[2] - v[1] );
        //return	( v[ klo+1 ] - v[ klo ] );
        
        if ( klo >= n )     return n-1;
        if ( klo <= 0 )     return 1;
        return klo;                 //  1 <= output <= n-1   &&  0 <= klo <= n
                                    //  v is assumed 1-offest
    };
    
        //  BEWARE:
        //      XA is assumed STRICTLY monotonic
        //      ... either  XA[i] < XA[i+1] for all i
        //      ... or      XA[i] > XA[i+1]
        //  Extrapolations:
        //      klo = 0 si X <= XA[0]           X <= xa1[1]
        //      klo = n si X > XA[n-1]          X > xa1[n]
    
    inline double  linear_interpolation_1( size_t n, const double* xa1, const double* ya1, double X, size_t& klo )     //  1-offset vectors
    {   size_t	k1 = bissectbracket( n, xa1, X, klo );      //  klo not used here, but transmitted to detect extrapolation
        unsigned long	k2 = k1+1;
        return	((xa1[k2] - X) * ya1[k1] + (X - xa1[k1]) * ya1[k2]) / (xa1[k2]-xa1[k1]);
    };

    
        //  BEWARE:
        //      x1 is assumed STRICTLY INCREASING !!!!
    
    inline double  linear_interp_integ_1( size_t n, const double* x1, const double* y1, double a, double b, unsigned long& klo )  //  1-offset vectors
    {
        if (a==b)   return 0.;
        
        bool	ascnd = (n == 1) || (x1[n] > x1[1]);      //	for n==1, ascnd is assumed true
        
        if (ascnd && a>b)   return -linear_interp_integ_1( n, x1, y1, b, a, klo );
        if (!ascnd && a<b)  return -linear_interp_integ_1( n, x1, y1, b, a, klo );
            //  on se ramène aux 2 cas a < b && x[1] <...< x[n]   ou    a > b && x[1] >...> x[n]
            //  c'est-à-dire, ia <= ib, s'il y a une ou plusieurs valeurs de x[i] entre a et b, et ia>ib sinon
            //  ascnd <=> a<b   et   !ascnd <=> a>b
        
        //		ia, ib:		indices min max tels que a < x[ia] <...< x[ib] < b     ou    a > x[ia] >...> x[ib] > b
        //					ia > ib:	x[ib] < a < b < x[ia]    ou    x[ib] > a > b > x[ia]
        //                  ia > ib <=> ia = ib+1
        
        double          ya = linear_interpolation_1( n, x1, y1, a, klo );       //	valeur pour a
        unsigned long   ia = klo+1;
        bool    a_extra_0 = (klo==0);   //      donc ia = 1
        bool    a_extra_n = (klo==n);   //      donc ia = n+1
        //if (ascnd  && a_extra_0)   ia = 1;         //	ya == extrapolation ( klo == 1, a < x[1] < x[..] )
        //if (!ascnd && a_extra_n)   ia = n-1;       //	ya == extrapolation ( klo == n, a > x[n] > x[..] )
        if (a_extra_n)   ia = n;
        
        double          yb = linear_interpolation_1( n, x1, y1, b, klo );       //	valeur pour b
        unsigned long	ib = klo;
        bool    b_extra_0 = (klo==0);   //      donc ib = 1
        bool    b_extra_n = (klo==n);   //      donc ib = n+1
        //if (ascnd  && b_extra_0)   ia = 1;         //	yb == extrapolation ( klo == 1, b < x[1] < x[..] )
        //if (!ascnd && b_extra_n)   ib = n-1;       //	yb == extrapolation ( klo == n, b > x[n] > x[..] )
        if (b_extra_n)   ib = n;
        
        
        /*
        if (klo==0)         ia = klo;		//	ya == extrapolation ( klo == 1, a < x[1] )
            //else if (klo==n)
            //if ( x1[klo] > a )			ia = klo;		//	ya == extrapolation ( klo == 1, a < x[1] )
            //else if ( x1[klo+1] < a )	ia = 0;			//	rien entre a et b ( ya et yb == extrapolations: klo+1 == n, x[n] < a < b )
        */
        
        /*
        unsigned long	ib = klo;
        if ( x1[klo] > b )			ib = 0;			//	rien entre a et b ( ya et yb == extrapolations: klo == 1, a < b < x[1] )
        else if ( x1[klo+1] < b )	ib = klo+1;		//	yb == extrapolation ( klo+1 == n, x[n] < b )
        */
        
        bool    empty = (a_extra_0 && b_extra_0) || (a_extra_n && b_extra_n) || (ia>ib);       //	rien entre a et b ( x[1] > a,b   ou   x[n] < a,b   ou   x[ia] < a,b < x[ia+1])
        /*
        if (ascnd && klo==0 && )
        if ( x1[klo] > b )			ib = 0;			//	rien entre a et b ( ya et yb == extrapolations: klo == 1, a < b < x[1] )
        else if ( x1[klo+1] < b )	ib = klo+1;		//	yb == extrapolation ( klo+1 == n, x[n] < b )
        */
        
        double	s;
        //if ( (ia==0) || (ib==0) || (ia > ib) )				//	rien entre a et b
        if ( empty )                        //	rien entre a et b
            s = (yb + ya) * (b - a);
        else
        {	s = (y1[ia] + ya) * (x1[ia] - a) + (yb + y1[ib]) * (b - x1[ib]);
            for (unsigned long i=ia; i<=(ib-1); i++)
                s += (y1[i+1] + y1[i]) * (x1[i+1] - x1[i]);
        }
        if (ascnd)  s *= 0.5;
        else        s *= -0.5;
        return s;
    };
    
    inline double  linear_interpolation_0( size_t n, const double* XA, const double* YA, double X, size_t& klo )       //  0-offset vectors
    {   const double* xa1 = XA-1;       //  change to 1-offset
        const double* ya1 = YA-1;
        return linear_interpolation_1( n, xa1, ya1, X, klo );
    };
    inline double  linear_interp_integ_0( size_t n, const double* x0, const double* y0, double a, double b, unsigned long& klo )       //  0-offset vectors
    {   const double* x1 = x0-1;       //  change to 1-offset
        const double* y1 = y0-1;
        return linear_interp_integ_1( n, x1, y1, a, b, klo );
    };
    
   inline void	outputVect( const char* s, size_t n, const double* v, size_t max = 500 )
    {	std::cout << s << " = { ";
        size_t imax = std::min( n, max );
        for (size_t i=0; i<imax-1; i++)     std::cout << v[i] << ", ";
        if (imax==n)    std::cout << v[imax-1] << " } " << '\n';
        else            std::cout << v[imax-1] << " , ... } " << '\n';
    };

};




///////////////////////////////////////////////
//
//  standard atmospheric profiles from McClatchey
//
///////////////////////////////////////////////

class StdMcClatcheyProfile {
private:
    constexpr static const double pressure_table[5][33]    =
    {   //Tropical
        {1013, 904, 805, 713, 633, 559, 492, 432, 378, 329, 286, 247, 213, 182, 156, 132, 111, 93.7, 78.9, 66.6, 56.5, 48, 40.9, 35, 30, 25.7, 12.2, 6., 3.05, 1.59, 0.854, 0.0579, 0.0003},
        //MidLatitudeSummer
        {1013, 902, 802, 710, 628, 554, 487, 426, 372, 324, 281, 243, 209, 179, 153, 130, 111, 95, 81.2, 69.5, 59.5, 51, 43.7, 37.6, 32.2, 27.7, 13.2, 6.52, 3.33, 1.76, 0.951, 0.0671, 0.0003},
        //MidLatitudeWinter
        {1018, 897.3, 789.7, 693.8, 608.1, 531.3, 462.7, 401.6, 347.3, 299.2, 256.8, 219.9, 188.2, 161, 137.8, 117.8, 100.7, 86.1, 73.5, 62.8, 53.7, 45.8, 39.1, 33.4, 28.6, 24.3, 11.1, 5.18, 2.53, 1.29, 0.682, 0.0467, 0.0003},
        //SubArcticSummer
        {1010, 896, 792.9, 700, 616, 541, 473, 413, 359, 310.7, 267.7, 230, 197.7, 170, 146, 125, 108, 92.8, 79.8, 68.6, 58.9, 50.7, 43.6, 37.5, 32.27, 27.8, 13.4, 6.61, 3.4, 1.81, 0.987, 0.0707, 0.0003},
        //SubArcticWinter
        {1013, 887.8, 777.5, 679.8, 593.2, 515.8, 446.7, 385.3, 330.8, 282.9, 241.8, 206.7, 176.6, 151, 129.1, 110.3, 94.31, 80.58, 68.82, 58.75, 50.14, 42.77, 36.47, 31.09, 26.49, 22.56, 10.2, 4.701, 2.243, 1.113, 0.5719, 0.04016, 0.0003}
    };
    constexpr static const double temperature_table[5][33]	=
    {   //Tropical
        {300, 294, 288, 284, 277, 270, 264, 257, 250, 244, 237, 230, 224, 217, 210, 204, 197, 195, 199, 203, 207, 211, 215, 217, 219, 221, 232, 243, 254, 265, 270, 219, 210},
        //MidLatitudeSummer
        {294, 290, 285, 279, 273, 267, 261, 255, 248, 242, 235, 229, 222, 216, 216, 216, 216, 216, 216, 217, 218, 219, 220, 222, 223, 224, 234, 245, 258, 270, 276, 218, 210},
        //MidLatitudeWinter
        {272.2, 268.7, 265.2, 261.7, 255.7, 249.7, 243.7, 237.7, 231.7, 225.7, 219.7, 219.2, 218.7, 218.2, 217.7, 217.2, 216.7, 216.2, 215.7, 215.2, 215.2, 215.2, 215.2, 215.2, 215.2, 215.2, 217.4, 227.8, 243.2, 258.5, 265.7, 230.7, 210.2},
        //SubArcticSummer
        {287, 282, 276, 271, 266, 260, 253, 246, 239, 232, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 226, 228, 235, 247, 262, 274, 277, 216, 210},
        //SubArcticWinter
        {257.1, 259.1, 255.9, 252.7, 247.7, 240.9, 234.1, 227.3, 220.6, 217.2, 217.2, 217.2, 217.2, 217.2, 217.2, 217.2, 216.6, 216, 215.4, 214.8, 214.1, 213.6, 213, 212.4, 211.8, 211.2, 216., 222.2, 234.7, 247, 259.3, 245.7, 210}
    };
    static constexpr double density_table[5][33]		=
    {   //Tropical
        {1167, 1064., 968.9, 875.6, 795.1, 719.9, 650.1, 585.5, 525.8, 470.8, 420.2, 374, 331.6, 292.9, 257.8, 226, 197.2, 167.6, 138.2, 114.5, 95.15, 79.38, 66.45, 56.18, 47.63, 40.45, 18.31, 8.6, 4.181, 2.097, 1.101, 0.0921, 0.0005},
        //MidLatitudeSummer
        {1191, 1080, 975.7, 884.6, 799.8, 721.1, 648.7, 583, 522.5, 466.9, 415.9, 369.3, 326.9, 288.2, 246.4, 210.4, 179.7, 153.5, 130.5, 111, 94.53, 80.56, 68.72, 58.67, 50.14, 42.88, 13.22, 6.519, 3.33, 1.757, 0.9512, 0.06706, 0.0005},
        //MidLatitudeWinter
        {1301, 1162, 1037, 923, 828.2, 741.1, 661.4, 588.6, 522.2, 461.9, 407.2, 349.6, 299.9, 257.2, 220.6, 189, 182, 138.8, 118.8, 101.7, 86.9, 74.21, 63.38, 54.15, 46.24, 39.5, 17.83, 7.924, 3.625, 1.741, 0.8954, 0.07051, 0.0005},
        //SubArcticSummer
        {1220, 1110, 997.1, 898.5, 807.7, 724.4, 651.9, 584.9, 523.1, 466.3, 414.2, 355.9, 305.9, 263, 226, 194.3, 167.1, 143.6, 123.5, 106.2, 91.28, 78.49, 67.5, 58.05, 49.63, 42.47, 13.38, 6.614, 3.404, 1.817, 0.9868, 0.07071, 0.0005},
        //SubArcticWinter
        {1372, 1193, 1058, 936.6, 833.9, 745.7, 664.6, 590.4, 522.6, 453.8, 387.9, 331.5, 283.4, 242.2, 207.1, 177, 151.7, 130, 111.3, 95.29, 81.55, 69.76, 59.66, 51, 43.58, 37.22, 16.45, 7.368, 3.33, 1.569, 0.7682, 0.05695, 0.0005}
    };
    static constexpr double wvdensity_table[5][33]		=
    {   //Tropical
        {19., 13., 9.3, 4.7, 2.2, 1.5, 0.85, 0.47, 0.25, 0.12, 0.05, 0.017, 0.006, 0.0018, 0.001, 0.00076, 0.00064, 0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011, 0.000043, 0.000019, 0.0000063, 0.00000014, 0.000000001},
        //MidLatitudeSummer
        {14, 9.3, 5.9, 3.3, 1.9, 1.0, 0.61, 0.37, 0.21, 0.12, 0.064, 0.022, 0.006, 0.0018, 0.001, 0.00076, 0.00064, 0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011, 0.000043, 0.000019, 0.0000063, 0.00000014, 0.000000001},
        //MidLatitudeWinter
        {3.5, 2.5, 1.8, 1.2, 0.66, 0.38, 0.21, 0.085, 0.035, 0.016, 0.0075, 0.0069, 0.006, 0.0018, 0.001, 0.00076, 0.00064, 0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011, 0.000043, 0.000019, 0.0000063, 0.00000014, 0.000000001},
        //SubArcticSummer
        {9.1, 6, 4.2, 2.7, 1.7, 1, 0.54, 0.29, 0.013, 0.042, 0.015, 0.0094, 0.006, 0.0018, 0.001, 0.00076, 0.00064, 0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011, 0.000043, 0.000019, 0.0000063, 0.00000014, 0.000000001},
        //SubArcticWinter
        {1.2, 1.2, 0.94, 0.68, 0.41, 0.2, 0.098, 0.054, 0.011, 0.0084, 0.0055, 0.0038, 0.0026, 0.001, 0.001, 0.00076, 0.00064, 0.00056, 0.0005, 0.00049, 0.00045, 0.00051, 0.00051, 0.00054, 0.0006, 0.00067, 0.00036, 0.00011, 0.000043, 0.000019, 0.0000063, 0.00000014, 0.000000001}
    };
    static constexpr double o3density_table[5][33]		=
    {   //Tropical
        {0.000056, 0.000056, 0.000054, 0.000051, 0.000047, 0.000045, 0.000043, 0.000041, 0.000039, 0.000039, 0.000039, 0.000041, 0.000043, 0.000045, 0.000045, 0.000047, 0.000047, 0.000069, 0.00009, 0.00014, 0.00019, 0.00024, 0.00028, 0.00032, 0.00034, 0.00034, 0.00024, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086, 0.000000000043},
        //MidLatitudeSummer
        {0.00006, 0.00006, 0.00006, 0.000062, 0.000064, 0.000066, 0.000069, 0.000075, 0.000079, 0.000086, 0.00009, 0.00011, 0.00012, 0.00015, 0.00018, 0.00019, 0.00021, 0.00024, 0.00028, 0.00032, 0.00034, 0.00036, 0.00036, 0.00034, 0.00032, 0.0003, 0.0002, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086, 0.000000000043},
        //MidLatitudeWinter
        {0.00006, 0.000054, 0.000049, 0.000049, 0.000049, 0.000058, 0.000064, 0.000077, 0.00009, 0.00012, 0.00016, 0.00021, 0.00026, 0.0003, 0.00032, 0.00034, 0.00036, 0.00039, 0.00041, 0.00043, 0.00045, 0.00043, 0.00043, 0.00039, 0.00036, 0.00034, 0.00019, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086, 0.000000000043},
        //SubArcticSummer
        {0.000049, 0.000054, 0.000056, 0.000058, 0.00006, 0.000064, 0.000071, 0.000075, 0.000079, 0.00011, 0.00013, 0.00018, 0.00021, 0.00026, 0.00028, 0.00032, 0.00034, 0.00039, 0.0004, 0.00041, 0.00039, 0.00036, 0.00032, 0.0003, 0.00028, 0.00026, 0.00014, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086, 0.000000000043},
        //SubArcticWinter
        {0.000041, 0.000041, 0.000041, 0.000043, 0.000045, 0.000047, 0.000049, 0.000071, 0.00009, 0.00015, 0.00024, 0.00032, 0.00043, 0.00047, 0.00049, 0.00056, 0.00062, 0.00062, 0.00062, 0.0006, 0.00056, 0.00051, 0.00047, 0.00043, 0.00036, 0.00032, 0.00015, 0.000092, 0.000041, 0.000013, 0.0000043, 0.000000086, 0.000000000043}
    };
    
public:
    static constexpr double  pressure( size_t profile_code, size_t i )      //  profile_code = 1 à 7, i = 0 à 32
    {   return (profile_code < 6 ? pressure_table[profile_code-1][i] :
                (profile_code == 6 ? 0.5*(pressure_table[1][i]+pressure_table[2][i]) :
                 (profile_code == 7 ? 0.5*(pressure_table[3][i]+pressure_table[4][i]) : 0.)));
    };
    static constexpr double  temperature( size_t profile_code, size_t i )      //  profile_code = 1 à 7, i = 0 à 32
    {   return (profile_code < 6 ? temperature_table[profile_code-1][i] :
                (profile_code == 6 ? 0.5*(temperature_table[1][i]+pressure_table[2][i]) :
                 (profile_code == 7 ? 0.5*(temperature_table[3][i]+pressure_table[4][i]) : 0.)));
    };
    static constexpr double  density( size_t profile_code, size_t i )      //  profile_code = 1 à 7, i = 0 à 32
    {   return (profile_code < 6 ? density_table[profile_code-1][i] :
                (profile_code == 6 ? 0.5*(density_table[1][i]+density_table[2][i]) :
                 (profile_code == 7 ? 0.5*(density_table[3][i]+density_table[4][i]) : 0.)));
    };
    static constexpr double  wvdensity( size_t profile_code, size_t i )      //  profile_code = 1 à 7, i = 0 à 32
    {   return (profile_code < 6 ? wvdensity_table[profile_code-1][i] :
                (profile_code == 6 ? 0.5*(wvdensity_table[1][i]+wvdensity_table[2][i]) :
                 (profile_code == 7 ? 0.5*(wvdensity_table[3][i]+wvdensity_table[4][i]) : 0.)));
    };
    static constexpr double  o3density( size_t profile_code, size_t i )      //  profile_code = 1 à 7, i = 0 à 32
    {   return (profile_code < 6 ? o3density_table[profile_code-1][i] :
                (profile_code == 6 ? 0.5*(o3density_table[1][i]+o3density_table[2][i]) :
                 (profile_code == 7 ? 0.5*(o3density_table[3][i]+o3density_table[4][i]) : 0.)));
    };
    
    static double  relativeH( size_t profile_code, size_t i )
    {   return physics::Mair/physics::Mh2o * wvdensity( profile_code, i )/density( profile_code, i ) * pressure( profile_code, i )/physics::ew_kelvin( temperature( profile_code, i ) );  }
    static double  humidity( size_t profile_code, size_t i )
    {   return (10./physics::g)*wvdensity( profile_code, i )/density( profile_code, i );                    };
    static double  o3_sw( size_t profile_code, size_t i )
    {   return (10./physics::g)*o3density( profile_code, i )/density( profile_code, i )/physics::MvolO3;    };
    
    static double  scaling( size_t profile_code, size_t i, double p0, double n, double T0, double m )
    {   double rp = pressure( profile_code, i )/p0;
        double rt = T0/temperature( profile_code, i );
        return pow( rp, n ) * pow( rt, m );
    };
    static double  co2( size_t profile_code, size_t i )
    {   return (10./physics::g)*1.e-6*100.*physics::Mco2/physics::Rgp*pressure( profile_code, i )/(temperature( profile_code, i )*density( profile_code, i ));      };
    
        //  avec interpolation entre les profils ... cf (Herbert et al. QJRMS 2011)
    
    static constexpr double ReferenceLatitudes[7] = {-90., -60., -50., 0., 45., 55., 90.};  //  SubArctic=7, SubArctic=7, MidLat=6, Tropical=1, MidLat=6, SubArctic=7, SubArctic=7
    
    typedef double  (*function_to_interpolate)( size_t , size_t );
    static double  interpol( double latitude, size_t i, function_to_interpolate f )
    {   size_t  klo = 0;
        size_t  k1 = radiatif_util::bissectbracket( 7, ReferenceLatitudes-1, latitude, klo );
        double refValues[7] = { f(7,i), f(7,i), f(6,i), f(1,i), f(6,i), f(7,i), f(7,i) };
        return	((ReferenceLatitudes[k1] - latitude) * refValues[k1-1] + (latitude - ReferenceLatitudes[k1-1]) * refValues[k1]) / (ReferenceLatitudes[k1]-ReferenceLatitudes[k1-1]);
    };
    static double  scaling_inter( double latitude, size_t i, double p0, double n, double T0, double m )
    {   size_t  klo = 0;
        size_t  k1 = radiatif_util::bissectbracket( 7, ReferenceLatitudes-1, latitude, klo );
        double refValues[7] = { scaling(7,i,p0,n,T0,m), scaling(7,i,p0,n,T0,m), scaling(6,i,p0,n,T0,m), scaling(1,i,p0,n,T0,m), scaling(6,i,p0,n,T0,m), scaling(7,i,p0,n,T0,m), scaling(7,i,p0,n,T0,m) };
        return	((ReferenceLatitudes[k1] - latitude) * refValues[k1-1] + (latitude - ReferenceLatitudes[k1-1]) * refValues[k1]) / (ReferenceLatitudes[k1]-ReferenceLatitudes[k1-1]);
    };
    static double  pressure_inter( double latitude, size_t i )      {   return interpol( latitude, i, pressure );       }
    static double  temperature_inter( double latitude, size_t i )   {   return interpol( latitude, i, temperature );    }
    static double  o3_sw_inter( double latitude, size_t i )         {   return interpol( latitude, i, o3_sw );          }
    static double  co2_inter( double latitude, size_t i )           {   return interpol( latitude, i, co2 );            }
    static double  humidity_inter( double latitude, size_t i )      {   return interpol( latitude, i, humidity );       }
    static double  relativeH_inter( double latitude, size_t i )     {   return interpol( latitude, i, relativeH );      }
    
};

struct stdProfOptions {
    scalingType hsw_scaling;    //  re-scale H2O profile
    scalingType hlw_scaling;    //  re-scale H2O profile
    scalingType co2_scaling;    //  re-scale H2O profile
    bool use_Rh;                //  humidity is computed from relative humidity => dh/dT = h.dlogew(T)
    
    constexpr stdProfOptions( scalingType hsw_sc, scalingType hlw_sc, scalingType co2_sc, bool rh ):
        hsw_scaling(hsw_sc),hlw_scaling(hlw_sc),co2_scaling(co2_sc),use_Rh(rh){};
};
static constexpr stdProfOptions    defaultProfOptions = stdProfOptions(scalingType::fixedT_scaling,scalingType::fixedT_scaling,scalingType::fixedT_scaling,false);

class StdProfile {
private:
    static constexpr double T0 = 273.;      //  K
    static constexpr double P0 = 1013.;
    size_t  klo;
    
    void initPression( double p_surf )
    {   double delta_p = p_surf/nAtmoLevels;        //  on initialise les niveaux de pressions (equirepartis)
        for (int i=0; i<nAtmoLevels+1; i++)
            press_bot[i] = p_surf - delta_p*i;
        press_top = press_bot+1;                    //  press_top[0] = press_bot[1]
        
        for (int i=0; i<nAtmoLevels; i++)
        {   press[i] = 0.5 * (press_bot[i] + press_top[i]);
            press_pi_bot[i] = pow( press_bot[i]/press[i], physics::kappa );
            press_pi_top[i] = pow( press_top[i]/press[i], physics::kappa );
        }
    };
    
public:
    size_t nAtmoLevels;
    stdProfOptions options;
    
    double* press;      	//hPa   [p1_mid, ... pN_mid ]
    double* press_bot;      //hPa   [p_surf = p1_bottom, p1_top=p2_bottom, ... pN_top = 0 ]
    double* press_top;      // = press_bot+1
    double* press_pi_bot;
    double* press_pi_top;
    double* temp;           //K
    double* temp_atm;       // = temp+1
    double* o3_sw;          //
    double  o3_sum;         //
    double* co2;            //
    double* co2_lw;         //
    double* humidity;       //
    double* h2o_sw0;        //
    double* h2o_lw0;        //
    double* relativeH;      //
    double* h2o_lw;         //
    double* h2o_sw;         //
    double  co2ppm;         //
    
public:
    ~StdProfile()
    {   delete press_bot; delete press; delete press_pi_bot; delete press_pi_top; delete temp; delete o3_sw; delete co2; delete co2_lw;
        delete humidity; delete relativeH; delete h2o_lw; delete h2o_sw; delete h2o_sw0; delete h2o_lw0;
    }
    
    StdProfile( int profile_code, size_t nAtm, double p_surf, stdProfOptions opt, double pco2, double latitude = 0.0 ):klo(0),options(opt)
                //  profile_code = 1 à 7 => McClatchey - profile_code = 8 => McClatchey interpolé en latitude
    {   nAtmoLevels = nAtm;
        co2ppm = pco2;
        press_bot = new double[nAtmoLevels+1];
        press = new double[nAtmoLevels];
        press_pi_bot = new double[nAtmoLevels];
        press_pi_top = new double[nAtmoLevels];
        co2 = new double[nAtmoLevels];
        temp  = new double[nAtmoLevels+1];          //  temp[0] = temperature de surface
        o3_sw  = new double[nAtmoLevels];
        relativeH = new double[nAtmoLevels];
        temp_atm = temp+1;                          //  temp_atm[0] = temperature du 1er niveau atm
        
            //  humidity, h2o_lw, h2o_sw, co2_lw peuvent être modifié, si option = useRH, ou scaling = full
        humidity = new double[nAtmoLevels];
        h2o_lw = new double[nAtmoLevels];
        h2o_sw = new double[nAtmoLevels];
        co2_lw = new double[nAtmoLevels];
        /*
        h2o_sw0 = new double[nAtmoLevels];
        h2o_lw0 = new double[nAtmoLevels];
        */
        if (options.use_Rh && options.hsw_scaling == fixedT_scaling)
                h2o_sw0 = new double[nAtmoLevels];
        else    h2o_sw0 = NULL;
        if (options.use_Rh && options.hlw_scaling == fixedT_scaling)
                h2o_lw0 = new double[nAtmoLevels];
        else    h2o_lw0 = NULL;
        
            //  on récupère les profils McClatchey de référence sur 33 niveaux (éventuellement avec une interpolation horizontale, si profile_code == 8  )
        
        double pres_ref[33];
        double temp_ref[33];
        double o3sw_ref[33];
        double co2lw_ref[33];
        double h_ref[33];
        double rh_ref[33];
        double h2olw_ref[33];
        double h2osw_ref[33];
        
        if (profile_code<8)
        {   for (int k=0; k<33; k++)
            {   pres_ref[k] = StdMcClatcheyProfile::pressure( profile_code, k );
                temp_ref[k] = StdMcClatcheyProfile::temperature( profile_code, k );
                o3sw_ref[k] = StdMcClatcheyProfile::o3_sw( profile_code, k );
                co2lw_ref[k] = StdMcClatcheyProfile::co2( profile_code, k ) * StdMcClatcheyProfile::scaling( profile_code, k, P0, 1.75, T0, 8.0 );
                h_ref[k] = StdMcClatcheyProfile::humidity( profile_code, k );
                rh_ref[k] = StdMcClatcheyProfile::relativeH( profile_code, k );
                h2olw_ref[k] = h_ref[k] * StdMcClatcheyProfile::scaling( profile_code, k, P0, 0.75, T0, 0.45 );
                h2osw_ref[k] = h_ref[k] * StdMcClatcheyProfile::scaling( profile_code, k, P0, 1.0, T0, 0.45 );
            }
        }
        else if (profile_code==8)
        {   for (int k=0; k<33; k++)
            {   pres_ref[k] = StdMcClatcheyProfile::pressure_inter( latitude, k );
                temp_ref[k] = StdMcClatcheyProfile::temperature_inter( latitude, k );
                o3sw_ref[k] = StdMcClatcheyProfile::o3_sw_inter( latitude, k );
                co2lw_ref[k] = StdMcClatcheyProfile::co2_inter( latitude, k ) * StdMcClatcheyProfile::scaling_inter( latitude, k, P0, 1.75, T0, 8.0 );
                h_ref[k] = StdMcClatcheyProfile::humidity_inter( latitude, k );
                rh_ref[k] = StdMcClatcheyProfile::relativeH_inter( latitude, k );
                h2olw_ref[k] = h_ref[k] * StdMcClatcheyProfile::scaling_inter( latitude, k, P0, 0.75, T0, 0.45 );
                h2osw_ref[k] = h_ref[k] * StdMcClatcheyProfile::scaling_inter( latitude, k, P0, 1.0, T0, 0.45 );
            }
        }
            //  on initialise le profil standard
        
        initPression( p_surf );
        constexpr double co2_cte = (10./physics::g)*(physics::Mco2/physics::Mair)*1.e-6;
        for (size_t k=0; k<nAtmoLevels; k++)
            co2[k] = co2ppm * co2_cte * (press_bot[k]-press_top[k]);
        
                    //	ici on interpole, on n'intègre pas entre 2 niveaux
        temp[0] = radiatif_util::linear_interpolation_0( 33, pres_ref, temp_ref, press_bot[0], klo );
        for (size_t k=0; k<nAtmoLevels; k++)
        {   temp_atm[k] = radiatif_util::linear_interpolation_0( 33, pres_ref, temp_ref, press[k], klo );
            relativeH[k] = radiatif_util::linear_interpolation_0( 33, pres_ref, rh_ref, press[k], klo );
        }
                    //	ici on intègre entre 2 niveaux
                    //  ATTENTION integ_interp a l'envers !!!!
        o3_sum = 0.;
        for (size_t k=0; k<nAtmoLevels; k++)
        {   humidity[k] = radiatif_util::linear_interp_integ_0( 33, pres_ref, h_ref,     press_bot[k], press_top[k], klo );
            h2o_lw[k]   = radiatif_util::linear_interp_integ_0( 33, pres_ref, h2olw_ref, press_bot[k], press_top[k], klo );
            h2o_sw[k]   = radiatif_util::linear_interp_integ_0( 33, pres_ref, h2osw_ref, press_bot[k], press_top[k], klo );
            if (h2o_sw0) h2o_sw0[k] = h2o_sw[k]/humidity[k];
            if (h2o_lw0) h2o_lw0[k] = h2o_lw[k]/humidity[k];
            co2_lw[k]   = co2ppm * radiatif_util::linear_interp_integ_0( 33, pres_ref, co2lw_ref, press_bot[k], press_top[k], klo );
            o3_sw[k]    = radiatif_util::linear_interp_integ_0( 33, pres_ref, o3sw_ref,  press_bot[k], press_top[k], klo );
            o3_sum += o3_sw[k];
        }
        
            //  on initialise scaling si different de "fixed"   -> h2o_sw, h2o_lw, co2_lw sont des copies
        if (options.hsw_scaling == no_scaling)
            h2o_sw = humidity;
        if (options.hlw_scaling == no_scaling)
            h2o_lw = humidity;
        if (options.co2_scaling == no_scaling)
            co2_lw = co2;
    }
    
    void set_temperature( const double* Ta )
    {       /*
        if (options.use_Rh)         //  computes humidity from (fixed profile) relativeH and (varying) temperature
        {   constexpr double c_cte = (10./physics::g)*(physics::Mh2o/physics::Mair);
            for (size_t k=0; k<nAtmoLevels; k++)
            {   double c = c_cte * (press_bot[k]-press_bot[k+1])/press[k];
                humidity[k] = c * relativeH[k] * physics::ew_kelvin(Ta[k]);
            }
            
            // le cas use_Rh + fixedT_scaling est plus problématique... on utilise le scaling en température sur 33 niveaux ...
            if (options.hsw_scaling == fixedT_scaling)
                for (int k=0; k<nAtmoLevels; k++)  h2o_sw[k] = h2o_sw0[k] * humidity[k];
            if (options.hlw_scaling == fixedT_scaling)
                for (int k=0; k<nAtmoLevels; k++)  h2o_lw[k] = h2o_lw0[k] * humidity[k];
            
        }
        
        if (options.hsw_scaling == full_scaling)
            for (int k=0; k<nAtmoLevels; k++)  h2o_sw[k] = humidity[k] * (press[k]/1013.) * pow( fabs(273./Ta[k]), 0.45 );
        if (options.hlw_scaling == full_scaling)
            for (int k=0; k<nAtmoLevels; k++)  h2o_lw[k] = humidity[k] * pow( fabs(press[k]/1013.), 0.75) * pow( fabs(273./Ta[k]), 0.45 );
        if (options.co2_scaling == full_scaling)
            for (int k=0; k<nAtmoLevels; k++)  co2_lw[k] = co2[k] * pow( fabs(press[k]/1013.), 1.75) * pow( fabs(273./Ta[k]), 8 );
                */
        set_hscal_from_temp_rh( Ta, relativeH, humidity, h2o_sw, h2o_lw, co2_lw );
    }
    
        //  { h, hs, hl, cl } = profile.function( Ta, rh )
    void set_hscal_from_temp_rh( const double* Ta, const double* rh, double* h, double* hs, double* hl, double* cl ) const
    {   if (options.use_Rh)         //  computes humidity from (fixed profile) relativeH and (varying) temperature
        {   constexpr double c_cte = (10./physics::g)*(physics::Mh2o/physics::Mair);
            for (size_t k=0; k<nAtmoLevels; k++)
            {   double c = c_cte * (press_bot[k]-press_bot[k+1])/press[k];
                h[k] = c * rh[k] * physics::ew_kelvin(Ta[k]);
            }
            
            // le cas use_Rh + fixedT_scaling est plus problématique... on utilise le scaling en température sur 33 niveaux ...
            if (options.hsw_scaling == fixedT_scaling)
                for (int k=0; k<nAtmoLevels; k++)  hs[k] = h2o_sw0[k] * h[k];
            if (options.hlw_scaling == fixedT_scaling)
                for (int k=0; k<nAtmoLevels; k++)  hl[k] = h2o_lw0[k] * h[k];
        }
        
        if (options.hsw_scaling == full_scaling)
            for (int k=0; k<nAtmoLevels; k++)  hs[k] = h[k] * (press[k]/1013.) * pow( fabs(273./Ta[k]), 0.45 );
        if (options.hlw_scaling == full_scaling)
            for (int k=0; k<nAtmoLevels; k++)  hl[k] = h[k] * pow( fabs(press[k]/1013.), 0.75) * pow( fabs(273./Ta[k]), 0.45 );
        if (options.co2_scaling == full_scaling)
            for (int k=0; k<nAtmoLevels; k++)  cl[k] = co2[k] * pow( fabs(press[k]/1013.), 1.75) * pow( fabs(273./Ta[k]), 8 );
        
    }
    
};



///////////////////////////////////////////////
//
//      Short Wave model
//
///////////////////////////////////////////////

struct swOptions {
    bool usingO3;
    bool usingH2O;
    scalingType do_H_scaling;      //  re-scale H2O profile
    bool fast;
    bool use_Rh;            //  humidity is computed from relative humidity => dh/dT = h.dlogew(T)
    
    //constexpr swOptions( bool o3, bool h2o, int h_sc, bool f, bool rh ):
    //    usingO3(o3),usingH2O(h2o),do_H_scaling(static_cast<scalingType>(h_sc)),fast(f),use_Rh(rh){};
    constexpr swOptions( bool o3, bool h2o, scalingType hsw_sc, bool f, bool rh ):
        usingO3(o3),usingH2O(h2o),do_H_scaling(hsw_sc),fast(f),use_Rh(rh){};
};

static constexpr swOptions    defaultSwOptions = swOptions(true,true,scalingType::fixedT_scaling,false,false);   //  useO3, useH2O, HswScaling, !fast, !rh

//#define not_const_sw

class swModel {
    
    /////////////   Partie statique     ///////////
    
private:
    static constexpr double Rrstar = 0.0685;
    static constexpr double Rastar = 0.144;
    static constexpr double Mbar = 1.9;
    
    static double	Awvyamamoto(double u)					{	return 2.9*u/(pow(1 + 141.5*u, 0.635) + 5.925*u);	}
    static double	Awvfowler(double u)						{	return 0.0946*pow(u, 0.303);	}
    static double	Awvkorb(double u)						{	return (u==0. ? 0.: 0.5*pow(10.,(-0.74+log10(u)*(0.347-log10(u)*(0.056+0.006*log10(u) )))));	}
    
    static double	trueAwv( double u )						{	return Awvyamamoto(u);	}
    
    //static interpolLookUpTableFunction	approxAwv;						//approxAwv(0., 20., .0005, swModel::trueAwv);
    //static double   apAwv( double x )                       {	return approxAwv( x );	};
    //inline static double Awv( double x )					{	return approxAwv( x );	};
    //inline static double Awv( double x )					{	return trueAwv( x );	};
    
    static double	Aoz( double x )
    {	return	(0.02118*x)/(1 + 0.042*x + 0.000323*x*x) + (1.082*x)/pow((1 + 138.6*x),0.805) + (0.0658*x)/(1 + (103.6*x)*(103.6*x)*(103.6*x));		}
    static double	Aozone( double u, double cosz )		{	return Aoz( 35.*u/sqrt(1224.*cosz*cosz + 1.) );		}
    static double	Rr( double cosz )					{	return 0.28/(1 + 6.43*cosz);		}
    static double	Ra( double cosz )					{	return 0.219/(1 + 0.816*cosz);		}

    static double Mozone(double cosz)					{ return 35./sqrt(1224.*cosz*cosz+1.);	}
    
    static double	DAwv( double u )		//	la dérivée de Awv (formule de Yamamoto)
    {	double z = 1. + 141.5*u;
        double p = pow(z, 0.635);
        double x = 1.0/(p + 5.925*u);
        return 2.9*x*(1.0 - u*x*( 5.925 + 89.8525*p/z ));
    };
    
    static double	swscat( double O3, double cosZ )                    {	return 0.353 + (0.647 - Rr( cosZ ) - Aozone( O3, cosZ ));   };	//	sabar( alb = 0 )
    static double	sabar( double O3, double alb, double cosZ )         {	return 0.353 + (swscat(O3,cosZ) - 0.353)/(1 - Rrstar*alb);  };
    static double   alboz( double alb, double cosz )					{	return Ra(cosz)+(1-Ra(cosz))*alb*(1-Rastar)/(1-alb*Rastar); };
    
    static double	dsabar_dA( double O3, double alb, double cosZ )     {	return (swscat(O3,cosZ) - 0.353)*Rrstar/radiatif_util::square(1 - Rrstar*alb);	};
    static double	dalboz_dA( double alb, double cosz )                {	return (1-Ra(cosz))*(1-Rastar)/radiatif_util::square(1 - alb*Rastar);                 };
    
    static void AxDown( size_t n, double* AxD, const double* v, double c, radiatif_util::doubleFunc f )              //  n = v.size  -  AxD.size = n+1
    {	double  s = 0;
        for (int i=n-1; i>=0; i--)  {	s += v[i];  AxD[i] = f(s*c);        };
        AxD[n] = s*c;
    };
    static void AxUp( size_t n, double* AxU, const double* v, double c, double z, radiatif_util::doubleFunc f )      //  n = v.size  -  AxU.size = n+1
    {	double s = 0;
        AxU[0] = f(z);
        for (int i=0; i<n; i++)     {	s += v[i];	 AxU[i+1] = f(z+s*c);   };
    };
    static void set_sw_sum( size_t n, double* sw, double a, const double* a_up, const double* a_down, const double* sab = NULL )
    {   for (int i=0; i<n; i++)
        {	sw[i+1] += ((a_down[i]-a_down[i+1]) + a*(a_up[i+1]-a_up[i]));
        }
        if (sab)    sw[0] += (1-a) * (*sab - a_down[0]);     //  pour h2o
        //else      sw[0] += 0.;                             //  pour o3
    };
/*    static void set_dsw_sum( size_t n, double* sw, double a, const double* a_up, const double* a_down, const double* sab = NULL )
    {   for (int i=0; i<n; i++)
        {	sw[i+1] += ((a_down[i]-a_down[i+1]) + a*(a_up[i+1]-a_up[i]));
        }
        if (sab)    sw[0] += (1-a) * (*sab - a_down[0]);     //  pour h2o
        //else      sw[0] += 0.;                             //  pour o3
    };*/
    
    /////////////   buffer     ///////////
    
    class sw_buffer {       //  allocation en début de call, par  "double temp[sw_buffer::buffer_size(n)];  sw_buffer b(n,temp);"
    public:
        static size_t buffer_size( size_t n )   {   return 5*n+1; };    //  (n+1) + (n+1) + (3n-1)
    private:
        size_t          n_atm;
        double*         buffer, *dsw_dh;
    public:
        double  *aUp, *aDown;
        sw_buffer( size_t n, double* buf ):n_atm(n),buffer(buf)
        {   aUp = buffer;   aDown = buffer+n+1;   dsw_dh = buffer+2*n+2;  };
        const double operator()( size_t i, size_t j ) const
        {   if (i==0)       return dsw_dh[0];                        //  = dsw_0j
            size_t ia = i-1;
            if (ia<j)       return dsw_dh[1+ia];           //  = dsw_i_lt_j[ia]
            else if (ia>j)  return dsw_dh[n_atm+ia-1];     //  = dsw_i_gt_j[ia]
            else            return dsw_dh[2*n_atm-1+ia];   //  = dsw_i_eq_j[ia]
        };
        double& operator()()            {   return dsw_dh[0];              };
        double& i_lt_j( size_t ia )     {   return dsw_dh[1+ia];           };
        double& i_gt_j( size_t ia )     {   return dsw_dh[n_atm+ia];       };
        double& i_eq_j( size_t ia )     {   return dsw_dh[2*n_atm+ia-1];   };
    };
    
    /////////////   Partie dynamique     ///////////
    
private:
    size_t      nAtmoLevels;
    swOptions   options;
    //thread_local static double *aUp, *aDown;
    //thread_local static double *h2Osw, *_dsw_dh;
    //double  *h2Osw;       //
//#ifdef not_const_sw
//    double  *aUp, *aDown;           //  temporary storage - there is only one instance of swModel, so all variables could be declared static
//    double  *_dsw_dh;       //
//#endif
    
//#ifdef not_const_sw
//    void set_sw_o3( double* sw, double alboz, const double* o3, double cosZ )
//#else
    void set_sw_o3( double* sw, double alboz, const double* o3, double cosZ, sw_buffer& b ) const
//#endif
    {
//#ifndef not_const_sw
        //double aUp[nAtmoLevels+1];
        //double aDown[nAtmoLevels+1];
        double* aUp = b.aUp;
        double* aDown = b.aDown;
//#endif
        AxDown( nAtmoLevels, aDown, o3, Mozone(cosZ), Aoz );            //  dépend de o3, cosZ
        AxUp( nAtmoLevels, aUp, o3, Mbar, aDown[nAtmoLevels], Aoz );    //  dépend de o3
        aDown[nAtmoLevels] = 0;
        set_sw_sum( nAtmoLevels, sw, alboz, aUp, aDown );               //  dépend de o3, alb_ozone, cosZ
    };
    
//#ifdef not_const_sw
//    void set_sw_h2o( double* sw, double alb, const double* h2osw, double cosZ, double sab )
//#else
    void set_sw_h2o( double* sw, double alb, const double* h2osw, double cosZ, double sab, sw_buffer& b ) const
//#endif
    {
//#ifndef not_const_sw
        //double aUp[nAtmoLevels+1];
        //double aDown[nAtmoLevels+1];
        double* aUp = b.aUp;
        double* aDown = b.aDown;
//#endif
        AxDown( nAtmoLevels, aDown, h2osw, 1.0/cosZ, trueAwv );                 //  dépend de h2osw, cosZ
        AxUp( nAtmoLevels, aUp, h2osw, 5./3., aDown[nAtmoLevels], trueAwv );    //  dépend de h2osw
        aDown[nAtmoLevels] = 0;
        set_sw_sum( nAtmoLevels, sw, alb, aUp, aDown, &sab );                   //  dépend de h2osw, alb, cosZ, sabar(o3,alb,cosZ)
    };
    
    //  set the derivative of sw vs. albedo : dsw/da  [0, nLevel]
    //  A (re-)TESTER !!!
    
//#ifdef not_const_sw
//    void set_dsw_da( double* dsw, double eps, double alb, double O3, const double* o3, const double* h2Osw, double cosZ )
//#else
    void set_dsw_da( double* dsw, double eps, double alb, double O3, const double* o3, const double* h2Osw, double cosZ, sw_buffer& b ) const
//#endif
    {
//#ifndef not_const_sw
        //double aUp[nAtmoLevels+1];
        //double aDown[nAtmoLevels+1];
        double* aUp = b.aUp;
        double* aDown = b.aDown;
//#endif
        //doubleFunc AwvF = (options.fast ? apAwv : trueAwv);
        double sab = sabar( O3, alb, cosZ );
        double dsab = dsabar_dA( O3, alb, cosZ );
        double dalb = dalboz_dA( alb, cosZ );
        
            //  ozone
        AxUp( nAtmoLevels, aUp, o3, Mbar, Mozone(cosZ)*O3, Aoz );
        for (int i=0; i<nAtmoLevels; i++)
            dsw[i+1] += dalb*(aUp[i+1]-aUp[i]);
        
            //  water
        AxDown( nAtmoLevels, aDown, h2Osw, 1.0/cosZ, trueAwv );
        AxUp( nAtmoLevels, aUp, h2Osw, 5./3., aDown[nAtmoLevels], trueAwv );
        for (int i=0; i<nAtmoLevels; i++)
            dsw[i+1] += (aUp[i+1]-aUp[i]);
        
        dsw[0] = (1-alb)*dsab - sab + aDown[0];
        
        for (int i=0; i<=nAtmoLevels; i++)  dsw[i] *= eps;
    };
/*
#ifdef not_const_sw
    double dsw_dh( size_t i, size_t j )         //  = dsw(i)/dh(j)   with  i = [0, nAtmoLevels]  j = [0, nAtmoLevels-1]
    {   if (i==0)       return *_dsw_dh;                        //  = dsw_0j
        size_t ia = i-1;
        if (ia<j)       return *(_dsw_dh+1+ia);                 //  = dsw_i_lt_j[ia]
        else if (ia>j)  return *(_dsw_dh+nAtmoLevels+ia-1);     //  = dsw_i_gt_j[ia]
        else            return *(_dsw_dh+2*nAtmoLevels-1+ia);   //  = dsw_i_eq_j[ia]
    }
#else
*/
    //  set the derivative of sw vs. h2o : dsw/dh  [0, nLevel]x[1, nLevel]  (dsw(i)/dh(j))
    /*
    class dsw_dh_mat {
    //public:
    private:
        size_t          n_atm;
        double_vector   _dsw_dh;
        //double*         _dsw_dh;
    public:
        dsw_dh_mat(size_t n):n_atm(n),_dsw_dh(3*n-1)
        {
        }
        const double operator()( size_t i, size_t j ) const
        {   if (i==0)       return _dsw_dh(0);                        //  = dsw_0j
            size_t ia = i-1;
            if (ia<j)       return _dsw_dh(1+ia);           //  = dsw_i_lt_j[ia]
            else if (ia>j)  return _dsw_dh(n_atm+ia-1);     //  = dsw_i_gt_j[ia]
            else            return _dsw_dh(2*n_atm-1+ia);   //  = dsw_i_eq_j[ia]
        };
        double& operator()()            {   return _dsw_dh(0);              };
        double& i_lt_j( size_t ia )     {   return _dsw_dh(1+ia);           };
        double& i_gt_j( size_t ia )     {   return _dsw_dh(n_atm+ia);       };
        double& i_eq_j( size_t ia )     {   return _dsw_dh(2*n_atm+ia-1);   };
    };*/
//#endif
    
//#ifdef not_const_sw
//    void set_dsw_dh( double eps, double alb, const double* h2osw, double cosz )
//#else
    void set_dsw_dh( double eps, double alb, const double* h2osw, double cosz /*, dsw_dh_mat& dsw_dh*/, sw_buffer& b ) const
//#endif
    {
//#ifndef not_const_sw
        //double aUp[nAtmoLevels+1];
        //double aDown[nAtmoLevels+1];
        double* aUp = b.aUp;
        double* aDown = b.aDown;
//#endif
        AxDown( nAtmoLevels, aDown, h2osw, 1.0/cosz, DAwv );
        AxUp( nAtmoLevels, aUp, h2osw, 5./3., aDown[nAtmoLevels], DAwv );
    /*
#ifdef not_const_sw
        //  computes only (3*nLevel-1) values
        double *dsw_0j      = _dsw_dh;
        double *dsw_i_lt_j  = _dsw_dh+1;
        double *dsw_i_gt_j  = _dsw_dh+nAtmoLevels;
        double *dsw_i_eq_j  = _dsw_dh+2*nAtmoLevels-1;
        
        *dsw_0j = -eps*(1-alb)*aDown[0]/cosz;          //  1 valeur
        for (int i=0; i<nAtmoLevels-1; i++) dsw_i_lt_j[i] = eps * ((aDown[i]-aDown[i+1]) + alb*(aUp[i+1] - aUp[i]))/cosz;               //  nLevel-1
        for (int i=0; i<nAtmoLevels-1; i++) dsw_i_gt_j[i] = eps * alb*(5./3 + 1/cosz)*(aUp[i+2] - aUp[i+1]);                            //  nLevel-1
        for (int i=0; i<nAtmoLevels; i++)   dsw_i_eq_j[i] = eps * ( aDown[i]/cosz + alb*((5./3 + 1/cosz)*aUp[i+1] - aUp[i]/cosz) );     //  nLevel
        
        //radiatif_util::outputVect("dsw_",3*nAtmoLevels-1,_dsw_dh);
#else
     */
        b() = -eps*(1-alb)*aDown[0]/cosz;          //  1 valeur
        for (int i=0; i<nAtmoLevels-1; i++) b.i_lt_j(i) = eps * ((aDown[i]-aDown[i+1]) + alb*(aUp[i+1] - aUp[i]))/cosz;               //  nLevel-1
        for (int i=0; i<nAtmoLevels-1; i++) b.i_gt_j(i) = eps * alb*(5./3 + 1/cosz)*(aUp[i+2] - aUp[i+1]);                            //  nLevel-1
        for (int i=0; i<nAtmoLevels; i++)   b.i_eq_j(i) = eps * ( aDown[i]/cosz + alb*((5./3 + 1/cosz)*aUp[i+1] - aUp[i]/cosz) );     //  nLevel
        
        //radiatif_util::outputVect("dsw_",3*nAtmoLevels-1,&(dsw_dh._dsw_dh(0)));
//#endif
        /*
        radiatif_util::outputVect("dsw_0j",1,dsw_0j);
        radiatif_util::outputVect("dsw_i_lt_j",nAtmoLevels-1,dsw_i_lt_j);
        radiatif_util::outputVect("dsw_i_gt_j",nAtmoLevels-1,dsw_i_gt_j);
        radiatif_util::outputVect("dsw_i_eq_j",nAtmoLevels,dsw_i_eq_j);
        */
        /*
        std::cout << "dsw_dh = { ";
        for (size_t i=0; i<=nAtmoLevels; i++)
        for (size_t j=0; j<=nAtmoLevels-1; j++)     std::cout << dsw_dh(i,j) << ", ";
        std::cout << " }\n";
        */
    };
    
public:
    swModel( size_t n, swOptions opt = defaultSwOptions ):nAtmoLevels(n),options(opt)
    {
        //h2Osw = new double[nAtmoLevels];
/*#ifdef not_const_sw
        aUp = new double[nAtmoLevels+1];
        aDown = new double[nAtmoLevels+1];
        _dsw_dh = new double[3*nAtmoLevels-1];
#endif
*/
    };
    ~swModel()
    {   /* delete h2Osw; */
/*#ifdef not_const_sw
        delete aUp; delete aDown; delete _dsw_dh;
#endif*/
    }
    
    
    
//#ifdef not_const_sw
//    void setSW( double* sw, double* dswdT, const double* T, StdProfile& pfl, double alb, double solar, double cosZ )
//#else
    void setSW( double* sw, double* dswdT, const double* T, /*const double* h,*/ const double* h2Osw, StdProfile& pfl, double alb, double solar, double cosZ ) const
//#endif
    {
//#ifndef not_const_sw
        //dsw_dh_mat dsw_dh(nAtmoLevels);
        double temp[sw_buffer::buffer_size(nAtmoLevels)];       //  buffer = { aUp, aDown, dsw_dh }
        sw_buffer b( nAtmoLevels, temp );
//#endif

   /*     setSW( sw, dswdT, T, pfl.humidity, pfl.h2o_sw, alb, solar, pfl.o3_sum, pfl.o3_sw, cosZ, pfl.press );
    };
    
    void setSW( double* sw, double* dswdT, const double* T, const double* h, const double* hSW,
               double alb, double solar, double O3, const double* o3, double cosZ, const double* press )
    {*/
        /*
        if (options.do_H_scaling == full_scaling)
        {   for (int k=0; k<nAtmoLevels; k++)  h2Osw[k] = h[k] * (press[k]/1013.) * pow( fabs(273./Ta[k]), 0.45 );
        }
        else if (options.do_H_scaling == fixedT_scaling)
        {   for (int k=0; k<nAtmoLevels; k++)  h2Osw[k] = hSW[k];
        }
        else if (options.do_H_scaling == no_scaling)
        {   for (int k=0; k<nAtmoLevels; k++)  h2Osw[k] = h[k];
        }
        */
        //radiatif_util::outputVect("h2Osw",nAtmoLevels,h2Osw);
        //const double* h2Osw = hSW;
        
        //const double* h2Osw = pfl.h2o_sw;
        //const double* h = pfl.humidity;
        
        const double  O3 = pfl.o3_sum;
        const double* o3 = pfl.o3_sw;
        const double* press = pfl.press;
        
        const double* Ta = T+1;                             //  la première température est la température de surface
#ifdef resetSW
        for (int k=0; k<=nAtmoLevels; k++)
            sw[k] = 0.;
#endif //reset
        
/*#ifdef not_const_sw
        if (options.usingH2O)   set_sw_h2o( sw, alb, h2Osw, cosZ, sabar(O3,alb,cosZ) );
        if (options.usingO3)    set_sw_o3( sw, alboz(alb, cosZ), o3, cosZ );
#else*/
        if (options.usingH2O)   set_sw_h2o( sw, alb, h2Osw, cosZ, sabar(O3,alb,cosZ), b );
        if (options.usingO3)    set_sw_o3( sw, alboz(alb, cosZ), o3, cosZ, b );
//#endif
        for (int k=0; k<=nAtmoLevels; k++)  sw[k] *= solar;     //  on multiplie par le forçage solaire
            
        if (dswdT && (options.do_H_scaling == full_scaling || options.use_Rh) )      //  on calcule aussi les dérivées
        {
//#ifdef not_const_sw
//            set_dsw_dh( solar, alb, h2Osw, cosZ );              //  -> set the derivatives dswdh (on multiplie aussi ici par solar)
//#else
            //set_dsw_dh( solar, alb, h2Osw, cosZ, dsw_dh );      //  -> set the derivatives dswdh (on multiplie aussi ici par solar)
            set_dsw_dh( solar, alb, h2Osw, cosZ, b );      //  -> set the derivatives dswdh (on multiplie aussi ici par solar)
//#endif
            
#ifdef resetSW
            for (int i=0; i<=nAtmoLevels; i++)
                dswdT[(nAtmoLevels+1)*i] = 0.;
#endif
            for (int k=0; k<nAtmoLevels; k++)
            {   double dHdT_k = 0;

                if (options.do_H_scaling == full_scaling)
                    dHdT_k += -0.45 * h2Osw[k]/Ta[k];                 //  dh/dT
                if (options.use_Rh)
                    dHdT_k += h2Osw[k]*physics::dlogew_kelvin(Ta[k]); //  dh/dT
                for (int i=0; i<=nAtmoLevels; i++)
#ifdef resetSW
                    dswdT[1+k+(nAtmoLevels+1)*i] = b(i,k)*dHdT_k;
#else
                    dswdT[1+k+(nAtmoLevels+1)*i] += b(i,k)*dHdT_k;
#endif //resetSW
            }
        }
        
        //          fixed albedo !!
        //  set_dsw_da( dswda.firstPtr(), eps, alb, O3, o3, h2Osw, cosZ );      //  dSW/dT due to albedo as a function of T
    };
};







///////////////////////////////////////////////
//
//      Long Wave model
//
///////////////////////////////////////////////

struct lwOptions {
    scalingType do_H_scaling;       //  re-scale H2O profile
    scalingType do_C_scaling;       //  re-scale CO2 profile
    int  continuum_absorption;      //  0=no water vapor continuum absorption, 1=Rodgers & Walshaw 1966 constant 0.1 absorption (default), 2=e-type absorption, cf Stephens 1984 (to come)
    int  homogeneouspathmethod;     //  0=consider the path is homogeneous->no correction; 1=pressure scaling; 2=VCG approximation
    double  outputCoef;             //  1.0:W/m2  - 1/Sover4 = dimensionless
    bool useRh;
    
    constexpr lwOptions( int h, int c, int ca, int hp, double coef, bool rh ):
        do_H_scaling(static_cast<scalingType>(h)),do_C_scaling(static_cast<scalingType>(c)),continuum_absorption(ca),homogeneouspathmethod(hp),outputCoef(coef),useRh(rh) {};
};

static constexpr lwOptions    defaultLwOptions = lwOptions(scalingType::fixedT_scaling,scalingType::fixedT_scaling,1,0,1.,false);
    //lwOptions( do_H_scaling=fixedT_scaling, do_C_scaling=fixedT_scaling, continuum_absorption=1, homogeneouspathmethod=0, outputCoef=1., useRh=false );


class interpolLookUpTableFunction   {             //	interpolation, more explicit ...
private:
    inline long		static	Round( const double* a )	{	return _mm_cvttsd_si32( _mm_load_sd(a) );	};      //  fast Round function...
    
    //			__m128i _mm_cvttpd_epi32 (__m128d a);		//	2 double -> 2 int
    //	extern	__m128i _mm_cvtps_epi32 (__m128 a);			//	4 float -> 4 int	???
    
    //	extern  _mm256_cvttps_epi32(__m256 m1);			//	8 float -> 8 int	???
    //	extern  _mm256_cvttpd_epi32(__m256d m1);		//	4 double -> 4 int	???
    
    double x_start;
    double x_end;
    double stp;
    unsigned long n;
    
    radiatif_util::doubleFunc	f;
    
    //double* y;
    double* a;
    double* b;
    bool clean_up_at_end;
    
public:
    interpolLookUpTableFunction( double x0, double x1, double s, const double* ta, const double* tb, radiatif_util::doubleFunc ff )
    : x_start(x0), x_end(x1), stp(s), clean_up_at_end(false)
    {   n = 1 + round( (x_end-x_start)/s );
        f = ff;
        a = (double*)ta;
        b = (double*)tb;
    }
    interpolLookUpTableFunction( double x0, double x1, double s, radiatif_util::doubleFunc ff )
    : x_start(x0), x_end(x1), stp(s), clean_up_at_end(true)
    {	n = 1 + round( (x_end-x_start)/s );
        //y = new double[n+1];
        a = new double[n];
        b = new double[n];
        init(ff);
/*
        std::ofstream outfile("file.out");
        outfile << std::setprecision(18);
        
        outfile << "a[] = {";
        for (int i=0; i<n-1; i++)
        {   outfile << a[i] << ",";
            if (i%10==9)    outfile << "\n";
        }
        outfile << a[n-1] << "};\n";
        outfile << "b[] = {";
        for (int i=0; i<n-1; i++)
        {   outfile << b[i] << ",";
            if (i%10==9)    outfile << "\n";
        }
        outfile << b[n-1] << "};\n";
*/
    };
    ~interpolLookUpTableFunction()
    {   if (clean_up_at_end)
        {   delete a; delete b;   }
    };
    /*
    void init( radiatif_util::doubleFunc ff )       //  y[] inutile !!
    {	f = ff;
        double  x = x_start;
        //y[0] = f(x);
        double  y0 = f(x);
        for (int i=1; i<=n; i++)
        {	x += stp;
            
            //y[i] = f(x);         //  y[0] = f(x_start]  &   y[n-1] = f[x_start + (n-1)*stp] = f[x_end]
            //a[i-1] = (y[i]-y[i-1])/stp;
            //b[i-1] = (x*y[i-1] - (x-stp)*y[i])/stp;
            
            double yi = f(x);
            a[i-1] = (yi-y0)/stp;
            b[i-1] = (x*y0 - (x-stp)*yi)/stp;
            y0 = yi;
        };
    };*/
    inline virtual double	operator()( const double t ) const
    {	double  z = t/stp + .4999999;
        long    klo = Round(&z);
        if (klo >= n || klo < 0)	return (*f)(t);     //  out of tabulated range -> use true function
        return	t*a[klo] + b[klo];
    };
    
    void init( radiatif_util::doubleFunc ff )
    {   f = ff;
        double  x0 = x_start;
        double  y0 = f(x0);
        double  x1 = x0;
        double  y1;
        for (int i=1; i<=n; i++)
        {   x1 += stp;
            y1 = f(x1);                              //  y[0] = f(x_start]  &   y[n-1] = f[x_start + (n-1)*stp] = f[x_end]
            a[i-1] = (y1-y0)/stp;
            b[i-1] = (x1*y0 - x0*y1)/stp;
            y0 = y1;
            x0 = x1;
        };
    };
};


//#define new_storage

class irBandModel {
    
private:
    static constexpr size_t	irBands_size    = 23;
    
public:
    struct irband_vector {
        double v[irBands_size+1];   //  =[24]
    };
    
private:
    //static constexpr size_t	irBands_size    = 23;
    static constexpr double	irBands_start[] = {0, 40, 160, 280, 380, 500, 600, 667, 720, 800, 837, 900, 1000, 1200, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2200};
    static constexpr double	irBands_end[]   = {40, 160, 280, 380, 500, 600, 667, 720, 800, 837, 900, 1000, 1200, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2200, 1000000};
    static constexpr double	irBands_co2_k[] = {0, 0, 0, 0, 0, 0, 0, 653.8, 653.8, 653.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr double	irBands_co2_a[] = {0, 0, 0, 0, 0, 0, 0, 0.129, 0.129, 0.129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr double	irBands_h2o_k[] = {579.75, 7210.3, 6024.8, 1614.1, 139.03, 21.64, 2.919, 2.919, 0.3856, 0.0715, 0.0715, 0.0209, 0, 12.65, 134.4, 632.9, 331.2, 434.1, 136., 35.65, 9.015, 1.529, 0};
    static constexpr double	irBands_h2o_a[] = {0.093, 0.182, 0.094, 0.081, 0.08, 0.068, 0.06, 0.06, 0.059, 0.067, 0.067, 0.051, 0, 0.089, 0.23, 0.32, 0.296, 0.452, 0.359, 0.165, 0.104, 0.116, 0};
    static constexpr double	beta            = 1.66;
    
    //static constexpr double    irBands_co2_k_beta[] = beta*irBands_co2_k[];
    
    static double tauC( int k, double x )
    {	double c = irBands_co2_k[k]*x*beta;
        return (irBands_co2_k[k]>0 ? c/sqrt(1+c/irBands_co2_a[k]) : 0);
    };
    static double  exptauC( int k, double x )
    {	return	exp( -tauC( k, x ) );
    };
    static double tauH( int k, double x )
    {	double h = irBands_h2o_k[k]*x*beta;
        return (irBands_h2o_k[k]>0 ? h/sqrt(1+h/irBands_h2o_a[k]) : 0);
    };
    static double dtau_dH( int k, double h )        //    attention i=0,22 !!
    {   double ax = irBands_h2o_k[k]*beta;
        double bx = irBands_h2o_a[k];
        return (irBands_h2o_k[k]>0 ? ax*(2*bx+ax*h)/sqrt(1.+ax*h/bx)/(bx+h*ax)/2. : 0);
    };
    static double tau_continuum_RW( double uH )		//Absorption dans le continuum de la vapeur d'eau par la methode grossiere de Rodgers & Walshaw 1966 (p74)
    {	return 0.1*uH*beta; };
    static double  exptauH_continuum( int k, double x )
    {	if (k>=9 && k<=12)	return	exp( -tauH( k, x ) - tau_continuum_RW( x ) );
        else				return	exp( -tauH( k, x ) );
    };
    static double  dexptauH_dH_continuum( int k, double x )
    {   if (k>=9 && k<=12)  return    (-dtau_dH( k, x ) - 0.1*beta) * exp( -tauH( k, x ) - tau_continuum_RW( x ) );
        else                return    -dtau_dH( k, x ) * exp( -tauH( k, x ) );
    };
    static double dtau_dC( int k, double h )
    {   double ax = irBands_co2_k[k]*beta;
        double bx = irBands_co2_a[k];
        return (irBands_co2_k[k]>0 ? ax*(2*bx+ax*h)/sqrt(1.+ax*h/bx)/(bx+h*ax)/2. : 0);
    };
public:
    //static void setexptau_with_continuum( double* v, double uC, double uH )
    //{	for (int k=0; k<irBands_size; k++)	*(v++) = exptauC( k, uC ) * exptauH_continuum( k, uH );             //	k=0, ..., k=22
   // };
    static void setexptau_with_continuum( irband_vector& ir_v, double uC, double uH )
    {   for (int k=0; k<irBands_size; k++)	ir_v.v[k] = exptauC( k, uC ) * exptauH_continuum( k, uH );          //	k=0, ..., k=22
    };
    static void setexptau_dH_with_continuum( irband_vector& ir_v, double uC, double uH )
    {   for (int k=0; k<irBands_size; k++)  ir_v.v[k] = exptauC( k, uC ) * dexptauH_dH_continuum( k, uH );      //    k=0, ..., k=22
    };
    static void setexptau_dC_with_continuum( irband_vector& ir_v, double uC, double uH )
    {   for (int k=0; k<irBands_size; k++)  ir_v.v[k] = -dtau_dC( k, uC ) * exptauC( k, uC ) * exptauH_continuum( k, uH );        };

    static constexpr double	ctePlck = 0.1539897338;		//	= 15/Pi^4
    static constexpr double	cte = 1.43878;              //	= 100 (h c)/k
    
    static double planck( double x )                {	return ctePlck * x*x*x/(exp(x)-1);	};
    //static constexpr double planck( double x )		{	return ctePlck * x*x*x/(exp(x)-1);	};
    
    static constexpr double iPlanckS( double x )		//	ok si x > 2 ou 3  (Polylogarithm series truncation, Eq. (38) in Clark J.Comput.Phys. 1987)
    {	double s = 0;
        double x2 = x*x;
        double x3 = x2*x;
#ifdef use_compile_time_table
        for (int i=1; i<16; i++)	s += (x3/i + 3.*x2/(i*i) + 6.*x/(i*i*i) + 6./(i*i*i*i))*gcem::exp(-i*x);
#else
        for (int i=1; i<16; i++)    s += (x3/i + 3.*x2/(i*i) + 6.*x/(i*i*i) + 6./(i*i*i*i))*exp(-i*x);
#endif
        return 1. - ctePlck*s;
    };
    static constexpr double iPlanckX( double x )		//	ok si x < 1 ou 2 (Formule (32) dans Clark J.Comput.Phys. 1987)
    {	double c[] = {1./3, 1./60, -1./5040, 1./272160, -1./13305600, 1./622702080, -691./19615115520000., 1./1270312243200., -3617./202741834014720000.,                                   43867./107290978560589824000., -174611./18465726242060697600000., 77683./352527500984795136000000.};
        double x2 = x*x;
        double xn = x;		//	on commence par x3
        double s = -x2*x2/8;	//	le seul xn avec n pair
        for (int n=0; n<12; n++)	{	xn = xn*x2;		s += xn*c[n];	}
        return	ctePlck*s;
    };
    static constexpr double primitivePlanck( double x )		//	integrale de planck(x)
    {	if (x>1.9)	return iPlanckS( x );
        if (x<1.8)	return iPlanckX( x );
        return std::min( iPlanckS( x ), iPlanckX( x ) );
    };/*
    static double dotProduct( const double* x, const double* y )
    {	double	s = 0;
        for (int i=0; i<irBands_size; i++)	s += (*x++) * (*y++);
        return s;
    };*/
    static double dotProduct( const irband_vector& x, const irband_vector& y )
    {	double	s = 0;
        for (int i=0; i<irBands_size; i++)	s += x.v[i] * y.v[i];
        return s;
    };
    class integPlanckLookupTable : public interpolLookUpTableFunction {
    public:
        //integPlanckLookupTable():interpolLookUpTableFunction( 0., 20., .0005, irBandModel::primitivePlanck ) {};
        static const double tab_a[];
        static const double tab_b[];
        integPlanckLookupTable():interpolLookUpTableFunction( 0., 20., .0005, tab_a, tab_b, irBandModel::primitivePlanck )  {};
    };
    static const integPlanckLookupTable  approxIntegPlanck;
    //static const interpolLookUpTableFunction  approxIntegPlanck;// = interpolLookUpTableFunction( 0., 20., .0005, irBandModel::primitivePlanck );
    
    static void	setPlanckVector( double T, irband_vector& ir_v )	//	v[0 .. irBands_size]
    {	double a = (cte/T);
        ir_v.v[0] = 0.;
        for (int k=1; k<irBands_size; k++)		ir_v.v[k] = planck( a*irBands_start[k] );
        ir_v.v[irBands_size] = 0.;
    }
    
    static void	setIntegPlanckVector( double T, irband_vector& ir_v )	//	v[0 .. irBands_size-1]
    {	double a = (cte/T);
        irband_vector prim;
        prim.v[0] = 0.;
        for (int k=1; k<irBands_size; k++)		prim.v[k] = approxIntegPlanck( a*irBands_start[k] );
        for (int k=0; k<irBands_size-1; k++)	ir_v.v[k] = (prim.v[k+1] - prim.v[k]);
        ir_v.v[irBands_size-1] = (1. - prim.v[irBands_size-1]);
    }
    
    static void	setDIntegPlanckVector( double T, irband_vector& ir_v )	//	v[0 .. irBands_size-1]
    {	double a = (cte/T);
        double asT = -(a/T);
        irband_vector prim;
        prim.v[0] = 0.;
        for (int k=1; k<irBands_size; k++)      prim.v[k] = irBands_start[k] * planck( a*irBands_start[k] );
        for (int k=0; k<irBands_size-1; k++)	ir_v.v[k] = (prim.v[k+1] - prim.v[k])*asT;
        ir_v.v[irBands_size-1] = -prim.v[irBands_size-1]*asT;
    }
};

 //const interpolLookUpTableFunction  irBandModel::approxIntegPlanck = interpolLookUpTableFunction( 0., 20., .0005, irBandModel::primitivePlanck );


template <typename T>
class array_2d {
public:
    T *first, **a;
    array_2d( size_t n, size_t m )   {
        first = new T[n * m];
        a = new T*[n];
        for (size_t i=0; i<n; i++)  a[i] = first + i*m;
    }
    array_2d()
    {   delete first;   delete a;   };
    inline T*&      operator[] (size_t posx)        {	return a[posx]; }
    inline const T* operator[] (size_t posx) const  {	return a[posx];	}
};

typedef  array_2d<double>                       double_array_2d;
typedef  array_2d<irBandModel::irband_vector>   irband_array_2d;

//#define use_c_array

class lwModel   {
    
    /////////////   Partie statique     ///////////
    
private:
    //static constexpr double	ctePlck = 0.1539897338;		//	= 15/Pi^4
    //static constexpr double	cte = 1.43878;              //	= 100 (h c)/k
    //static constexpr double	sigmaStefan = 5.6704e-8;    //	= sigma
    static constexpr double sigmaStefan = physics::sigmaStefan;    //    = sigma
    
    

#ifdef use_c_array
    static void set_B_mat( size_t n, int kc, double b[n][n+1], const irBandModel::irband_vector ip[n], const irBandModel::irband_vector expT[n-1][n] )   //  expT[n-1][n]
    {   double  c[n][n];
        double  d[n][n];
#else
    static void set_B_mat( size_t n, int kc, double_array_2d& b, const irBandModel::irband_vector ip[], const irband_array_2d expT )   //  expT[n-1][n]
    {   double_array_2d c(n,n);
        double_array_2d d(n,n);
#endif
        
        double constante = (kc==0 ? 1. : 0.);   //  par défaut (?) kc = 0 => la matrice B, sinon (kc<0 ou kc>0), c'est une dérivée...
        bool noBlock = (kc<=0);                 //  !noBlock (kc>0), dérivée par rapport à expT(h2o ou co2). Dérivée par rapport à T (ip) si kc = -1
        //size_t k1 = kc-1;
        
        //	C mat
        for (int i=0; i<n; i++)		c[i][i] = constante;				//	i==j
        
        for (int i=1; i<n; i++)			//	0<=j<i<=n
            for (int j=0; j<i; j++)
                if (noBlock || (i>=kc && j+1<=kc))
                    c[i][j] = irBandModel::dotProduct( expT[i-1][j], ip[i] );           //  j<=i-1
                else
                    c[i][j] = 0.;
        //outputVect("c", temp_double_vector1(c.firstPtr(),c.size()));
        
        for (int i=0; i<n-1; i++)				//	j>i
            for (int j=i+1; j<n; j++)
                if (noBlock || (j>=kc && i+1<=kc))
                    c[i][j] = irBandModel::dotProduct( expT[j-1][i], ip[i] );          //  i<=j-1
                else
                    c[i][j] = 0.;
        //outputVect("c", temp_double_vector1(c.firstPtr(),c.size()));
        
        //	D mat
        for (int j=0; j<n; j++)		d[0][j] = 0;				//	i=0
        for (int i=1; i<n; i++)		d[i][i-1] = constante;      //	j=i-1
        for (int i=1; i<n; i++)		d[i][i] = c[i][i-1];        //	j=i>0
        
        for (int i=1; i<n; i++)			//	j<i-1
            for (int j=0; j<i-1; j++)
                if (noBlock || (i-1>=kc && j+1<=kc))
                    d[i][j] = irBandModel::dotProduct( expT[i-2][j], ip[i] );             //  j<=i-2
                else
                    d[i][j] = 0.;
        //outputVect("d", temp_double_vector1(d.firstPtr(),c.size()));
        
        for (int i=1; i<n; i++)			//	j>i
            for (int j=i+1; j<n; j++)
                if (noBlock || (j>=kc && i<=kc))
                    d[i][j] = irBandModel::dotProduct( expT[j-1][i-1], ip[i] );            //  i-1<=j-2<j-1
                else
                    d[i][j] = 0.;
        //outputVect("d", temp_double_vector1(d.firstPtr(),c.size()));
        
        //	B mat
        for (int i=0; i<n; i++)
        {   for (int j=1; j<n; j++)
                b[i][j] = (c[i][j]-c[i][j-1]) - (d[i][j]-d[i][j-1]);
            b[i][0] = c[i][0] - d[i][0];
            b[i][n] = -c[i][n-1] + d[i][n-1];
            b[i][i] = 0.;
        }
        
    };
        
#ifdef use_c_array
    static void set_L_mat( size_t n, int kc, double l[n][n], const irBandModel::irband_vector ip[n], const irBandModel::irband_vector expT[n-1][n] )
    {   double  b[n][n+1];
#else
        //  Nouvelle formulation ....
    static void set_L_mat( size_t n, int kc, double_array_2d l, const irBandModel::irband_vector ip[], const irband_array_2d expT )   // l[n][n]   const irBandModel::irband_vector expT[n-1][n] )
    {   //size_t n = ip.dim1();
        //double_matrix00 b(n,n+1);
        //double b[n][n+1];
        double_array_2d b(n,n+1);
#endif
        
        set_B_mat( n, kc, b, ip, expT );
        
        for (int i=0; i<n; i++)
            for (int j=0; j<n; j++)	l[i][j] = -b[j][i];  //l(i,j) = -b(j,i);			//	0<=i,j<=n
        
        for (int i=0; i<n; i++)
        {	double s = 0;
            for (int k=0; k<n+1; k++)	s += b[i][k];   //b(i,k);				//	0<=i<=n	& 0<=k<=n+1
            l[i][i] += s;
        }
    };
/*
    //  on ne remplit que les valeurs de t[i][j] pour j<=i
    static void set_exptau_tensor( size_t n_atmo, irband_array_2d t, const double uC[n_atmo], const double uH[n_atmo] )       //  t[n_atmo,n_atmo+1]
    {	//size_t n = uH.length();
        for (int i=0; i<n_atmo; i++)
        {	double sC = 0;
            double sH = 0;
            for (int j=i; j>=0; j--)
            {	sC += uC[j];	sH += uH[j];
                irBandModel::setexptau_with_continuum( t[i][j], sC, sH );         //  par defaut :   setexptau_with_continuum( &(t(i,j-1,0)), sC, sH );
        }	}
    };
*/
    /////////////   Partie dynamique     ///////////
#ifdef new_storage
    /////////////   buffer     ///////////
        //  //  allocation en début de call, par
        //
        //  irBandModel::irband_vector temp_ir[lw_buffer::irband_size(n)];   //  ip, dip,...
        //  double temp_d[lw_buffer::double_size(n)];
        //  lw_buffer b(n,temp_ir,temp_d);
        //
    class lw_buffer {
    private:
        size_t  n_atm;
        irBandModel::irband_vector  *_ip, *_dip;                    //  [nAtmoLevels+1]
        irBandModel::irband_vector  *_expT, *_expT_dC, *_expT_dH;   //  [nAtmoLevels,nAtmoLevels+1]
        double* _l;                                                 //  [nAtmoLevels+1,nAtmoLevels+1];
    public:
        static size_t irband_size( size_t n )    {   return 2*(n+1)+3*n*(n+1);  };
        static size_t double_size( size_t n )    {   return (n+1)*(n+1);        };
        lw_buffer( size_t n, irBandModel::irband_vector* tir, double* td ):n_atm(n)
        {   _ip = tir; _dip = tir+n+1;  _expT = tir+2*(n+1); _expT_dC = tir+2*(n+1)+n*(n+1); _expT_dH = tir+2*(n+1)+2*n*(n+1);  _l = td;    };
        
        irBandModel::irband_vector& temp_ip( size_t i )     {   return _ip[i];  };
        irBandModel::irband_vector& temp_dip( size_t i )    {   return _dip[i]; };
        irBandModel::irband_vector& temp_expT( size_t i, size_t j )     {   return _expT[i*n_atm+j];    };
        irBandModel::irband_vector& temp_expT_dC( size_t i, size_t j )  {   return _expT_dC[i*n_atm+j]; };
        irBandModel::irband_vector& temp_expT_dH( size_t i, size_t j )  {   return _expT_dH[i*n_atm+j]; };
        double& temp_l( size_t i, size_t j )    {   return _l[i*(n_atm+1)+j]; };
        
    private:
        //  on ne remplit que les valeurs de t[i][j] pour j<=i
        void set_exptau_tensor( bool with_dC, bool with_dH, const double *uC, const double *uH )       //  t[n_atmo,n_atmo+1]
        {    //size_t n = uH.length();
            for (int i=0; i<n_atm; i++)
            {   double sC = 0;
                double sH = 0;
                for (int j=i; j>=0; j--)
                {   sC += uC[j];    sH += uH[j];
                    irBandModel::setexptau_with_continuum( temp_expT(i,j), sC, sH );           //  par defaut :   setexptau_with_continuum( &(t(i,j-1,0)), sC, sH );
                    if (with_dC)
                        irBandModel::setexptau_dC_with_continuum( temp_expT_dC(i,j), sC, sH );
                    if (with_dH)
                        irBandModel::setexptau_dH_with_continuum( temp_expT_dH(i,j), sC, sH );
            }    }
        };
        //  [nAtmoLevels+1,nAtmoLevels+1];
    public:
        void set_L_mat( bool with_dC, bool with_dH, const double *T, const double *uC, const double *uH )
        {   set_exptau_tensor( with_dC, with_dH, uC, uH );     //  computes temp_expT, temp_expT_dC, temp_expT_dH
            for (size_t i=0; i<=n_atm; i++)
                irBandModel::setIntegPlanckVector( T[i], temp_ip(i) );
            set_L_mat( n_atm+1, 0, temp_l, temp_ip, temp_expT );
        }
    };
#endif
    
private:
    size_t      nAtmoLevels;
    lwOptions   options;

#ifndef new_storage
        //  temporary storage
    irBandModel::irband_vector  *temp_ip, *temp_dip;   //  [nAtmoLevels+1]
    double_array_2d             temp_l;         //  [nAtmoLevels+1,nAtmoLevels+1];
    irband_array_2d             temp_expT;      //  [nAtmoLevels,nAtmoLevels+1]
    irband_array_2d             temp_expT_dC;   //  [nAtmoLevels,nAtmoLevels+1]
    irband_array_2d             temp_expT_dH;   //  [nAtmoLevels,nAtmoLevels+1]
    
    //  on ne remplit que les valeurs de t[i][j] pour j<=i
    void set_exptau_tensor( size_t n_atmo, bool with_dC, bool with_dH, const double uC[], const double uH[] )       //  t[n_atmo,n_atmo+1]
    {    //size_t n = uH.length();
        for (int i=0; i<n_atmo; i++)
        {   double sC = 0;
            double sH = 0;
            for (int j=i; j>=0; j--)
            {   sC += uC[j];    sH += uH[j];
                irBandModel::setexptau_with_continuum( temp_expT[i][j], sC, sH );           //  par defaut :   setexptau_with_continuum( &(t(i,j-1,0)), sC, sH );
                if (with_dC)
                    irBandModel::setexptau_dC_with_continuum( temp_expT_dC[i][j], sC, sH );
                if (with_dH)
                    irBandModel::setexptau_dH_with_continuum( temp_expT_dH[i][j], sC, sH );
        }    }
    };
#endif
    
    
public:
#ifdef new_storage
    lwModel( size_t n, lwOptions opt=defaultLwOptions ):nAtmoLevels(n),options(opt){};
    ~lwModel(){};
#else
    lwModel( size_t n, lwOptions opt=defaultLwOptions ):nAtmoLevels(n),options(opt),
        temp_l(nAtmoLevels+1,nAtmoLevels+1),
        temp_expT(nAtmoLevels,nAtmoLevels+1),temp_expT_dC(nAtmoLevels,nAtmoLevels+1),temp_expT_dH(nAtmoLevels,nAtmoLevels+1)
    {   temp_ip  = new irBandModel::irband_vector[nAtmoLevels+1];
        temp_dip = new irBandModel::irband_vector[nAtmoLevels+1];
    };
    ~lwModel()
    {   delete temp_ip; delete temp_dip; };
#endif
    
    void setLW( double* lw, double* dlwdT, const double* T, const double* uH, const double* uC, StdProfile& pfl ) //const
    {
   /*     setLW( lw, dlwdT, T, pfl.humidity, pfl.h2o_lw, pfl.co2_lw, pfl.press, pfl.press_bot );
    };

    void setLW( double* lw, double* dlwdT, const double* T, const double* h, const double* hLW, const double* vc, const double* press, const double* press_bottom )
    //void setLW( double* lw, double* dldT, const double_vector0& T, const double_vector& h, const double_vector& hLW, const double_vector& vc, const double_vector0& press, const double_vector1& deltapress )
    {*/
        //const double* uH = hLW;
        //const double* uC = vc;
        //const double* h = pfl.humidity;
        
        //const double* uH = pfl.h2o_lw;
        //const double* uC = pfl.co2_lw;
        const double* press = pfl.press;
        const double* press_bottom = pfl.press_bot;
        
        
        double output_coef = sigmaStefan * options.outputCoef;
        const double  *Ta = T+1;
        
        //double  uH[nAtmoLevels];
        //double  uC[nAtmoLevels];
        double  sigT4[nAtmoLevels+1];
    
        for (int i=0; i<=nAtmoLevels; i++)
            sigT4[i] = output_coef * radiatif_util::square(radiatif_util::square(T[i]));
        /*
        if (options.do_H_scaling == full_scaling)           //   && homogeneouspathmethod == 0)
        {   for (int k=0; k<nAtmoLevels; k++)
                uH[k] = h[k] * pow( press[k]/1013., 0.75 ) * pow( fabs(273./Ta[k]), 0.45 );
        }
        else if (options.do_H_scaling == fixedT_scaling)    //   && homogeneouspathmethod == 0)
        {   for (int k=0; k<nAtmoLevels; k++)   uH[k] = hLW[k];
        }
        else        //  == no_scaling
        {   for (int k=0; k<nAtmoLevels; k++)   uH[k] = h[k];
        }
         
        static constexpr double c_cte = (10./physics::g)*(physics::Mco2/physics::Mair)*1.e-6;
        if (options.do_C_scaling == full_scaling)
        {   for (int k=0; k<nAtmoLevels; k++)
                uC[k] = 280 * c_cte * (press_bottom[k]-press_bottom[k+1]) * pow( press[k]/1013., 1.75 ) * pow( fabs(273./Ta[k]), 8 );       //  ATTENTION 280 ppm
        }
        else        //  attention: ici scaling sur T/P McClatchey ?
        {   for (int k=0; k<nAtmoLevels; k++)   uC[k] = vc[k];   }
        */
        
        //radiatif_util::outputVect("sigT4",nAtmoLevels+1,sigT4);
        //radiatif_util::outputVect("uH",nAtmoLevels,uH);
        //radiatif_util::outputVect("uC",nAtmoLevels,uC);
        
        bool need_dH = (dlwdT)&&(options.do_H_scaling == full_scaling || options.useRh);
        bool need_dC = (dlwdT)&&(options.do_C_scaling == full_scaling);
        
#ifdef new_storage
        
        irBandModel::irband_vector temp_ir[lw_buffer::irband_size(nAtmoLevels)];   //  ip, dip,...
        double temp_d[lw_buffer::double_size(nAtmoLevels)];
        lw_buffer b(nAtmoLevels,temp_ir,temp_d);
        
        b.set_exptau_tensor( nAtmoLevels, need_dC, need_dH, uC, uH );     //  computes temp_expT, temp_expT_dC, temp_expT_dH
        for (int i=0; i<=nAtmoLevels; i++)
            irBandModel::setIntegPlanckVector( T[i], b.temp_ip(i) );
        set_L_mat( nAtmoLevels+1, 0, temp_l, temp_ip, temp_expT );
#else
        set_exptau_tensor( nAtmoLevels, need_dC, need_dH, uC, uH );     //  computes temp_expT, temp_expT_dC, temp_expT_dH
        
        for (int i=0; i<=nAtmoLevels; i++)
            irBandModel::setIntegPlanckVector( T[i], temp_ip[i] );
        set_L_mat( nAtmoLevels+1, 0, temp_l, temp_ip, temp_expT );
#endif
        
        for (int i=0; i<=nAtmoLevels; i++)
        {
#ifdef resetLW
            lw[i] = 0.;
#endif //reset
            for (int k=0; k<=nAtmoLevels; k++)
                lw[i] += temp_l[i][k]*sigT4[k];
        }
        
        if (dlwdT)      //  on calcule aussi les dérivées
        {   for (int i=0; i<=nAtmoLevels; i++)
            for (int k=0; k<=nAtmoLevels; k++)
#ifdef resetLW
                dlwdT[k+(nAtmoLevels+1)*i] = 4*temp_l[i][k]*sigT4[k]/T[k];
#else
                dlwdT[k+(nAtmoLevels+1)*i] += 4*temp_l[i][k]*sigT4[k]/T[k];
#endif //reset
            
            for (int i=0; i<=nAtmoLevels; i++)
                irBandModel::setDIntegPlanckVector( T[i], temp_dip[i] );
            set_L_mat( nAtmoLevels+1, -1, temp_l, temp_dip, temp_expT );     //  Attention !! écrase temp_l (->dérivée / dT)
            for (int i=0; i<=nAtmoLevels; i++)
            for (int k=0; k<=nAtmoLevels; k++)
                dlwdT[k+(nAtmoLevels+1)*i] += temp_l[i][k]*sigT4[k];
            
            /*
            std::cout << "dlw_dx (r2) = ";
            for (int i=0; i<5; i++)
            for (int j=0; j<5; j++)
                std::cout << dlwdT[5*i+j]*(-T[j]*T[j]/278.6782678962288)/342 << ", ";      //  solar???
            std::cout << "\n";
            */
            
            if (options.do_C_scaling == full_scaling)
            {   double  dlw_dc[nAtmoLevels][nAtmoLevels+1];
                for (size_t kc=0; kc<nAtmoLevels; kc++)
                {   set_L_mat( nAtmoLevels+1, kc+1, temp_l, temp_ip, temp_expT_dC );
                    for (int i=0; i<=nAtmoLevels; i++)
                    {   dlw_dc[kc][i] = 0;
                        for (int j=0; j<=nAtmoLevels; j++)
                            dlw_dc[kc][i] += temp_l[i][j]*sigT4[j];
                    }
                    double dCdT_k = -8 * uC[kc]/Ta[kc];                 //  dc/dT
                    for (int i=0; i<=nAtmoLevels; i++)
                    {   dlwdT[1+kc+(nAtmoLevels+1)*i] += dlw_dc[kc][i]*dCdT_k;
                        //std::cout << (dlw_dc[kc][i]*dCdT_k)*(-T[kc+1]*T[kc+1]/278.6782678962288)/342 << ", ";
                    }
                }
                
            }
            
            if (options.do_H_scaling == full_scaling || options.useRh)
            {   double  dlw_dh[nAtmoLevels][nAtmoLevels+1];
                for (size_t kc=0; kc<nAtmoLevels; kc++)
                {   set_L_mat( nAtmoLevels+1, kc+1, temp_l, temp_ip, temp_expT_dH );
                    for (int i=0; i<=nAtmoLevels; i++)
                    {   dlw_dh[kc][i] = 0;
                        for (int j=0; j<=nAtmoLevels; j++)
                            dlw_dh[kc][i] += temp_l[i][j]*sigT4[j];
                    }
                    
                    double dHdT_k = 0;
                    if (options.do_H_scaling == full_scaling)
                        dHdT_k += -0.45 * uH[kc]/Ta[kc];                 //  dh/dT
                    if (options.useRh)
                        dHdT_k += uH[kc]*physics::dlogew_kelvin(Ta[kc]);                 //  dh/dT
                    for (int i=0; i<=nAtmoLevels; i++)
                        dlwdT[1+kc+(nAtmoLevels+1)*i] += dlw_dh[kc][i]*dHdT_k;
                    
                }
                
            }
        }
    }
};








///////////////////////////////////////////////
//
//      Radiative model = routine to be used
//
///////////////////////////////////////////////



class radiativeModel {
private:
    StdProfile  std_profile;    //  ( 1, na, 1013.25, stdProfOptions(HswScaling,HlwScaling,ClwScaling,useRh), pCO2 );
    lwModel     lw_model;       //  (na, lwOptions(HlwScaling,ClwScaling,1,0,1.,useRh));
    swModel     sw_model;       //  (na, swOptions(true,true,HswScaling,false,useRh));
    
    static constexpr double  Sover4 = (physics::solarConstant/4);
public:
        //  local values...
    //static constexpr double  p_surface = 1013.25;
    static constexpr double  albedo = 0.1;
    static constexpr double  cosZ = 0.25;
    static constexpr double  eps = 1.00013951364302;
    static constexpr double  latitude = 0.;     //  not used when std_prof_no < 8
    
#if 0       //  output in W/m2
    static constexpr double  sw_coef = Sover4;
    static constexpr double  lw_coef = 1.0;
#else       //  dimensionless output
    static constexpr double  sw_coef = 1.0;
    static constexpr double  lw_coef = 1.0/Sover4;
#endif
    
#ifdef use_compile_time_table
    static constexpr double  Tref = gcem::pow(Sover4/physics::sigmaStefan,0.25);
#else
    static const double  Tref;
#endif // use_compile_time_table
        
    
    radiativeModel( size_t na, size_t std_prof_no, double p_surface, double pCO2, bool useRh, scalingType scal ):
        std_profile( std_prof_no, na, p_surface, stdProfOptions(scal,scal,scal,useRh), pCO2, latitude ),
        lw_model(na, lwOptions(scal,scal,1,0,lw_coef,useRh)),
    //lwOptions( do_H_scaling=fixedT_scaling, do_C_scaling=fixedT_scaling, continuum_absorption=1, homogeneouspathmethod=0, outputCoef=1., useRh=false );
        sw_model(na, swOptions(true,true,scal,false,useRh))  {};
    //swOptions( useO3, useH2O, HswScaling, fast, rh)
    
    size_t size()   {   return std_profile.nAtmoLevels + 1; };
    
    void bilan_T( double* b, double* dbdT, const double* T )    //  bilan(T) et dbilan/dT     ATTENTION: dbdT==NULL ok
    {
        std_profile.set_temperature(T+1);   //  T+1 = Tatmo[] without the first value (Tsurface)
        
        //  { h, hs, hl, cl } = profile.function( Ta, rh )
        //double h[std_profile.nAtmoLevels];
        //double hs[std_profile.nAtmoLevels];
        //double hl[std_profile.nAtmoLevels];
        //double cl[std_profile.nAtmoLevels];
        //std_profile.set_hscal_from_temp_rh( T+1, std_profile.relativeH, h, hs, hl, cl );
        
//#define resetSW mais pas resetLW => setSW met à zéro, mais pas setLW
        sw_model.setSW( b, dbdT, T, std_profile.h2o_sw, std_profile, albedo, sw_coef * eps, cosZ );
        lw_model.setLW( b, dbdT, T, std_profile.h2o_lw, std_profile.co2_lw, std_profile );
    }
    
    void bilan_x( double* b, double* dbdx, const double* x )    //  bilan(x) et dbilan/dx     ATTENTION: dbdx==NULL ok
    {
        size_t n = std_profile.nAtmoLevels + 1;
        double T[n];
        for (int i=0; i<n; i++)
            T[i] = Tref/x[i];
        if (dbdx)
        {   double dbdT[n*n];
            bilan_T( b, dbdT, T );
            //std::cout << "dbdT[5] = " << dbdT[5] << "\n";
            
            for (int i=0; i<n; i++)
            //{
            for (int j=0; j<n; j++)
                dbdx[n*i+j] = (-T[j]/x[j])*dbdT[n*i+j];
                
                
                //std::cout << "dbdT[4-6] = " << dbdT[4] << ", " << dbdT[5] << ", " << dbdT[6] << "\n";
                //std::cout << "dbdx[4-6] = " << dbdx[4] << ", " << dbdx[5] << ", " << dbdx[6] << "\n";
            //}
            //std::cout << "dbdx[5] = " << dbdx[5] << "\n";
        }
        else
            bilan_T( b, NULL, T );
    }
    
    double diag_matE( size_t i )    {   return std_profile.press_pi_bot[i] - 1.;    };
    double offdiag_matE( size_t i ) {   return std_profile.press_pi_bot[i] - std_profile.press_pi_top[i];    };
    double pression( size_t i )     {   return std_profile.press[i];    };
};


#endif  //RADIATIF_H
