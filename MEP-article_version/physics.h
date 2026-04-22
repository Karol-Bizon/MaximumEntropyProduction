//
//  physics.h
//  NRmathematica
//
//  Created by Didier Paillard on 21/08/2020.
//
//

#ifndef physics_h
#define physics_h



//  constexpr_sqrt
double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
{   return (curr == prev ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr));     };
double constexpr constexpr_sqrt(double x)
{   return (x >= 0 && x < std::numeric_limits<double>::infinity() ? sqrtNewtonRaphson(x, x, 0) : std::numeric_limits<double>::quiet_NaN());  };


#include <cmath>

class physics {
public:
    
    static constexpr double    solarConstant = 1368.0;     //    = constante solaire W/m2
    static constexpr double    sigmaStefan = 5.6704e-8;    //    = sigma
    
    constexpr static const double Mco2 = 44.;           //  g.mol-1
    constexpr static const double Mair = 28.97;         //  g.mol-1
    constexpr static const double Mh2o = 18.01524;      //  g.mol-1
    constexpr static const double Rgp = 8.314472;       //
    constexpr static const double g = 9.80665;          //  m.s-2
    constexpr static const double MvolO3 = 0.002144;    //
    
    constexpr static const double kappa = 0.2854;               //  ~ 2/7
    constexpr static const double Rdryair = 287.0;              //  (J/K/kg)
    constexpr static const double cpDryAir = Rdryair/kappa;     //    = 1005.6 [J/K/kg]  au lieu de 1004.5 ???
    constexpr static const double Lwater = 2.5e6;               //  (J/kg) chaleur latente d'évaporation de l'eau

    constexpr static const double Tref = constexpr_sqrt( constexpr_sqrt( solarConstant/4/sigmaStefan ) );
    constexpr static const double LoCP = Lwater/(cpDryAir * Tref);
    
    
        //	pression de vapeur saturante (en hPa) fonction de t en Celsius
        //  voir   http://cires1.colorado.edu/~voemel/vp.html
    static double	ew_CIMO( double t )    {   return (6.112*exp(17.62*t/(243.12+t)));     };
        //      Guide to Meteorological Instruments and Methods of Observation (CIMO Guide) (WMO, 2008)
    static double	ew_Bolton( double t )  {   return (6.112*exp(17.67*t/(243.5+t)));      };
        //      Bolton 1980
    
    static double	dew_CIMO( double t )
    {   double d = 1/(243.12+t);
        return (6.112*17.62*243.12*d*d*exp(17.62*t*d));
    };
    static double	dew_Bolton( double t )
    {   double d = 1/(243.5+t);
        return (6.112*17.67*243.5*d*d*exp(17.67*t*d));
    };
    
    static double	dlogew_CIMO( double t )         //  = dew/ew
    {   double d = 1/(243.12+t);
        return (17.62*243.12*d*d);
    };
    static double	dlogew_Bolton( double t )
    {   double d = 1/(243.5+t);
        return (17.67*243.5*d*d);
    };
    
    static double   ew_celsius( double t0 )			//	pression de vapeur saturante (hPa)
    {	 //	cut-off pour les très basse températures (sinon probleme)
        double t = (t0 > 100. ? 100. : (t0 < -200. ? -200. : t0));
        return physics::ew_CIMO( t );
    };
    static double	dew_celsius( double t )         //	la dérivée
    {	if (t > 100.)   return 0.;
        if (t < -200.)  return 0.;
        return physics::dew_CIMO( t );
    };
    static double	dlogew_celsius( double t )      //	la dérivée logarithmique
    {	if (t > 100.)   return 0.;
        if (t < -200.)  return 0.;
        return physics::dlogew_CIMO( t );
    };
    
    //	idem fonction de tk en Kelvin
    
    static double	ew_kelvin( double tk )      {	return physics::ew_celsius( tk-273.15 );     };
    static double	dew_kelvin( double tk )     {	return physics::dew_celsius( tk-273.15 );    };
    static double	dlogew_kelvin( double tk )  {	return physics::dlogew_celsius( tk-273.15 ); };
    
    static double   rsat_kelvin( double t, double p )      //  saturation mixing ratio
    {   double ew = ew_kelvin(t);
        //double emax = p/1.622;
        if (ew >= p/1.622) return 1.0;      //  rsat = 1 si ew == p/1.622
        return 0.622 * ew/(p-ew);
    };
    static double   drsat_kelvin( double t, double p )      //  saturation mixing ratio
    {   double ew = ew_kelvin(t);
        if (ew >= p/1.622) return 0.0;
        double dew = dew_kelvin(t);
        return 0.622 * dew * p/(p-ew)/(p-ew);
    };
    
    static double   qsat_kelvin( double t, double p )      //  saturation humidity
    {   double ew = ew_kelvin(t);
        if (ew >= p) return 1.0;
        double qsat = 0.622 * ew/(p-0.378*ew);
        return std::min( qsat, 1.0 );
    };
    
};


#endif /* physics_h */
