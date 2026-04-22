/*
 *  radiatif.cpp
 *
 *  Created by Didier on 12/02/13.
 *  Copyright 2013-2020 LSCE. All rights reserved.
 *
 *     std = C++11
 */




#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

#include "radiatif.h"



constexpr double StdMcClatcheyProfile::pressure_table[5][33];
constexpr double StdMcClatcheyProfile::temperature_table[5][33];
constexpr double StdMcClatcheyProfile::density_table[5][33];
constexpr double StdMcClatcheyProfile::wvdensity_table[5][33];
constexpr double StdMcClatcheyProfile::o3density_table[5][33];
constexpr double StdMcClatcheyProfile::ReferenceLatitudes[7];

constexpr double irBandModel::irBands_start[];
constexpr double irBandModel::irBands_end[];
constexpr double irBandModel::irBands_co2_k[];
constexpr double irBandModel::irBands_co2_a[];
constexpr double irBandModel::irBands_h2o_k[];
constexpr double irBandModel::irBands_h2o_a[];


const irBandModel::integPlanckLookupTable  irBandModel::approxIntegPlanck;

#include "integPlanckLookupTable.cpp"
//const double irBandModel::integPlanckLookupTable::tab_a[] =
//const double irBandModel::integPlanckLookupTable::tab_b[] =


#ifndef use_compile_time_table
const double  radiativeModel::Tref = pow(Sover4/physics::sigmaStefan,0.25);
#endif // use_compile_time_table

//const interpolLookUpTableFunction  irBandModel::approxIntegPlanck = interpolLookUpTableFunction( 0., 20., .0005, irBandModel::primitivePlanck );


#ifdef use_compile_time_table
//#ifdef use_compile_time_table

class test {
        //  avec Clang, nécessite de repousser un peu les limites: -fconstexpr-steps=200000000   (default = 1048576)
        //  gcc??? -fconstexpr-loop-limit=???
    static constexpr auto a = Table<40001>( irBandModel::primitivePlanck, 0., .0005 );          //  x = 0 à 20.0
    static_assert(a.values[0] == irBandModel::primitivePlanck(0.0), "Error: b[0] != 0.0");      //  compile time verification
};

#endif  //use_compile_time_table




////////


namespace py = pybind11;

//template<>
class from_py_array : public std::vector<double>
{
public:
    from_py_array(py::array_t<double, py::array::c_style | py::array::forcecast> array)
    :std::vector<double>(array.size())
    {   // copy py::array -> std::vector
        std::memcpy(data(),array.data(),array.size()*sizeof(double));
    }
};

py::array to_py_array(std::vector<double>& v, std::vector<ssize_t>& shape)
{   ssize_t              ndim    = shape.size();
    //std::cout << "ndim = " << ndim << "\n";
    //std::cout << "shape = (" << shape[0] << ", " << shape[1] << ")\n";
    
    std::vector<ssize_t> strides(ndim);
    strides[ndim-1] = sizeof(double);
    //std::cout << "strides = (" << strides[0] << ", " << strides[1] << ")\n";
    
    if (ndim>1)
        for (int i=ndim-2; i>=0; i--)
        {   //std::cout << "i = " << i << " strides[i+1] " << strides[i+1] << ", " << shape[i+1] << ")\n";
            strides[i] = strides[i+1]*shape[i+1];
        }
    
    //std::cout << "strides = (" << strides[0] << ", " << strides[1] << ")\n";
    
  // return generic NumPy array
      return py::array(py::buffer_info(
        v.data(),                                /* data as contiguous array  */
        sizeof(double),                          /* size of one scalar        */
        py::format_descriptor<double>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
      ));
};
py::array to_py_array_1D( std::vector<double>& v )
{   std::vector<ssize_t> shape   = { (ssize_t)v.size() };
    return to_py_array( v, shape );
    /*
    ssize_t              ndim    = 1;
    std::vector<ssize_t> shape   = { (ssize_t)v.size() };
    std::vector<ssize_t> strides = { sizeof(double) };
    
    return py::array(py::buffer_info(
      v.data(),
      sizeof(double),
      py::format_descriptor<double>::format(),
      ndim,
      shape,
      strides
    ));
    */
};

py::array to_py_array_2D( std::vector<double>& v, ssize_t d1, ssize_t d2 )
{   //if (v.size()!=d1*d2) throw std::runtime_error("bad size for dbdx");
    std::vector<ssize_t> shape   = { d1, d2 };
    return to_py_array( v, shape );
}


/*
// wrap C++ function with NumPy array IO
py::array py_length(py::array_t<double, py::array::c_style | py::array::forcecast> array)
{
  // check input dimensions
  if ( array.ndim()     != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");
  if ( array.shape()[1] != 2 )
    throw std::runtime_error("Input should have size [N,2]");

  from_py_array pos(array);
  // allocate std::vector (to pass to the C++ function)
  //std::vector<double> pos(array.size());

  // copy py::array -> std::vector
  //std::memcpy(pos.data(),array.data(),array.size()*sizeof(double));

  // call pure C++ function
    std::vector<double> result = length(pos);
    
    std::vector<ssize_t> shape   = { array.shape()[0] , 3 };
    return to_py_array( result, shape );
}
*/


class PyRadiatif : public radiativeModel
{
public:
        // inherit the constructors
    using radiativeModel::radiativeModel;
    
    PyRadiatif( size_t na, size_t std_prof_no, double p_surface, double pCO2 ):
        radiativeModel( na, std_prof_no, p_surface, pCO2, true, full_scaling ) {};
    PyRadiatif( size_t na, size_t std_prof_no, double p_surface, double pCO2, bool useRh /*= true*/, scalingType scal /*= full_scaling*/ ):
        radiativeModel( na, std_prof_no, p_surface, pCO2, useRh, scal ) {};
    
    py::tuple py_bilan_x( /*double* b, double* dbdx,*/ const py::array_t<double, py::array::c_style | py::array::forcecast> py_x )
    {   from_py_array x(py_x);
        if (x.size()!=size())
            throw std::runtime_error("bad size for x");
        std::vector<double> b(size());
        std::vector<double> dbdx(size()*size());
        bilan_x( b.data(), dbdx.data(), x.data() );
        //return py::make_tuple( 1, 2 );
        return py::make_tuple( to_py_array_1D(b) , to_py_array_2D( dbdx, size(), size()) );
        //return py::make_tuple(std::move(to_py_array_1D(b)),std::move(to_py_array_2D( dbdx, size(), size()) ));
        //return to_py_array_1D( dbdx );
        //return to_py_array_2D( dbdx, size(), size() );
    };
    
    py::array py_bilanR( const py::array_t<double, py::array::c_style | py::array::forcecast> py_x )
    {   from_py_array x(py_x);
        if (x.size()!=size()) throw std::runtime_error("bad size for x");
        std::vector<double> b(size());
        bilan_x( b.data(), NULL, x.data() );
        return to_py_array_1D( b );
    };
};



// wrap as Python module
PYBIND11_MODULE(radiatif2,m)
{
    m.doc() = "radiatif: pybind11 radiatif C++ code";
    //m.def("rad", &py_radiatif, "computes radiative budget R(x) and derivatives dRdx");
    
    py::class_<PyRadiatif>(m, "Rad")
        .def(py::init<size_t, size_t, double, double, bool, scalingType>())
        .def(py::init<size_t, size_t, double, double>())
        .def("bilanR", &PyRadiatif::py_bilanR)
        .def("ddx_bilanR", &PyRadiatif::py_bilan_x);
}


///   code python
///
/// from radiatif import Rad
/// radia = Rad(4,1,1013.25,280.0)
/// b = radia.bilanR([1.,1.,1.,1.,1.])
/// b, db = radia.ddx_bilanR([1.,1.,1.,1.,1.])
