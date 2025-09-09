//
//  GIHON.hpp
//  RIVERS_XCODE
//
//  Created by Owner on 8/27/25.
//

#ifndef GIHON_hpp
#define GIHON_hpp

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <thread>
#include <iostream>
#include <random>
#include <memory>

using namespace std;
const double G = 6.6743e-11; //m3 kg-1 s-2 (Gravitational Constant)
const double epsilson=1e-7;

//Vector Operations
double norm(vector<double> r);
vector<double> operator+(vector<double>a, vector<double> b);
vector<double> operator-(vector<double>a, vector<double> b);
vector<double> operator*(double c, vector<double> v);
vector<double> operator +=(vector<double> a, vector<double> b);


class Body{
public:
    vector<double> r;
    vector<double> v;
    double mass;
    
    Body(vector<double> r0, vector<double> v0, double m); 
};

class Equation{
public:
    
    vector<double> dv_1(vector<Body> b,vector<double>K);
    vector<double> dv_2(vector<Body> b,vector<double>K);
    vector<double> dv_3(vector<Body> b,vector<double>K);
    
    vector<double> dr_1(vector<double> J);
    vector<double> dr_2(vector<double> J);
    vector<double> dr_3(vector<double> J);


};

//The Runge-Kutta Method applied. 
void RK45_Method(vector<Body> b);


#endif /* GIHON_hpp */
