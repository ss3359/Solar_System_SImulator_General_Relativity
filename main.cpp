//
//  main.cpp
//  RIVERS_XCODE
//
//  Created by Owner on 5/2/25.
//

#include <cmath>
#include <vector>
#include <thread>
#include <iostream>
#include <random>
#include <memory> 
#include "TIGRIS.hpp"
#include "EUPHRATES.hpp"
#include "PISHON.hpp"
#include "GIHON.hpp"

using namespace std;


/*
 | body |  m |           x |           y |  z |            vx |          vy| vz | | ---- | -: | ----------: | ----------: | -: | ------------: | ------------: | -: |
 | 1    |  1 | -0.97000436 |  0.24308753 |  0 |  0.4662036850 |  0.4323657300 |  0 |
 | 2    |  1 |  0.97000436 | -0.24308753 |  0 |  0.4662036850 |  0.4323657300 |  0 |
 | 3    |  1 |  0.00000000 |  0.00000000 |  0 | -0.9324073700 | -0.8647314600 |  0 |

 */

int main(){
    Body b1({-9.7,2.4,0.0},{ 4.6,4.3,0.0},1.0);
    Body b2({9.7,-2.4,0.0},{4.6,4.3,0.0},1.0);
    Body b3({0.0,0.0 ,0}, {-9.3,-8.6,0.0},1.0);
    
    vector<Body> Bodies={b1,b2,b3};
    
    
    RK45_Method(Bodies);
    
    return 0;
}

//    double s0 = 69.21, i0=10;
//    SIR s(s0, i0);
//    s.UpdateSIR();
