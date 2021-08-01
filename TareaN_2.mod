#FUNCIONES 

function gsl_cdf_ugaussian_P ;
function gsl_ran_ugaussian_pdf ;
function gsl_cdf_ugaussian_Pinv ; 

#CONJUNTOS

set C := {1..32};

#PARAMETROS

param alfa_L;
param S;
param h;
param mu;
param sigma;
param L;

param alfa_L_a{C};
param S_a{C};
param h_a{C};
param mu_a{C};
param sigma_a{C};
param L_a{C}; 

#VARIABLES 

var Q;
var r>=0; 

#FUNCION OBJETIVO

var x1 = (r-mu*L)/(sqrt(sigma*L));
var x2 = (r-mu*L+Q)/(sqrt(sigma*L));

var dis1 = gsl_cdf_ugaussian_P(x1); 
var dis2 = gsl_cdf_ugaussian_P(x2); 

var den1 = gsl_ran_ugaussian_pdf(x1);
var den2 = gsl_ran_ugaussian_pdf(x2);

var Hx1 = 0.5*((x1*x1+1)*(1-dis1)-x1*den1);
var Hx2 = 0.5*((x2*x2+1)*(1-dis2)-x2*den2);

var B = sigma*L/Q*(Hx1-Hx2);

var FO = S*(mu/Q) + h*(Q/2+r-mu*L+B);

minimize costos: FO;

#RESTRICCIONES

s.t. r1:
	r >= mu*L+gsl_cdf_ugaussian_Pinv(alfa_L)*sqrt(sigma*L);
	
s.t. r2:
	Q >= sqrt((2*mu*S/h));
	
