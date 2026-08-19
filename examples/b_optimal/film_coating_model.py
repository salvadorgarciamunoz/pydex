#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
film_coating_model.py
=====================
Tablet film-coating thermodynamic model, used by scenario_1_film_coating.py.
Named for consistency with cstr_model.py and suzuki_model.py.

Originally `pyfilmcoaterss` (created 26 Jul 2021, @author: c184156), an
implementation of the Ebey (1987) / Am Ende & Berchielli (2005)
thermodynamic coating model.

Modified from the original: `pct_used_drying_capacity` previously used only
the coating solution's water in its numerator, ignoring the moisture already
carried by the inlet drying air, so it under-reported saturation whenever the
inlet air was not dry. The numerator is now the total outlet vapour
(`massrate_vapor_outlet_kg_min`), in both `pred_exhaust()` and the duplicate
logic in `calc_rh_ex()`. `T_exh_C` and `RH_exh` were never affected.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize

def latent_heat_vap_water_kJ_kg(T_K):
    return(1.91846E3*((T_K/(T_K-33.91))**2))

#Taken from "A new formula for latent heat of vaporization of water as a function of temperature",Quart. J. R. Met. SOC. (1984). 110, pp. 1186-1190 

def psat_Pa (T_K):
    return(np.exp(-6096.9385*(1/T_K) + 21.2409642 - 2.711193E-2*T_K 
                  +1.673952E-5*(T_K**2) + 2.433502*np.log(T_K)))
#Vapor pressure of water Eq.12-4a from Perry's Chem. Eng. Handbook 9th Ed

def dens_dry_air_kg_m3(T_K):
    P_Pa=101325
    #Rs=R/Mw
    R =8.31446261815324 # J/K mol
    Mw=28.84 #gr/mol
    Rs = 1000*R/Mw # J/kg K
    dda=P_Pa/(Rs*T_K)   #kg/m3
    return dda
# Ideal gas Law

def pred_exhaust(*,T_dp_in_C= 11.2,T_in_C= 60,Fair_in_CFM= 1700,
                 SolnFR_in_gpm= 450 ,Solids_in_soln_pcnt= 12,T_room_C= 20,HLF= 5.63078536 *.75,P_Pa= 101325):
    '''
    

    Parameters
    ----------

    T_dp_in_C : TYPE, Real number
        Dew point temperature of inlet in deg C. The default is 11.2.
    T_in_C : TYPE, Real number
        Inlet air temperature in deg C . The default is 60.
    Fair_in_CFM : TYPE, Real number
        Drying air flowrate in ft3/min. The default is 1700.
    SolnFR_in_gpm : TYPE, Real number
        Total solution flow rate in gr/min. The default is 450.
    Solids_in_soln_pcnt : TYPE, Real number
        Percent of solids in solution. The default is 12.
    T_room_C : TYPE, Real number
        Room temperature in deg C - assumed solution temperature and ambient. The default is 20.
    HLF : TYPE, Real number
        Heat Loss Factor in kJ/min deg K. The default is 5.63078536 *.75.
    P_Pa : TYPE, Real number
        Pressure in coater in Pa. The default is atmospheric 101325.

    Returns
    -------
    T_exh_C : TYPE Real number
        Temperature of exhaust air in deg C.
    RH_exh : TYPE Real number
        Relative Humidity of exhaust air (%).
    pct_used_drying_capacity : TYPE Real number
        Percent used of drying capacity (for exhaust air to be 100% saturated).
    Y_abs_humidity_out: TYPE Real number
        Absolute humidity at the exhaust (kg water / kg dry air)

    '''

    # Some useful "constants" OK they are not constant, they vary with temp
    # but neither is the system completely at steady state.
    
    Cp_dry_air       = 1.0063 # kJ/kg K
    Cp_water_vapor   = 1.864  # kJ/kg K
    Cp_water_liq     = 4.184  # kJ/kg K
    Heat_vap_water   = latent_heat_vap_water_kJ_kg(45+273.15) #Initial guess for Heat of Vap at 45C


    vap_press_in = psat_Pa(T_dp_in_C + 273.15)

    
    Y_abs_humidity_in  = 0.622*vap_press_in/(P_Pa-vap_press_in) #kg water / kg dry air #Table 12-1
    Yw_specific_humidity_in = Y_abs_humidity_in/(1+Y_abs_humidity_in) # kg water / kg humid air #page 12-3 Perry's 9th edition
    Yv_volumetric_humidity_in = vap_press_in*0.002167/(T_in_C+273.15)  # kg water / m3 humid air #Table 12.1
    dens_humid_air_in = Yv_volumetric_humidity_in/Yw_specific_humidity_in # kg humid air / m3 humid air
    # Perry's Chem. Eng Handbook Chapter table 12-1
    
    
    massrate_water_solution_kg_min = (1/1000)*SolnFR_in_gpm*(1-(Solids_in_soln_pcnt/100)) #kg water / min
    
    massrate_vapor_inlet_kg_min = Fair_in_CFM * 0.028316846999 * Yv_volumetric_humidity_in #kg water vapor / min
    #                        (ft3 humid air / min)   (m3 humid air /ft3 humid air) * (kg water / m3 humid air)
    
    massrate_vapor_outlet_kg_min = massrate_vapor_inlet_kg_min + massrate_water_solution_kg_min  #kg water vapor / min
    #assuming complete evaporation to air
    
    massrate_humid_air_inlet_kg_min = Fair_in_CFM * 0.028316846999 *  dens_humid_air_in # kg/min
    #              (ft3 humid air / min)   (m3 humid air /ft3 humid air) * (kg humid air / m3 humid air)
    massrate_dry_air_inlet_kg_min = massrate_humid_air_inlet_kg_min-massrate_vapor_inlet_kg_min
    
    Y_abs_humidity_out =massrate_vapor_outlet_kg_min/massrate_dry_air_inlet_kg_min #kg water / kg dry air
    
    vap_press_out = P_Pa*Y_abs_humidity_out/(0.622+Y_abs_humidity_out)
    # Perry's Chem. Eng Handbook Chapter Table 12.1
      
    #Calculate initial Texh
    T_exh_C_0 =(
        (  massrate_vapor_inlet_kg_min    * Cp_water_vapor * (T_in_C+273.15)
         + massrate_dry_air_inlet_kg_min  * Cp_dry_air     * (T_in_C+273.15)
         + massrate_water_solution_kg_min * Cp_water_liq   * (T_room_C+273.15)
         - massrate_water_solution_kg_min * Heat_vap_water
         + HLF*(T_room_C+273.15) 
         )
        /
        (  massrate_vapor_inlet_kg_min*Cp_water_vapor
         + massrate_dry_air_inlet_kg_min* Cp_dry_air
         + massrate_water_solution_kg_min*Cp_water_liq
         + HLF
         )
        )-273.15
    #Correct heat of vaporization
    #
    Heat_vap_water=latent_heat_vap_water_kJ_kg(T_exh_C_0+273.15)
    not_converged=True
    
    #Convergence loop to calculate T_exh with the correct heat of vaporization
    while not_converged:
        T_exh_C_1 =(
        (  massrate_vapor_inlet_kg_min    * Cp_water_vapor * (T_in_C+273.15)
         + massrate_dry_air_inlet_kg_min  * Cp_dry_air     * (T_in_C+273.15)
         + massrate_water_solution_kg_min * Cp_water_liq   * (T_room_C+273.15)
         - massrate_water_solution_kg_min * Heat_vap_water
         + HLF*(T_room_C+273.15) 
         )
        /
        (  massrate_vapor_inlet_kg_min*Cp_water_vapor
         + massrate_dry_air_inlet_kg_min* Cp_dry_air
         + massrate_water_solution_kg_min*Cp_water_liq
         + HLF
         )
        )-273.15
        if abs(T_exh_C_1-T_exh_C_0) < .001:
            not_converged=False
        else:
            T_exh_C_0=T_exh_C_1
            Heat_vap_water=latent_heat_vap_water_kJ_kg(T_exh_C_0+273.15)
        
    T_exh_C=T_exh_C_1
    sat_press_out = psat_Pa(T_exh_C + 273.15)
    RH_exh = 100*vap_press_out/sat_press_out
    
    #Calculate % drying capacity, basically making vap_press_out = sat_press_out
    Ysat_out=0.622*sat_press_out/(P_Pa- sat_press_out) #kg water / kg dry air for air to be 100% RH # Perry's Chem. Eng Handbook Chapter 12.
    max_water_to_dry_kg_min =Ysat_out * massrate_dry_air_inlet_kg_min
    # FIX: the numerator must be the TOTAL water vapor carried by the outlet
    # stream (inlet baseline humidity + solution water), not the solution's
    # water alone -- otherwise this metric ignores the drying capacity
    # already consumed by the inlet air's own humidity, and can read <=100%
    # even when the exhaust air is supersaturated (RH_exh > 100%).
    # This form is exactly 100% iff Y_abs_humidity_out == Ysat_out, i.e.
    # iff vap_press_out == sat_press_out, i.e. iff RH_exh == 100%, so it is
    # now a consistent cross-check against RH_exh rather than a separate,
    # looser criterion.
    pct_used_drying_capacity = 100*massrate_vapor_outlet_kg_min/max_water_to_dry_kg_min
#    if pct_used_drying_capacity >100:
#        T_exh_C = np.nan
#        RH_exh  = np.nan
    Y_abs_humidity_out  = 0.622*vap_press_out/(P_Pa-vap_press_out) #kg water / kg dry air
    return T_exh_C,RH_exh,pct_used_drying_capacity,Y_abs_humidity_out

def _t_exhaust(x_vector,hlf):                 
    t_dew_in= x_vector[:,0]
    t_in    = x_vector[:,1]
    F_in    = x_vector[:,2]
    S_in    = x_vector[:,3]
    sol_pct = x_vector[:,4]
    t_room  = x_vector[:,5]
    press   = x_vector[:,6]
    
    texh=[]
    for (td,t,f,s,sol,tr,p) in zip(t_dew_in,t_in,F_in,S_in,sol_pct,t_room,press):
        texh_,rhexh,pct_drying_cap,abs_hum=pred_exhaust(T_dp_in_C=td, T_in_C=t, Fair_in_CFM=f, SolnFR_in_gpm=s,Solids_in_soln_pcnt=sol,
                                                T_room_C=tr,P_Pa=p,HLF=hlf)
        texh.append(texh_)
    texh=np.array(texh)
    return texh   

def estimate_hlf(inlet_conditions,t_exhaust_actual):
    '''
    
    Parameters
    ----------
    inlet_conditions : Numpy Array of n x 7 size 
        Matrix containing the process parameters for each sample (n)
        the parameters (columns) are in the same order and units as in 
        pred_exhaust:
            1. Dew Temp at inlet in C
            2. Inlet Temperature in C 
            3. Drying Air Flowrate in ft3/min
            4. Total Solution Flow Rate in gr/min
            5. Solids Percent in Solution in %
            6. Ambient (room) Temperature in C
            7. System Pressure in Pa
        
    t_exhaust_actual : TYPE numpy Array of size n
        Vector of size n with the actual exhaust temperature (in C).

    Returns
    -------
    HLF estimated value in KJ/K min
    Variance of the HLF estimation

    '''
    popt,covp=curve_fit(_t_exhaust,inlet_conditions,t_exhaust_actual)  
    return popt,covp

def _fun_2_min(x,z):
    t_in_c,fair_in_cfm=x
    t_exh_sp,rh_exh_sp,t_dp_in_c,solnfr_in_gpm,solids_in_soln_pcnt,t_room_c,hlf,p_pa=z
    T_exh_C,RH_exh,pct_used_drying_capacity,Y_abs_humidity_out=pred_exhaust(T_dp_in_C= t_dp_in_c,T_in_C= t_in_c,Fair_in_CFM= fair_in_cfm,
                 SolnFR_in_gpm= solnfr_in_gpm ,Solids_in_soln_pcnt= solids_in_soln_pcnt,T_room_C= t_room_c,HLF=hlf,P_Pa= p_pa)
    f=((t_exh_sp-T_exh_C)/30)**2 + ((rh_exh_sp-RH_exh )/15)**2
    return f
    
def _fun_2_min_b(x,z):
    t_in_c,solnfr_in_gpm=x
    t_exh_sp,rh_exh_sp,t_dp_in_c,fair_in_cfm,solids_in_soln_pcnt,t_room_c,hlf,p_pa=z
    T_exh_C,RH_exh,pct_used_drying_capacity,Y_abs_humidity_out=pred_exhaust(T_dp_in_C= t_dp_in_c,T_in_C= t_in_c,Fair_in_CFM= fair_in_cfm,
                 SolnFR_in_gpm= solnfr_in_gpm ,Solids_in_soln_pcnt= solids_in_soln_pcnt,T_room_C= t_room_c,HLF=hlf,P_Pa= p_pa)
    f=((t_exh_sp-T_exh_C)/30)**2 + ((rh_exh_sp-RH_exh )/15)**2
    return f

def _fun_2_min_c(x,z):
    t_in_c,solnfr_in_gpm=x
    t_exh_sp,abs_hum_sp,t_dp_in_c,fair_in_cfm,solids_in_soln_pcnt,t_room_c,hlf,p_pa=z
    T_exh_C,RH_exh,pct_used_drying_capacity,Y_abs_humidity_out=pred_exhaust(T_dp_in_C= t_dp_in_c,T_in_C= t_in_c,Fair_in_CFM= fair_in_cfm,
                 SolnFR_in_gpm= solnfr_in_gpm ,Solids_in_soln_pcnt= solids_in_soln_pcnt,T_room_C= t_room_c,HLF=hlf,P_Pa= p_pa)
    f=((t_exh_sp-T_exh_C)/30)**2 + ((abs_hum_sp-Y_abs_humidity_out)/.014)**2
    return f

def pred_inlet_t_sprayrate(*,T_exh=39.21,RH_exh=36.19, T_dp_in_C= 11.2, Fair_in_CFM=1700 ,
                        Solids_in_soln_pcnt= 12,T_room_C= 20,HLF= 5.63078536,P_Pa= 101325,
                        SolnFR_in_gpm_init_guess=250,
                        T_in_C_init_guess=55):
    
    tin_sprayrate=minimize(_fun_2_min_b,[T_in_C_init_guess,SolnFR_in_gpm_init_guess],args=([T_exh,RH_exh,T_dp_in_C,Fair_in_CFM,Solids_in_soln_pcnt,T_room_C,HLF,P_Pa]),tol=1E-10)
   
    T_in_C,SolnFR_in_gpm = tin_sprayrate.x
    t_exh_c_obt,rh_exh_obt,pct_used_drying_capacity,Y_abs_humidity_out=pred_exhaust(T_dp_in_C= T_dp_in_C,T_in_C= T_in_C,Fair_in_CFM= Fair_in_CFM,
                 SolnFR_in_gpm= SolnFR_in_gpm ,Solids_in_soln_pcnt= Solids_in_soln_pcnt,T_room_C= T_room_C,HLF=HLF,P_Pa= P_Pa)
    return T_in_C,SolnFR_in_gpm,pct_used_drying_capacity,Y_abs_humidity_out

def pred_inlet_t_sprayrate_hum(*,T_exh=39.21,Abs_Hum=0.016, T_dp_in_C= 11.2, Fair_in_CFM=1700 ,
                        Solids_in_soln_pcnt= 12,T_room_C= 20,HLF= 5.63078536,P_Pa= 101325):
    
    tin_sprayrate=minimize(_fun_2_min_c,[55,250],args=([T_exh,Abs_Hum,T_dp_in_C,Fair_in_CFM,Solids_in_soln_pcnt,T_room_C,HLF,P_Pa]),tol=1E-10)
   
    T_in_C,SolnFR_in_gpm = tin_sprayrate.x
    t_exh_c_obt,rh_exh_obt,pct_used_drying_capacity,Y_abs_humidity_out=pred_exhaust(T_dp_in_C= T_dp_in_C,T_in_C= T_in_C,Fair_in_CFM= Fair_in_CFM,
                 SolnFR_in_gpm= SolnFR_in_gpm ,Solids_in_soln_pcnt= Solids_in_soln_pcnt,T_room_C= T_room_C,HLF=HLF,P_Pa= P_Pa)
    return T_in_C,SolnFR_in_gpm,pct_used_drying_capacity,rh_exh_obt
    
def pred_inlet_t_airflw(*,T_exh=39.21,RH_exh=36.19, T_dp_in_C= 11.2, SolnFR_in_gpm= 450 ,
                        Solids_in_soln_pcnt= 12,T_room_C= 20,HLF= 5.63078536,P_Pa= 101325):
    tin_airflow=minimize(_fun_2_min,[55,500],args=([T_exh,RH_exh,T_dp_in_C,SolnFR_in_gpm,Solids_in_soln_pcnt,T_room_C,HLF,P_Pa]),tol=1E-10)
    T_in_C,Fair_in_CFM = tin_airflow.x
    t_exh_c_obt,rh_exh_obt,pct_used_drying_capacity,Y_abs_humidity_out=pred_exhaust(T_dp_in_C= T_dp_in_C,T_in_C= T_in_C,Fair_in_CFM= Fair_in_CFM,
                 SolnFR_in_gpm= SolnFR_in_gpm ,Solids_in_soln_pcnt= Solids_in_soln_pcnt,T_room_C= T_room_C,HLF=HLF,P_Pa= P_Pa)
    return T_in_C,Fair_in_CFM,pct_used_drying_capacity,Y_abs_humidity_out
    
def calc_rh_ex(*,T_exh_C=39.21,T_dp_in_C= 11.2,T_in_C= 60,Fair_in_CFM= 1700,
                 SolnFR_in_gpm= 450 ,Solids_in_soln_pcnt= 12,P_Pa= 101325):
    '''

    Parameters
    ----------
    T_exh     : TYPE, Real number
        Measured Exhaust temperature in deg C
    T_dp_in_C : TYPE, Real number
        Dew point temperature of inlet in deg C. The default is 11.2.
    T_in_C : TYPE, Real number
        Inlet air temperature in deg C . The default is 60.
    Fair_in_CFM : TYPE, Real number
        Drying air flowrate in ft3/min. The default is 1700.
    SolnFR_in_gpm : TYPE, Real number
        Total solution flow rate in gr/min. The default is 450.
    Solids_in_soln_pcnt : TYPE, Real number
        Percent of solids in solution. The default is 12.
    P_Pa : TYPE, Real number
        Pressure in coater in Pa. The default is atmospheric 101325.

    Returns
    -------

    RH_exh : TYPE Real number
        Relative Humidity of exhaust air (%).
    pct_used_drying_capacity : TYPE Real number
        Percent used of drying capacity (for exhaust air to be 100% saturated).
    Y_abs_humidity_out: TYPE Real number
        Absolute humidity at the exhaust (kg water / kg dry air)

    '''

    vap_press_in = psat_Pa(T_dp_in_C + 273.15)
    Y_abs_humidity_in  = 0.622*vap_press_in/(P_Pa-vap_press_in) #kg water / kg dry air #Table 12-1
    Yw_specific_humidity_in = Y_abs_humidity_in/(1+Y_abs_humidity_in) # kg water / kg humid air #page 12-3 Perry's 9th edition
    Yv_volumetric_humidity_in = vap_press_in*0.002167/(T_in_C+273.15)  # kg water / m3 humid air #Table 12.1
    dens_humid_air_in = Yv_volumetric_humidity_in/Yw_specific_humidity_in # kg humid air / m3 humid air
    # Perry's Chem. Eng Handbook Chapter table 12-1
    
    
    massrate_water_solution_kg_min = (1/1000)*SolnFR_in_gpm*(1-(Solids_in_soln_pcnt/100)) #kg water / min
    massrate_vapor_inlet_kg_min = Fair_in_CFM * 0.028316846999 * Yv_volumetric_humidity_in #kg water vapor / min
    #                        (ft3 humid air / min)   (m3 humid air /ft3 humid air) * (kg water / m3 humid air)
    massrate_vapor_outlet_kg_min = massrate_vapor_inlet_kg_min + massrate_water_solution_kg_min  #kg water vapor / min
    #assuming complete evaporation to air
    massrate_humid_air_inlet_kg_min = Fair_in_CFM * 0.028316846999 *  dens_humid_air_in # kg/min
    #              (ft3 humid air / min)   (m3 humid air /ft3 humid air) * (kg humid air / m3 humid air)
    massrate_dry_air_inlet_kg_min = massrate_humid_air_inlet_kg_min-massrate_vapor_inlet_kg_min
    
    Y_abs_humidity_out =massrate_vapor_outlet_kg_min/massrate_dry_air_inlet_kg_min #kg water / kg dry air
    
    vap_press_out = P_Pa*Y_abs_humidity_out/(0.622+Y_abs_humidity_out)
    # Perry's Chem. Eng Handbook Chapter Table 12.1
     
    sat_press_out = psat_Pa(T_exh_C + 273.15)
    RH_exh = 100*vap_press_out/sat_press_out    
    #Calculate % drying capacity, basically making vap_press_out = sat_press_out
    Ysat_out=0.622*sat_press_out/(P_Pa- sat_press_out) #kg water / kg dry air for air to be 100% RH # Perry's Chem. Eng Handbook Chapter 12.
    max_water_to_dry_kg_min =Ysat_out * massrate_dry_air_inlet_kg_min
    # FIX: see pred_exhaust() -- numerator must be total outlet vapor, not
    # just the solution's contribution, to be consistent with RH_exh.
    pct_used_drying_capacity = 100*massrate_vapor_outlet_kg_min/max_water_to_dry_kg_min

    Y_abs_humidity_out  = 0.622*vap_press_out/(P_Pa-vap_press_out) #kg water / kg dry air
    return RH_exh,pct_used_drying_capacity,Y_abs_humidity_out