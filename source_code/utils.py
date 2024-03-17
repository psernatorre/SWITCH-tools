""" Utils functions for the project
"""
# System packages
from pathlib import Path

# Third-party packages
import pandas as pd
import yaml
from joblib import Parallel, delayed
import numpy as np
import geopandas as gpd
import math
from datetime import datetime
from pytz import timezone
from shapely.ops import unary_union
from matplotlib import cm, colors, colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Constant definitions
with open("./config_1.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

tech = {key: val["map"] for key, val in config["technologies"].items()}
tech_colors = {key: val["color"] for key, val in config["technologies"].items()}
tech_order = config["tech_order"]

tech_dict = {
    new_tech: tech for tech, new_techs in tech.items() for new_tech in new_techs
}

data_path = "/home/psernatorre/power_systems_expansion/switch_scenarios/paper/"


# ----------------------------------------------------       ------------------------------------------------------------------------------------------------
# ---------------------------------------------------- TOOLS ------------------------------------------------------------------------------------------------
# ----------------------------------------------------       ------------------------------------------------------------------------------------------------

def get_single_df(scenario: str, 
                  fname: str, 
                  load_zone: str = None, 
                  fpath="outputs", *args, **kwargs):
    
    fname = data_path + scenario + "/" + fpath + "/"+  fname
    return (
        pd.read_csv(fname, *args, **kwargs)
        .pipe(tech_map)
        .pipe(timepoint_map)
        .assign(scenario=scenario)
    )

def get_data(scenario: str, 
             fname: str, 
             fpath="outputs", *args, **kwargs):
    """Small wrapper of get same file for multiple scenarios

    It uses joblib to read a df in each thread which by default (-1) uses
    all threads available in the computer.
    """
    if isinstance(scenario, list):
        fname_dfs = Parallel(n_jobs=-1)(
            delayed(get_single_df)(sce, fname, fpath=fpath,*args, **kwargs)
            for sce in scenario
        )

        return pd.concat(fname_dfs)
    else:
        return get_single_df(scenario, fname, fpath=fpath, *args, **kwargs)

def tech_map(df):
    """ Apply custom technology map"""
    # Create new column if data contains technology
    df = df.copy()
    if "gen_tech" in df.columns:
        df["tech_map"] = df["gen_tech"].map(tech_dict).astype("category")
        assert df["tech_map"].isnull().values.any() == False
    return df

def timepoint_map(df):
    """ Convert timepoint to datetime object"""
    df = df.copy()
    columns = ["timestamp", "timepoint"]
    if any(val in df.columns for val in columns):
        try:
            df["datetime"] = pd.to_datetime(df["timepoint"], format="%Y%m%d%H")
        except:
            print("exception")
            if "timestamp" in columns: 
                print("timestamp in column")
#                 df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")
        return df
    return df 

def default_value(Dataframe, column, default_value):
    if column not in Dataframe.columns:
        Dataframe[column] = default_value
    else:
        Dataframe[column] = Dataframe[column].replace('.',0).astype(float)

    return Dataframe

def closest(lst: list, K: float):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def read_scenarios(file_name: str):
    input_scenarios = pd.read_csv(file_name, index_col=False)
    scenario = list(input_scenarios['scenario'])
    short_names = dict(input_scenarios[['scenario', 'short_name']].values)
    inv_short_names = dict(zip(short_names.values(),short_names.keys()))
    order = dict(input_scenarios[['short_name', 'order']].values)

    return scenario, short_names, inv_short_names, order

def return_interval(number: float, df_intervals: pd.DataFrame):
     for idx, row in df_intervals.iterrows():
          if row["min_interval"]<=number<row['max_interval']:
               return row

def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"

def timestamp_info(scenario_file_name: str, analysis_scenario):

    # Output 1:
    #    scenario	ts_period	hours_in_period	period_start	period_end	err_plain	err_add_one	add_one_to_period_end_rule	period_length_years
    #    high_100	2050	    87,660.00	    2046	        2055	    -8,766.00	-0.00	                             1	                10
    #
    # Output 2:
    #   scenario	ts_period	timestamp	tp_weight	hours_in_period	period_start	period_end	err_plain	err_add_one	add_one_to_period_end_rule	period_length_years	tp_weight_in_year
    #   high_100	2050	    2050010200	40.14	    87,660.00	    2046	        2055	    -8,766.00	-0.00	    1	                        10	                4.01
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenario_file_name)

    if len(analysis_scenario)>0:
        scenario = [inv_short_names[i] for i in analysis_scenario]

    hours_per_year = 8766

    timepoints = get_data(scenario, "timepoints.csv", fpath='inputs')
    timepoints.columns= timepoints .columns.str.lower()
    timepoints.rename(columns={'timepoint_id': 'timepoint'}, inplace=True)
    timepoints = timepoints .replace({"scenario": short_names})

    timeseries = get_data(scenario, "timeseries.csv", fpath='inputs')
    timeseries.columns= timeseries .columns.str.lower()
    timeseries = timeseries .replace({"scenario": short_names})

    periods = get_data(scenario, "periods.csv", fpath='inputs')
    periods.columns= periods .columns.str.lower()
    periods.rename(columns={'investment_period': 'ts_period'}, inplace=True)
    periods = periods .replace({"scenario": short_names})

    time_info=pd.merge(left=timeseries,right=timepoints, on=['timeseries', 'scenario'])
    time_info=pd.merge(left=time_info,right=periods , on=['scenario', 'ts_period'])

    time_info['tp_weight']=time_info['ts_duration_of_tp']*time_info['ts_scale_to_period']

    time_info = time_info[['scenario', 'ts_period', 'timestamp', 'tp_weight']]

    period_info=time_info.pivot_table(index=['scenario','ts_period'], values='tp_weight',aggfunc=np.sum )
    period_info.reset_index(inplace=True)
    period_info.rename(columns={'tp_weight': 'hours_in_period'}, inplace=True)
    period_info = pd.merge(left=period_info, right = periods, on=['scenario', 'ts_period'])
    period_info['err_plain'] = (period_info['period_end'] - period_info['period_start'])*hours_per_year - period_info['hours_in_period']
    period_info['err_add_one'] =  (period_info['period_end'] + 1 - period_info['period_start'])*hours_per_year - period_info['hours_in_period']
    period_info.loc[:, 'add_one_to_period_end_rule']= period_info.apply(lambda x: 1 if np.absolute(x['err_add_one'])<np.absolute(x['err_plain']) else 0, axis=1)
    period_info['period_length_years'] = period_info['period_end'] - period_info['period_start'] + period_info['add_one_to_period_end_rule']

    time_df = pd.merge(left = time_info, right = period_info, on=['scenario', 'ts_period'] ).reset_index(drop=True)
    time_df['tp_weight_in_year'] = time_df['tp_weight']/time_df['period_length_years']

    return period_info, time_df

def value_to_color(color_palette, color_min, color_max, val):
    palette = colormaps[color_palette]
    val_position = float((val - color_min)) / (color_max - color_min) 
    return palette(val_position)

def adjust_tile_size(start_tile_size, tile_size_inc, min, max, interval_length, value):
    n_intervals = int(np.ceil((max - min)/interval_length))
    for j in range(1,n_intervals+1):
        if  min+(j-1)*interval_length<=value<= min + j*interval_length:
            return start_tile_size + j*tile_size_inc

    '''     
    Motivation: 
    wecc_cost.loc[:, 'scatter_size'] = wecc_cost.apply(lambda x: start_size if 0.6<x['relative']<=0.65 else
                                                             start_size+step if 0.65<x['relative']<=0.70 else
                                                             start_size+2*step if 0.70<x['relative']<=0.75 else
                                                             start_size+3*step if 0.75<x['relative']<=0.80 else
                                                             start_size+4*step if 0.80<x['relative']<=0.85 else
                                                             start_size+5*step if 0.85<x['relative']<=0.9 else
                                                             start_size+6*step if 0.9<x['relative']<=0.95 else 
                                                             start_size+7*step if 0.9<x['relative']<=1 else 0, axis=1)
    '''   
# ----------------------------------------------------                 --------------------------------------------------------------------------------------
# ---------------------------------------------------- FINANCIAL TOOLS --------------------------------------------------------------------------------------
# ----------------------------------------------------                 --------------------------------------------------------------------------------------

def crf(ir, t):
    if ir==0:
        return 1/t
    else:
        return (ir/(1-(1+ir)**(-t)))

def uniform_series_to_present_value(dr, t):
    if dr==0:
        return t
    else:
        return ((1-(1+dr)**(-t))/dr)

def future_to_present_value(dr, t):
    return (1+dr)**(-t)

def present_to_future_value(ir, t):
    return (1+ir)**(t)


# ---------------------------------------------------      --------------------------------------------------------------------------------------------------
# --------------------------------------------------- MAPS --------------------------------------------------------------------------------------------------
# ---------------------------------------------------      --------------------------------------------------------------------------------------------------

def wecc_map_geodf(file_location: str):
    # Load the wecc load zones shape file
    wecc_load_areas = gpd.read_file(file_location)
    wecc_load_areas.rename(columns={'LOAD_AREA':'load_zone'}, inplace=True)
    wecc_load_areas['centroid'] = wecc_load_areas['geometry'].apply(lambda x: x.centroid) #The centroids are later used to place pie charts

    #Modify the centroids of canada load zones. Canada load zones occupy much space in the graph, so it is nice-looking to place the piecharts close to the US border
    wecc_load_areas['centroid_partial_can'] = wecc_load_areas['centroid']
    wecc_load_areas.loc[wecc_load_areas['load_zone']=='CAN_ALB', 'centroid_partial_can']= gpd.points_from_xy(x=[-112.50762] ,y=[50.77121])
    wecc_load_areas.loc[wecc_load_areas['load_zone']=='CAN_BC', 'centroid_partial_can']= gpd.points_from_xy(x=[-119.73617] ,y=[50.77148])
    wecc_load_areas.loc[wecc_load_areas['load_zone']=='WA_SEATAC', 'centroid_partial_can']= gpd.points_from_xy(x=[-122] ,y=[47.5])
    wecc_load_areas.loc[wecc_load_areas['load_zone']=='WA_W', 'centroid_partial_can']= gpd.points_from_xy(x=[-123.2] ,y=[47])

    wecc_load_areas.set_index('load_zone', inplace=True)
    return wecc_load_areas

def merge_polygons(polygons):
    return gpd.GeoSeries(unary_union(polygons))

def wecc_states_geodf(file_location: str, zones_states: pd.DataFrame):
    wecc_load_areas = wecc_map_geodf(file_location)
    wecc_load_areas = wecc_load_areas[['geometry']]
    wecc_load_areas.reset_index(inplace=True)
    wecc_load_areas =  pd.merge(left = wecc_load_areas, right = zones_states, on='load_zone')
    wecc_special = wecc_load_areas.pivot_table(index='state', values = 'geometry', aggfunc=merge_polygons)
    wecc_special.reset_index(inplace=True, drop=False)
    wecc_special['centroid'] = wecc_special['geometry'].apply(lambda x: x.centroid)
    wecc_special['centroid_partial_can'] = wecc_special['centroid']
    wecc_special.loc[wecc_special['state']=='CAN', 'centroid_partial_can']= gpd.points_from_xy(x=[-116.121895] ,y=[50.77121])
    wecc_special.rename(columns={'state':'load_zone'}, inplace=True)
    wecc_special.set_index('load_zone', inplace=True)
    
    return wecc_special

# ---------------------------------------------------            --------------------------------------------------------------------------------------------
# --------------------------------------------------- GENERATION --------------------------------------------------------------------------------------------
# ---------------------------------------------------            --------------------------------------------------------------------------------------------

def gen_build_can_operate_in_period(gen_max_age, 
                                    build_year, 
                                    investment_period, 
                                    period_start, 
                                    period_length_years):
        if build_year==investment_period:
            online = period_start
        else:
            online = build_year
        
        retirement = online + gen_max_age

        if online <= period_start + 0.5*period_length_years < retirement :
            return 1
        else:
            return 0
        
def gen_cap_by_tech_by_scenario(scenarios_file_name: str, 
                                analysis_scenarios: list,
                                analysis_period:list, 
                                analysis_tech:list, 
                                analysis_zones:list, 
                                baseline_scenario):
    # Output 1 and 2 : The only difference is that the ouput 2 is in percentage relative to the baseline row.
    # scenario | Wind	Solar	Gas	    Oil	    Coal	Biomass	    Geothermal	Storage	Hydro	Nuclear	Waste	Total
    # low_5	     68.57	131.57	0.00	0.00	0.00	108.92	    0.00	    162.93	100.00	0.00	0.00	132.54

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 
    gen_cap = get_data(scenario, "gen_cap.csv" , usecols=["GENERATION_PROJECT", "PERIOD", "gen_tech", "gen_load_zone", "GenCapacity"])
    gen_cap.replace({"scenario": short_names}, inplace=True)
    gen_cap.insert(5,"GenCapacity_GW",gen_cap["GenCapacity"] / 1e3  )

    if len(analysis_zones)==0:
        analysis_zones = list(gen_cap.gen_load_zone.unique())
    if len(analysis_tech)==0:
        analysis_tech = gen_cap.tech_map.unique()
    if len(analysis_period)==0:
        analysis_zones = list(gen_cap.PERIOD.unique())

    analysis_gen_cap=gen_cap.copy()
    analysis_gen_cap=analysis_gen_cap.loc[(analysis_gen_cap.PERIOD.isin(analysis_period)) & (analysis_gen_cap.gen_load_zone.isin(analysis_zones))
                                          & (analysis_gen_cap.tech_map.isin(analysis_tech))]
    analysis_gen_cap_by_sc=analysis_gen_cap.pivot_table(index='scenario', columns="tech_map", values="GenCapacity_GW", aggfunc=np.sum)

    for c in tech_order:
        if not c in list(analysis_gen_cap_by_sc.columns):
            analysis_gen_cap_by_sc[c]=0

    analysis_gen_cap_by_sc = analysis_gen_cap_by_sc[tech_order]
    analysis_gen_cap_by_sc.loc[:,'Total']=analysis_gen_cap_by_sc.apply(lambda x: sum( x[c] for c in analysis_gen_cap_by_sc.columns), axis=1)
    analysis_gen_cap_by_sc['sc_order'] = analysis_gen_cap_by_sc.index.map(order)
    analysis_gen_cap_by_sc=analysis_gen_cap_by_sc.sort_values('sc_order').drop('sc_order',axis=1)

    if baseline_scenario=='':
        return analysis_gen_cap_by_sc
    else:
        relative_analysis_gen_cap_by_sc=analysis_gen_cap_by_sc.copy()
        relative_analysis_gen_cap_by_sc.loc['base',:]=relative_analysis_gen_cap_by_sc.loc[baseline_scenario,:]
        
        for t in scenario:
            r = short_names[t]
            relative_analysis_gen_cap_by_sc.loc[r]=relative_analysis_gen_cap_by_sc.loc[r]/relative_analysis_gen_cap_by_sc.loc['base']*100

        relative_analysis_gen_cap_by_sc.drop(index=('base'), inplace=True)
        relative_analysis_gen_cap_by_sc.replace(np.nan, 0, inplace=True)

        return analysis_gen_cap_by_sc, relative_analysis_gen_cap_by_sc

def tech_cap_by_zone(scenarios_file_name: str, 
                     analysis_scenario: list,
                     analysis_period: list, 
                     analysis_tech: list, analysis_zones: list, 
                     generation_length_interval: float, 
                     geodf_wecc: gpd.GeoDataFrame):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 
    gen_cap = get_data(scenario, "gen_cap.csv" , usecols=["GENERATION_PROJECT", "PERIOD", "gen_tech", "gen_load_zone", "GenCapacity"])
    gen_cap.replace({"scenario": short_names}, inplace=True)
    gen_cap.insert(5,"GenCapacity_MW",gen_cap["GenCapacity"] / 1  ) #for the maps, it is better to have it in MW.
    gen_cap.rename(columns={'PERIOD': 'period'}, inplace=True)

    if len(analysis_period)==0:
        analysis_zones = gen_cap.period.unique()
    if len(analysis_tech)==0:
        analysis_tech = gen_cap.tech_map.unique()
    if len(analysis_zones)==0:
        analysis_zones = gen_cap.gen_load_zone.unique()

    gen_cap = gen_cap.loc[gen_cap.tech_map.isin(analysis_tech) & gen_cap.period.isin(analysis_period)  & gen_cap.gen_load_zone.isin(analysis_zones)]
    
    gen_cap = gen_cap [['gen_load_zone','tech_map','period', 'GenCapacity_MW','scenario']]
    gen_cap .rename(columns={'gen_load_zone' : 'load_zone'}, inplace=True)
    gen_cap = gen_cap.pivot_table(index=['load_zone', 'tech_map', 'period', 'scenario'], values='GenCapacity_MW',  aggfunc=np.sum)
    gen_cap.reset_index(inplace=True)

    total_by_zone= gen_cap.pivot_table(index = ['scenario','load_zone'] , values = 'GenCapacity_MW', aggfunc=np.sum)
    total_by_zone.rename(columns={'GenCapacity_MW' : 'Total_by_zone'}, inplace=True)
    total_by_zone.reset_index(inplace=True)

    gen_cap = pd.merge(left=gen_cap, right=total_by_zone, on=['scenario','load_zone'])
    gen_cap ['Capacity_relative'] = gen_cap ['GenCapacity_MW']/gen_cap ['Total_by_zone']
    gen_cap = gen_cap.replace(np.nan, 0)
    gen_cap = gen_cap[gen_cap.Total_by_zone != 0]
    #gen_cap = gen_cap.loc[gen_cap.tech_map.isin(analysis_tech)] #We filter out other tech again since the "tech_map" could have other techs with zero generation.
                                                                 #However, you may comment this if you want to have the techs (not in analysis_tech) with zero
    if len(geodf_wecc)>0:
        gen_cap = pd.merge(left=gen_cap, right=geodf_wecc[['centroid_partial_can']].reset_index(), on='load_zone', how='left')

    if generation_length_interval>0:
        nintervals = int(math.floor(max(gen_cap['Total_by_zone'])/generation_length_interval))

        generation_intervals = []
        for n in range(0, nintervals+1):
            min_interval = generation_length_interval * n
            max_interval = generation_length_interval * (n+1)
            pie_radius = (n+1)/1000
            generation_intervals.append([min_interval, max_interval, pie_radius])

        generation_intervals = pd.DataFrame(generation_intervals, columns=['min_interval','max_interval', "pie_radius"])

        gen_cap.loc[:, 'zone_pie_radius'] = gen_cap.apply(lambda x: return_interval(x['Total_by_zone'], generation_intervals), axis=1).pie_radius

        return  gen_cap, generation_intervals
    
    return  gen_cap

def tech_generation(scenarios_file_name: list, 
                    analysis_scenario: list, 
                    analysis_period: list, 
                    analysis_tech: list):

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    dispatch_annual_summary = get_data(scenario, "dispatch_annual_summary.csv")
    dispatch_annual_summary.replace({"scenario": short_names}, inplace=True)

    if len(analysis_period)==0:
        analysis_period = dispatch_annual_summary.period.unique()
    if len(analysis_tech)==0:
        analysis_tech = dispatch_annual_summary.tech_map.unique()

    dispatch_annual_summary = dispatch_annual_summary.loc[dispatch_annual_summary.tech_map.isin(analysis_tech) &
                                                          dispatch_annual_summary.period.isin(analysis_period)]
    
    dispatch_annual_summary = dispatch_annual_summary[['tech_map','period', 'Energy_GWh_typical_yr','scenario']]
    dispatch_annual_summary = dispatch_annual_summary.pivot_table(index=['scenario','tech_map','period'], values=['Energy_GWh_typical_yr'], aggfunc=np.sum).reset_index(drop=False)
    total = dispatch_annual_summary.pivot_table(index='scenario', values='Energy_GWh_typical_yr', aggfunc=np.sum).reset_index(drop=False)
    total.rename(columns={'Energy_GWh_typical_yr' : 'Total_Energy_GWh_typical_yr'}, inplace=True)    
    dispatch_annual_summary = pd.merge(left = dispatch_annual_summary, right=total, on='scenario')
    
    dispatch_annual_summary['relative'] = dispatch_annual_summary['Energy_GWh_typical_yr']/dispatch_annual_summary['Total_Energy_GWh_typical_yr']*100
    
    return  dispatch_annual_summary, total

def tech_generation_by_zone_for_timestamp(scenarios_file_name: str, analysis_scenario: list, analysis_period: list, analysis_tech: list, analysis_zones: list, df_timestamps: pd.DataFrame):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    dispatch_annual_summary = get_data(scenario, 'dispatch.csv', usecols = ['timestamp', 'gen_tech', 'gen_load_zone', 'period', 'Energy_GWh_typical_yr'])

    dispatch_annual_summary.replace({"scenario": short_names}, inplace=True)
    
    if len(analysis_period)==0:
        analysis_period = dispatch_annual_summary.period.unique()
    if len(analysis_tech)==0:
        analysis_tech = dispatch_annual_summary.tech_map.unique()
    if len(analysis_zones)==0:
        analysis_zones = dispatch_annual_summary.gen_load_zone.unique()

    dispatch_annual_summary = dispatch_annual_summary.loc[dispatch_annual_summary.tech_map.isin(analysis_tech) &
                                                          dispatch_annual_summary.period.isin(analysis_period)  &
                                                          dispatch_annual_summary.gen_load_zone.isin(analysis_zones)]
    
    dispatch_annual_summary = pd.merge(left = dispatch_annual_summary, right = df_timestamps[['scenario', 'timestamp']], on=['scenario', 'timestamp'])

    dispatch_annual_summary = dispatch_annual_summary.pivot_table(index=['scenario', 'gen_tech', 'gen_load_zone', 'period', 'tech_map'], values='Energy_GWh_typical_yr', aggfunc=np.sum)
    dispatch_annual_summary.reset_index(inplace=True)

    dispatch_annual_summary = dispatch_annual_summary[['gen_load_zone','tech_map','period', 'Energy_GWh_typical_yr','scenario']]
    dispatch_annual_summary.rename(columns={'gen_load_zone' : 'load_zone'}, inplace=True)
    dispatch_annual_summary = dispatch_annual_summary.pivot_table(index=['load_zone', 'tech_map', 'period', 'scenario'], values='Energy_GWh_typical_yr',  aggfunc=np.sum)
    dispatch_annual_summary.reset_index(inplace=True)

    total_by_zone= dispatch_annual_summary.pivot_table(index = ['scenario','load_zone'] , values = 'Energy_GWh_typical_yr', aggfunc=np.sum)
    total_by_zone.rename(columns={'Energy_GWh_typical_yr' : 'Total_by_zone'}, inplace=True)
    total_by_zone.reset_index(inplace=True)

    dispatch_annual_summary = pd.merge(left=dispatch_annual_summary, right=total_by_zone, on=['scenario','load_zone'])
    dispatch_annual_summary ['Energy_relative'] = dispatch_annual_summary ['Energy_GWh_typical_yr']/dispatch_annual_summary ['Total_by_zone']
    dispatch_annual_summary = dispatch_annual_summary.replace(np.nan, 0)
    dispatch_annual_summary = dispatch_annual_summary[dispatch_annual_summary.Total_by_zone != 0]
    dispatch_annual_summary = dispatch_annual_summary.loc[dispatch_annual_summary.tech_map.isin(analysis_tech)] #We filter out other tech again since the "tech_map" could have other techs with zero generation.

    return dispatch_annual_summary

def generation_by_scenario_by_tech(scenarios_file_name: str, 
                                   analysis_scenario: list, 
                                   analysis_period: list, 
                                   analysis_tech: list):

    #Put only one analysis_period since the end table is meant for one period.
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    dispatch = tech_generation(scenario, short_names, order, analysis_period, analysis_tech)[0]
    total = tech_generation(scenario, short_names, order, analysis_period, analysis_tech)[1]

    dispatch = dispatch.pivot_table(index='scenario', columns='tech_map', values='Energy_GWh_typical_yr', aggfunc=np.sum)
    new_cols_to_add = set(tech_order) - set(dispatch.columns)  
    dispatch[list(new_cols_to_add)] = 0.0
    dispatch =  pd.DataFrame( dispatch, columns = tech_order)

    dispatch.reset_index(inplace=True)
    dispatch = pd.merge(left=dispatch, right=total, on='scenario')
    dispatch.rename(columns={'Total_Energy_GWh_typical_yr': 'Total'}, inplace=True)
    dispatch.set_index('scenario', inplace=True)

    return dispatch

def relative_generation(scenarios_file_name: str, 
                        analysis_scenario: list, 
                        analysis_period: list, 
                        only_one_analysis_tech: list, 
                        baseline_scenario: str):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    dispatch_annual_summary = get_data(scenario, "dispatch_annual_summary.csv")
    dispatch_annual_summary.replace({"scenario": short_names}, inplace=True)

    dispatch_annual_summary = dispatch_annual_summary.loc[dispatch_annual_summary.tech_map.isin(only_one_analysis_tech) &
                                                          dispatch_annual_summary.period.isin(analysis_period)] 
    dispatch_annual_summary = dispatch_annual_summary.pivot_table(index=['scenario','period'], values=['Energy_GWh_typical_yr'], aggfunc=np.sum).reset_index(drop=False)
    dispatch_annual_summary['sc_order'] = dispatch_annual_summary['scenario'].map(order)
    dispatch_annual_summary =dispatch_annual_summary .sort_values('sc_order').drop('sc_order',axis=1)
    dispatch_annual_summary['base']=dispatch_annual_summary.loc[dispatch_annual_summary.scenario==baseline_scenario,'Energy_GWh_typical_yr'].iloc[0]
    dispatch_annual_summary['relative']=dispatch_annual_summary['Energy_GWh_typical_yr']/dispatch_annual_summary['base']

    return dispatch_annual_summary

def tech_generation_by_zone(scenarios_file_name: str, 
                            analysis_scenario: list, 
                            analysis_period: list, 
                            analysis_tech: list, 
                            analysis_zones: list, 
                            generation_length_interval: float, 
                            geodf_wecc: gpd.GeoDataFrame):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    dispatch_annual_summary = get_data(scenario, "dispatch_zonal_annual_summary.csv")
    dispatch_annual_summary.replace({"scenario": short_names}, inplace=True)
    
    if len(analysis_period)==0:
        analysis_period = dispatch_annual_summary.period.unique()
    if len(analysis_tech)==0:
        analysis_tech = dispatch_annual_summary.tech_map.unique()
    if len(analysis_zones)==0:
        analysis_zones = dispatch_annual_summary.gen_load_zone.unique()

    dispatch_annual_summary = dispatch_annual_summary.loc[dispatch_annual_summary.tech_map.isin(analysis_tech) &
                                                          dispatch_annual_summary.period.isin(analysis_period)  &
                                                          dispatch_annual_summary.gen_load_zone.isin(analysis_zones)]
    dispatch_annual_summary = dispatch_annual_summary[['gen_load_zone','tech_map','period', 'Energy_GWh_typical_yr','scenario']]
    dispatch_annual_summary.rename(columns={'gen_load_zone' : 'load_zone'}, inplace=True)
    dispatch_annual_summary = dispatch_annual_summary.pivot_table(index=['load_zone', 'tech_map', 'period', 'scenario'], values='Energy_GWh_typical_yr',  aggfunc=np.sum)
    dispatch_annual_summary.reset_index(inplace=True)

    total_by_zone= dispatch_annual_summary.pivot_table(index = ['scenario','load_zone'] , values = 'Energy_GWh_typical_yr', aggfunc=np.sum)
    total_by_zone.rename(columns={'Energy_GWh_typical_yr' : 'Total_by_zone'}, inplace=True)
    total_by_zone.reset_index(inplace=True)

    dispatch_annual_summary = pd.merge(left=dispatch_annual_summary, right=total_by_zone, on=['scenario','load_zone'])
    dispatch_annual_summary ['Energy_relative'] = dispatch_annual_summary ['Energy_GWh_typical_yr']/dispatch_annual_summary ['Total_by_zone']
    dispatch_annual_summary = dispatch_annual_summary.replace(np.nan, 0)
    dispatch_annual_summary = dispatch_annual_summary[dispatch_annual_summary.Total_by_zone != 0]
    dispatch_annual_summary = dispatch_annual_summary.loc[dispatch_annual_summary.tech_map.isin(analysis_tech)] #We filter out other tech again since the "tech_map" could have other techs with zero generation.

    if len(geodf_wecc)>0:
        dispatch_annual_summary = pd.merge(left=dispatch_annual_summary, right=geodf_wecc[['centroid_partial_can']].reset_index(), on='load_zone', how='left')

    if generation_length_interval>0:
        nintervals = int(math.floor(max(dispatch_annual_summary['Total_by_zone'])/generation_length_interval))

        generation_intervals = []
        for n in range(0, nintervals+1):
            min_interval = generation_length_interval * n
            max_interval = generation_length_interval * (n+1)
            pie_radius = (n+1)/1000
            generation_intervals.append([min_interval, max_interval, pie_radius])

        generation_intervals = pd.DataFrame(generation_intervals, columns=['min_interval','max_interval', "pie_radius"])

        dispatch_annual_summary.loc[:, 'zone_pie_radius'] = dispatch_annual_summary.apply(lambda x: return_interval(x['Total_by_zone'], generation_intervals), axis=1).pie_radius

        return  dispatch_annual_summary, generation_intervals
    
    return  dispatch_annual_summary

def relative_generation_zone(scenarios_file_name: str,
                             analysis_scenario: list,
                             analysis_period: list, 
                             only_one_analysis_tech: list, 
                             zones: list, 
                             baseline_scenario: str):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 
    dispatch_annual_summary_zone = get_data(scenario, "dispatch_zonal_annual_summary.csv")
    dispatch_annual_summary_zone.replace({"scenario": short_names}, inplace=True)

    dispatch_annual_summary_zone = dispatch_annual_summary_zone.loc[dispatch_annual_summary_zone.tech_map.isin(only_one_analysis_tech) &
                                                          dispatch_annual_summary_zone.period.isin(analysis_period) 
                                                          & dispatch_annual_summary_zone.gen_load_zone.isin(zones)] 
    dispatch_annual_summary_zone = dispatch_annual_summary_zone.pivot_table(index=['scenario','period'], values=['Energy_GWh_typical_yr'], aggfunc=np.sum).reset_index(drop=False)
    dispatch_annual_summary_zone['sc_order'] = dispatch_annual_summary_zone['scenario'].map(order)
    dispatch_annual_summary_zone =dispatch_annual_summary_zone .sort_values('sc_order').drop('sc_order',axis=1)
    dispatch_annual_summary_zone['base']=dispatch_annual_summary_zone.loc[dispatch_annual_summary_zone.scenario==baseline_scenario,'Energy_GWh_typical_yr'].iloc[0]
    dispatch_annual_summary_zone['relative']=dispatch_annual_summary_zone['Energy_GWh_typical_yr']/dispatch_annual_summary_zone['base']

    return dispatch_annual_summary_zone

def relative_capacity(scenarios_file_name: str,
                      analysis_scenario: list,
                      analysis_period: list, 
                      only_one_analysis_tech: list, 
                      baseline_scenario: str):
    
    #Function that returns a dataframe with columns: 
    #scenario, period, gen_capacity (MW of the tech until the period), base (capacity MW of the baseline scenario), and relative (gen_capacity/base)

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    gen_cap_per_tech = get_data(scenario, "gen_cap_per_tech.csv")
    gen_cap_per_tech.replace({"scenario": short_names}, inplace=True)

    gen_cap_per_tech = gen_cap_per_tech.loc[(gen_cap_per_tech.tech_map.isin(only_one_analysis_tech)) &
                                                          (gen_cap_per_tech.period.isin(analysis_period))] 
    gen_cap_per_tech = gen_cap_per_tech.pivot_table(index=['scenario','period'], values=['gen_capacity'], aggfunc=np.sum).reset_index(drop=False)
    gen_cap_per_tech['sc_order'] = gen_cap_per_tech['scenario'].map(order)
    gen_cap_per_tech =gen_cap_per_tech .sort_values('sc_order').drop('sc_order',axis=1)
    gen_cap_per_tech['base']=gen_cap_per_tech.loc[gen_cap_per_tech.scenario==baseline_scenario,'gen_capacity'].iloc[0]
    gen_cap_per_tech['relative']=gen_cap_per_tech['gen_capacity']/gen_cap_per_tech['base']

    return gen_cap_per_tech

def relative_capacity_zone(scenarios_file_name: str,
                           analysis_scenario: str, 
                           analysis_period: list, 
                           only_one_analysis_tech: list, 
                           zones: list, 
                           baseline_scenario: str):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    gen_cap_per_tech_per_zone = get_data(scenario, "gen_cap.csv")
    gen_cap_per_tech_per_zone.replace({"scenario": short_names}, inplace=True)

    gen_cap_per_tech_per_zone = gen_cap_per_tech_per_zone.loc[(gen_cap_per_tech_per_zone.tech_map.isin(only_one_analysis_tech)) &
                                                          (gen_cap_per_tech_per_zone.PERIOD.isin(analysis_period)) & (gen_cap_per_tech_per_zone.gen_load_zone.isin(zones))] 
    gen_cap_per_tech_per_zone = gen_cap_per_tech_per_zone.pivot_table(index=['scenario','PERIOD'], values=['GenCapacity'], aggfunc=np.sum).reset_index(drop=False)
    gen_cap_per_tech_per_zone['sc_order'] = gen_cap_per_tech_per_zone['scenario'].map(order)
    gen_cap_per_tech_per_zone =gen_cap_per_tech_per_zone .sort_values('sc_order').drop('sc_order',axis=1)
    gen_cap_per_tech_per_zone['base']=gen_cap_per_tech_per_zone.loc[gen_cap_per_tech_per_zone.scenario==baseline_scenario,'GenCapacity'].iloc[0]
    gen_cap_per_tech_per_zone['relative']=gen_cap_per_tech_per_zone['GenCapacity']/gen_cap_per_tech_per_zone['base']

    return gen_cap_per_tech_per_zone

def check_generation_ready_to_use(scenarios_file_name: str, 
                                  analysis_scenarios: list,
                                  analysis_zones: list):
    # Function to check how much power capacity by technology is ready to use in the horizon of optimization.
    # Output:
    # scenario	gen_load_zone	tech_map	capacity_to_use
    # -----------------------------------------------------
    # high_10	CA_IID	        Gas	        152.30
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 
        
    gen_build_predetermined = get_data(scenario, "gen_build_predetermined.csv", fpath="inputs", usecols=['GENERATION_PROJECT', 'build_year', 'gen_predetermined_cap'])
    gen_build_predetermined = gen_build_predetermined.replace({"scenario": short_names})

    generation_projects_info = get_data(scenario, "generation_projects_info.csv", fpath="inputs", usecols=['GENERATION_PROJECT', 'gen_load_zone', 'gen_tech', 'gen_energy_source', 'gen_max_age'])
    generation_projects_info = generation_projects_info.replace({"scenario": short_names})
    
    hours_per_year = 8766

    timepoints = get_data(scenario, "timepoints.csv", fpath='inputs')
    timepoints.columns= timepoints .columns.str.lower()
    timepoints.rename(columns={'timepoint_id': 'timepoint'}, inplace=True)
    timepoints = timepoints .replace({"scenario": short_names})

    timeseries = get_data(scenario, "timeseries.csv", fpath='inputs')
    timeseries.columns= timeseries .columns.str.lower()
    timeseries = timeseries .replace({"scenario": short_names})

    periods = get_data(scenario, "periods.csv", fpath='inputs')
    periods.columns= periods .columns.str.lower()
    periods.rename(columns={'investment_period': 'ts_period'}, inplace=True)
    periods = periods .replace({"scenario": short_names})

    time_info=pd.merge(left=timeseries,right=timepoints, on=['timeseries', 'scenario'])
    time_info=pd.merge(left=time_info,right=periods , on=['scenario', 'ts_period'])

    time_info['tp_weight']=time_info['ts_duration_of_tp']*time_info['ts_scale_to_period']

    period_info=time_info.pivot_table(index=['scenario','ts_period'], values='tp_weight',aggfunc=np.sum )
    period_info.reset_index(inplace=True)
    period_info.rename(columns={'tp_weight': 'hours_in_period'}, inplace=True)
    period_info = pd.merge(left=period_info, right = periods, on=['scenario', 'ts_period'])
    period_info['err_plain'] = (period_info['period_end'] - period_info['period_start'])*hours_per_year - period_info['hours_in_period']
    period_info['err_add_one'] =  (period_info['period_end'] + 1 - period_info['period_start'])*hours_per_year - period_info['hours_in_period']
    period_info.loc[:, 'add_one_to_period_end_rule']= period_info.apply(lambda x: 1 if np.absolute(x['err_add_one'])<np.absolute(x['err_plain']) else 0, axis=1)
    period_info['period_length_years'] = period_info['period_end'] - period_info['period_start'] + period_info['add_one_to_period_end_rule']

    period_info.rename(columns={'ts_period': 'investment_period'}, inplace=True)

    BLD_YRS_FOR_GEN_PERIOD = pd.merge(left=generation_projects_info[['GENERATION_PROJECT', 'gen_load_zone', 'gen_max_age', 'scenario']], right=period_info[['investment_period', 'period_start', 'period_length_years','scenario']], on='scenario')
    BLD_YRS_FOR_GEN_PERIOD = pd.merge(left = gen_build_predetermined, right = BLD_YRS_FOR_GEN_PERIOD, on = ['GENERATION_PROJECT', 'scenario'])
    BLD_YRS_FOR_GEN_PERIOD.loc[:,'operation']=BLD_YRS_FOR_GEN_PERIOD.apply(lambda x: gen_build_can_operate_in_period(x['gen_max_age'], x['build_year'], x['investment_period'], x['period_start'], x['period_length_years']), axis=1)  
    BLD_YRS_FOR_GEN_PERIOD['capacity_to_use'] = BLD_YRS_FOR_GEN_PERIOD['gen_predetermined_cap'] * BLD_YRS_FOR_GEN_PERIOD['operation']

    full_BLD_YRS_FOR_GEN_PERIOD = BLD_YRS_FOR_GEN_PERIOD.copy()

    BLD_YRS_FOR_GEN_PERIOD = BLD_YRS_FOR_GEN_PERIOD[['scenario', 'GENERATION_PROJECT', 'gen_load_zone', 'capacity_to_use']]

    if len(analysis_zones) != 0:
        BLD_YRS_FOR_GEN_PERIOD = BLD_YRS_FOR_GEN_PERIOD.loc[BLD_YRS_FOR_GEN_PERIOD.gen_load_zone.isin(analysis_zones)]
    
    BLD_YRS_FOR_GEN_PERIOD = pd.merge( left = BLD_YRS_FOR_GEN_PERIOD, right = generation_projects_info[['GENERATION_PROJECT', 'tech_map', 'scenario']], on = ['scenario', 'GENERATION_PROJECT'])

    already_built_capacity_to_use = BLD_YRS_FOR_GEN_PERIOD.pivot_table(index=['scenario', 'gen_load_zone', 'tech_map'], values = 'capacity_to_use', aggfunc=np.sum)
    already_built_capacity_to_use.reset_index(inplace=True)

    return already_built_capacity_to_use , full_BLD_YRS_FOR_GEN_PERIOD

def tech_generation_by_special_zones(scenario_file_name:str, 
                                     analysis_scenarios: list, 
                                     special_zones: pd.DataFrame, 
                                     analysis_period:list, 
                                     analysis_tech:list, 
                                     wecc_special:gpd.GeoDataFrame):

    states = list(special_zones.state.unique())

    wecc_generation = pd.DataFrame(data={'load_zone': ['zero'],
                                         'tech_map':['zero'],
                                         'Energy_GWh_typical_yr': 0,
                                         'Total_by_zone': 0,
                                         'Energy_relative': 0,
                                         'period':0,
                                         'scenario': 'zero',
                                            })

    for s in states:
        analysis_zones = list(special_zones.loc[special_zones.state==s, 'load_zone'])
        state_generation = tech_generation_by_zone(scenario_file_name , analysis_scenarios, analysis_period, analysis_tech, analysis_zones, 0, [])
        state_generation = state_generation[['load_zone', 'tech_map', 'period', 'scenario', 'Energy_GWh_typical_yr']]
        state_generation = state_generation.pivot_table(index=['tech_map', 'period', 'scenario'], values ='Energy_GWh_typical_yr', aggfunc=np.sum)
        state_generation.reset_index(inplace=True)
        total_generation = state_generation.pivot_table(index=['scenario', 'period'], values='Energy_GWh_typical_yr', aggfunc=np.sum).reset_index()
        state_generation = pd.merge(left = state_generation, right = total_generation, on=['scenario', 'period'])
        state_generation.rename(columns = {'Energy_GWh_typical_yr_x': 'Energy_GWh_typical_yr', 'Energy_GWh_typical_yr_y': 'Total_by_zone'}, inplace=True)
        state_generation['Energy_relative'] = state_generation['Energy_GWh_typical_yr']/state_generation['Total_by_zone']
        state_generation['load_zone'] = s
        wecc_generation = pd.concat([wecc_generation, state_generation])

    wecc_generation = wecc_generation.iloc[1:]
    wecc_generation = wecc_generation.replace(np.nan, 0)
    wecc_generation = wecc_generation[wecc_generation.Total_by_zone != 0]

    if len(wecc_special)>0:
        wecc_generation = pd.merge(left=wecc_generation, right=wecc_special.reset_index()[['load_zone','centroid_partial_can']], on='load_zone')

    return wecc_generation

def tech_generation_by_special_zones_for_tps(scenario_file_name:str, 
                                             analysis_scenarios: list, 
                                             special_zones: pd.DataFrame, 
                                             analysis_zones: list,
                                             analysis_period:list, 
                                             analysis_tech:list, 
                                             wecc_special:gpd.GeoDataFrame, 
                                             df_timestamps: pd.DataFrame):
    
    states = list(special_zones.state.unique())

    wecc_generation = pd.DataFrame(data={'load_zone': ['zero'],
                                         'tech_map':['zero'],
                                         'Energy_GWh_typical_yr': 0,
                                         'Total_by_zone': 0,
                                         'Energy_relative': 0,
                                         'period':0,
                                         'scenario': 'zero',
                                            })

    tech_gen = tech_generation_by_zone_for_timestamp(scenario_file_name, analysis_scenarios, analysis_period, analysis_tech, analysis_zones, df_timestamps)

    for s in states:
        zones = list(special_zones.loc[special_zones.state==s, 'load_zone'])
        state_generation = tech_gen.copy()
        state_generation = state_generation.loc[state_generation.load_zone.isin(zones)]
        state_generation = state_generation[['load_zone', 'tech_map', 'period', 'scenario', 'Energy_GWh_typical_yr']]
        state_generation = state_generation.pivot_table(index=['tech_map', 'period', 'scenario'], values ='Energy_GWh_typical_yr', aggfunc=np.sum)
        state_generation.reset_index(inplace=True)
        total_generation = state_generation.pivot_table(index=['scenario', 'period'], values='Energy_GWh_typical_yr', aggfunc=np.sum).reset_index()
        state_generation = pd.merge(left = state_generation, right = total_generation, on=['scenario', 'period'])
        state_generation.rename(columns = {'Energy_GWh_typical_yr_x': 'Energy_GWh_typical_yr', 'Energy_GWh_typical_yr_y': 'Total_by_zone'}, inplace=True)
        state_generation['Energy_relative'] = state_generation['Energy_GWh_typical_yr']/state_generation['Total_by_zone']
        state_generation['load_zone'] = s
        wecc_generation = pd.concat([wecc_generation, state_generation])

    wecc_generation = wecc_generation.iloc[1:]
    wecc_generation = wecc_generation.replace(np.nan, 0)
    wecc_generation = wecc_generation[wecc_generation.Total_by_zone != 0]

    if len(wecc_special)>0:
        wecc_generation = pd.merge(left=wecc_generation, right=wecc_special.reset_index()[['load_zone','centroid_partial_can']], on='load_zone')

    return wecc_generation

def stacked_dispatch(scenario_file_name:str,
                     analysis_scenario: list, #only one scenario
                     analysis_zones: list, 
                     time_zone, start_date: str, 
                     end_date: str, periods: list, 
                     transmission: int):

    # Dataframe that gives the average monthly dispatch for only one period.

    # Output 1: full_daily_dispatch, 
    #tech_map	Wind	Solar	Gas	    Oil	    Coal	Biomass	Geothermal	Storage	Hydro	Nuclear	Waste
    #hour											
    #0	        35.22	2.91	0.00	0.00	0.00	1.52	1.06	    317.36	165.25	19.59	0.05
    #1	        53.16	0.00	0.00	0.00	0.00	1.51	1.06	    203.97	145.33	19.59	0.04
    #2	        53.16	0.00	0.00	0.00	0.00	1.51	1.06	    203.97	145.33	19.59	0.04
    #3	        53.16	0.00	0.00	0.00	0.00	1.51	1.06	    203.97	145.33	19.59	0.04
    #4	        53.16	0.00	0.00	0.00	0.00	1.51	1.06	    203.97	145.33	19.59	0.04

    # Output 2: monthly_dispatch 
    # tech_map	Wind	    Solar	    Gas	    Oil	    Coal	Biomass	Geothermal	Storage	    Hydro	    Nuclear	    Waste
    # month											
    # 1	        14,988.50	54,336.40	0.00	0.00	0.00	166.10	193.65	    20,095.28	20,847.33	3,566.04	0.00
    # 2	        12,083.46	52,399.49	0.00	0.00	0.00	187.09	178.76	    19,855.73	18,706.71	3,291.73	0.00
    # 3	        9,363.20	58,043.73	0.00	0.00	0.00	196.11	197.91	    21,184.55	21,060.05	3,644.42	0.00
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenario_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    time_1=datetime.strptime(str(start_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    time_2=datetime.strptime(str(end_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))

    #Get energy dispatch
    dpch= get_data(scenario, "dispatch.csv", usecols=["generation_project", "timestamp", "gen_tech", "gen_load_zone", "tp_weight_in_year_hrs", "DispatchGen_MW"])
    dpch.replace({"scenario": short_names}, inplace=True)

    if len(analysis_zones)==0:
        analysis_zones= list(dpch.gen_load_zone.unique())

    dpch=dpch.loc[dpch.gen_load_zone.isin(analysis_zones)]
    dpch['DispatchGen_MWh']=dpch['DispatchGen_MW']*dpch['tp_weight_in_year_hrs']

    #Get charging energy
    stg= get_data(scenario, "storage_dispatch.csv")
    stg.replace({"scenario": short_names},inplace=True)
    stg=stg.loc[stg.load_zone.isin(analysis_zones)]
    stg.rename({'timepoint': 'timestamp', 'load_zone':'gen_load_zone', 'ChargeMW': 'DispatchGen_MW'}, axis=1, inplace=True)
    stg.drop(['datetime', 'DischargeMW', 'StateOfCharge'], axis=1, inplace=True)
    stg['gen_tech']='Battery_Storage_Charge'
    stg['tech_map']='Storage (C)'
    stg=pd.merge(left=stg, right=dpch[['generation_project','timestamp', 'tp_weight_in_year_hrs', 'scenario']], on=['generation_project','timestamp', 'scenario'], how='inner', copy='False')
    stg['DispatchGen_MWh']=stg['DispatchGen_MW']*stg['tp_weight_in_year_hrs']
    stg['DispatchGen_MW']=stg['DispatchGen_MW']*-1
    stg['DispatchGen_MWh']=stg['DispatchGen_MWh']*-1

    z_dpch=pd.concat([dpch, stg])

    if transmission==1:
        tx_dispatch = get_data(scenario, "transmission_dispatch.csv", usecols=['load_zone_from', 'load_zone_to', 'timestamp','transmission_dispatch'])
        tx_dispatch.replace({"scenario": short_names}, inplace=True)

        txs = get_data(scenario, "transmission_lines.csv", fpath='inputs')
        txs.replace({"scenario": short_names}, inplace=True)
        txs = txs[['trans_lz1', 'trans_lz2', 'scenario']]
        txs = txs.loc[((txs.trans_lz1.isin(analysis_zones)) ^ (txs.trans_lz2.isin(analysis_zones)))]
        txs_copy=txs.copy() #Duplicate the table to have NM in the sending zone and receiving zone
        txs_copy.rename(columns={'trans_lz1':'trans_lz2', 'trans_lz2':'trans_lz1'}, inplace=True)
        txs=pd.concat([txs,txs_copy])
        txs.reset_index(inplace=True, drop=True)
        txs.rename(columns={'trans_lz1': 'load_zone_from', 'trans_lz2': 'load_zone_to'}, inplace=True)

        tx_flow=pd.merge(txs, tx_dispatch, how='inner', on=['load_zone_from', 'load_zone_to', 'scenario']) 
        tx_flow['ex_im_sign']=np.where(tx_flow['load_zone_from'].isin(analysis_zones), -1, 1)
        tx_flow['tech_map']=np.where(tx_flow['ex_im_sign'] == -1, 'Exports', 'Imports') #-1 if it is exporting

        time=dpch[['timestamp', 'scenario', 'tp_weight_in_year_hrs']].copy()
        time.drop_duplicates(inplace=True)

        tx_flow=pd.merge(tx_flow, time, how='left', on=['timestamp', 'scenario'])
        tx_flow.rename({'transmission_dispatch':'DispatchGen_MW'}, axis=1, inplace=True)
        tx_flow['DispatchGen_MW'] = tx_flow['DispatchGen_MW']*tx_flow['ex_im_sign']
        tx_flow['DispatchGen_MWh']=tx_flow['DispatchGen_MW']*tx_flow['tp_weight_in_year_hrs']
        tx_flow['gen_load_zone']=np.where(tx_flow['load_zone_from'].isin(analysis_zones), tx_flow['load_zone_from'], tx_flow['load_zone_to'])
        tx_flow['generation_project']=tx_flow['load_zone_from']+"---"+tx_flow['load_zone_to']
        tx_flow['gen_tech']=tx_flow['tech_map']
        tx_flow.drop(['load_zone_from', 'load_zone_to', 'ex_im_sign'], axis=1, inplace=True)
        z_dpch=pd.concat([z_dpch, tx_flow])

    timepoints = get_data(scenario, "timepoints.csv", fpath='inputs', usecols=['timestamp', 'timeseries'])
    timeseries = get_data(scenario, "timeseries.csv", fpath='inputs', usecols=['TIMESERIES', 'ts_period'])
    timeseries.rename(columns={'TIMESERIES':'timeseries'}, inplace=True)
    timestamp_period = pd.merge(left=timepoints, right=timeseries, on=['timeseries', 'scenario'])
    timestamp_period = timestamp_period[['timestamp', 'ts_period']]

    z_dpch = pd.merge(left=z_dpch, right=timestamp_period, on='timestamp')

    z_dpch["timestamp"]=pd.to_datetime(z_dpch["timestamp"], format='%Y%m%d%H', utc=True)
    z_dpch["timestamp"]=z_dpch["timestamp"].dt.tz_convert(time_zone)
    z_dpch['DispatchGen_GWh']=z_dpch['DispatchGen_MWh']*10**(-3)

    a_1=z_dpch.copy()

    a_1['hour']=a_1['timestamp'].dt.hour  #Create a column that has the hour of the timestamp
    a_1=a_1[(time_1 <= a_1.timestamp) & (a_1.timestamp < time_2)]  

    daily_dispatch_source= a_1.pivot_table(index="hour", columns='tech_map', values="DispatchGen_GWh", aggfunc=np.sum)
    #Add the technologies that are in tech order but have dispatch zero. This serves to order the stacked area plot according to the tech order vector
    for k in tech_order:
        if not k in daily_dispatch_source.columns:
            daily_dispatch_source[k]=0

    daily_dispatch_source = daily_dispatch_source / ((time_2 - time_1).days) #Daily average

    #New dataframe
    full_daily_dispatch=daily_dispatch_source.copy()
    full_daily_dispatch=full_daily_dispatch[tech_order]

    #Replicate the rows to fill up the hours of the time interval. For example, if the time interval comprises 4 hours, then copy the result of 
    #hour and replicate them.
    #Number of hours of a time block
    time_range=4 
    for h in list(daily_dispatch_source.index):
        for k in range(0,time_range):
            full_daily_dispatch.loc[h+k,:]=full_daily_dispatch.loc[h,:]

    #If the hour index is greater than 24, the row is reindex with 24 or h-24, and then it is dropped.
    for h in list(full_daily_dispatch.index):
        if h>24: #use >= if you use h-24, if not use h>24
            full_daily_dispatch.loc[24,:]=full_daily_dispatch.loc[h,:] #you may use h-24, too. Try
            full_daily_dispatch.drop(index=h, inplace=True)

    #Sort the dataframe to have it chronologic
    full_daily_dispatch.sort_index(inplace=True)

    b_1=z_dpch.copy()

    if len(periods)==0:
        periods = list(b_1.ts_period.unique())
    
    b_1 = b_1.loc[b_1.ts_period.isin(periods)]
    
    b_1['month']=b_1['timestamp'].dt.month  #Create a column that has the month of the timestamp
    monthly_dispatch= b_1.pivot_table(index="month", columns='tech_map', values="DispatchGen_GWh", aggfunc=np.sum)
    for k in tech_order:
        if not k in monthly_dispatch.columns:
            monthly_dispatch[k]=0
            
    monthly_dispatch=monthly_dispatch[tech_order]
    return full_daily_dispatch, monthly_dispatch

def gen_df_to_plot_in_map(zone_annual_gen:pd. DataFrame, sc_plot: pd.DataFrame, select_scenario_by_index: int):

    # Inputs: 
    #       zone_annual_gen:
    #                           load_zone	tech_map	Energy_GWh_typical_yr	Total_by_zone	Energy_relative	period	scenario	centroid_partial_can
    #                           ------------------------------------------------------------------------------------------------------------------------
    #     
    #       sc_plot: 
    #                           scenarios  title
    #                           --------------------------------
    #                           high_100   325 \$/kW, 275 \$/kWh
    #       
    #      select_scenario_by_index: 0
    #  
    # Output: 
    #   Load zone | Biomass	    Coal	Gas	Geothermal	Hydro	Nuclear	Solar	Storage	Waste	Wind	Total_by_zone	centroid_partial_can	Oil	log_Total_by_zone
    #   -------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 
     
    zone_annual_gen_1 = zone_annual_gen.loc[zone_annual_gen.scenario == sc_plot.loc[select_scenario_by_index,'scenarios']]

    techs_in_df = list(zone_annual_gen_1['tech_map'].unique())
    missing_techs = set(tech_order) - set(techs_in_df)

    zone_annual_gen_1_by_tech = zone_annual_gen_1 .pivot_table(index='load_zone', columns='tech_map', values='Energy_relative')
    zone_annual_gen_1_by_tech = pd.merge(left =zone_annual_gen_1_by_tech, right=zone_annual_gen_1[['load_zone', 'Total_by_zone', 'centroid_partial_can']].drop_duplicates().set_index('load_zone'), left_index=True, right_index=True)

    for i in missing_techs:
        zone_annual_gen_1_by_tech[i]=0.00
    zone_annual_gen_1_by_tech['log_Total_by_zone'] = np.log(zone_annual_gen_1_by_tech['Total_by_zone'])

    return zone_annual_gen_1_by_tech


def gencap_df_to_plot_in_map(gencap:pd. DataFrame, sc_plot: pd.DataFrame, select_scenario_by_index: int):

    # Inputs: 
    #       gencap:
    #                           load_zone	tech_map period scenario	GenCapacity_MW	Total_by_zone	Capacity_relative	centroid_partial_can
    #                           ----------------------------------------------------------------------------------------------------------------------------
    #                           AZ_APS_E	Biomass	2035	wecc_10_10	10.00	        926.20	        0.01	            POINT (-110.09621 34.38540)
    #       sc_plot: 
    #                           scenarios   title
    #                           --------------------------------
    #                           wecc_10_10   325 \$/kW, 275 \$/kWh
    #       
    #      select_scenario_by_index: 0
    #  
    # Output: 
    #   Load zone | Biomass	Coal	Gas	Geothermal	Hydro	Nuclear	Oil	Solar	Storage	Waste	Wind	Total_by_zone	centroid_partial_can	log_Total_by_zone
    #   -------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 
     
    gencap_1 = gencap.loc[gencap.scenario == sc_plot.loc[select_scenario_by_index,'scenarios']]

    techs_in_df = list(gencap_1['tech_map'].unique())
    missing_techs = set(tech_order) - set(techs_in_df)

    gencap_1_by_tech = gencap_1 .pivot_table(index='load_zone', columns='tech_map', values='Capacity_relative')
    gencap_1_by_tech = pd.merge(left =gencap_1_by_tech, right=gencap_1[['load_zone', 'Total_by_zone', 'centroid_partial_can']].drop_duplicates().set_index('load_zone'), left_index=True, right_index=True)

    for i in missing_techs:
        gencap_1_by_tech[i]=0.00
    gencap_1_by_tech['log_Total_by_zone'] = np.log(gencap_1_by_tech['Total_by_zone'])

    return gencap_1_by_tech

# --------------------------------------------------             ---------------------------------------------------------------------------------------------------------
# -------------------------------------------------- CURTAILMENT ------------------------------------------------------------------------------------------------
# --------------------------------------------------             --------------------------------------------------------------------------------------------------------

def tech_curtailment_by_zone(scenarios_file_name: str, 
                             analysis_scenario: list, 
                             analysis_period: list, 
                             analysis_tech: list, 
                             analysis_zones: list, 
                             generation_length_interval: float, 
                             geodf_wecc: gpd.GeoDataFrame):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 
        
    curtailment_annual_summary = get_data(scenario, "curtailment_zonal_annual_summary.csv")
    curtailment_annual_summary.replace(short_names, inplace=True)
    
    if len(analysis_period)==0:
        analysis_period = curtailment_annual_summary.period.unique()
    if len(analysis_tech)==0:
        analysis_tech = curtailment_annual_summary.tech_map.unique()
    if len(analysis_zones)==0:
        analysis_zones = curtailment_annual_summary.gen_load_zone.unique()

    curtailment_annual_summary = curtailment_annual_summary.loc[curtailment_annual_summary.tech_map.isin(analysis_tech) &
                                                          curtailment_annual_summary.period.isin(analysis_period)  &
                                                          curtailment_annual_summary.gen_load_zone.isin(analysis_zones)
                                                        ]
    curtailment_annual_summary = curtailment_annual_summary[['gen_load_zone','tech_map','period', 'Curtailment_GWh_typical_yr','scenario']]
    curtailment_annual_summary.rename(columns={'gen_load_zone' : 'load_zone'}, inplace=True)
    curtailment_annual_summary = curtailment_annual_summary.pivot_table(index=['load_zone', 'tech_map', 'period', 'scenario'], values='Curtailment_GWh_typical_yr',  aggfunc=np.sum)
    curtailment_annual_summary.reset_index(inplace=True)
    
    total_by_zone= curtailment_annual_summary.pivot_table(index = ['scenario','load_zone'], values = 'Curtailment_GWh_typical_yr', aggfunc=np.sum)
    total_by_zone.rename(columns={'Curtailment_GWh_typical_yr' : 'Total_by_zone'}, inplace=True)
    total_by_zone.reset_index(inplace=True)

    curtailment_annual_summary = pd.merge(left=curtailment_annual_summary, right=total_by_zone, on=['scenario','load_zone'])
    curtailment_annual_summary ['Energy_relative'] = curtailment_annual_summary ['Curtailment_GWh_typical_yr']/curtailment_annual_summary ['Total_by_zone']
    curtailment_annual_summary = curtailment_annual_summary.replace(np.nan, 0)
    curtailment_annual_summary = curtailment_annual_summary[curtailment_annual_summary.Total_by_zone != 0]
    curtailment_annual_summary = curtailment_annual_summary.loc[curtailment_annual_summary.tech_map.isin(analysis_tech)] #We filter out other tech again since the "tech_map" could have other techs.
    
    return curtailment_annual_summary

def relative_curtailment(scenarios_file_name: str, 
                         analysis_scenario: list, 
                         analysis_period: list, 
                         analysis_tech: list, 
                         baseline_scenario: str):
    # Return total curtailment for the selected technologies and selected periods. It gives curtailment relative to the baseline scenario.
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    curtailment_annual_summary = get_data(scenario, "curtailment_zonal_annual_summary.csv")
    curtailment_annual_summary.replace(short_names, inplace=True)
    curtailment_annual_summary = curtailment_annual_summary.pivot_table(index=['tech_map', 'period', 'scenario'], values = 'Curtailment_GWh_typical_yr', aggfunc = np.sum)
    curtailment_annual_summary.reset_index(inplace=True)
    curtailment_annual_summary = curtailment_annual_summary.loc[curtailment_annual_summary.tech_map.isin(analysis_tech) & curtailment_annual_summary.period.isin(analysis_period)]
    curtailment_annual_summary = curtailment_annual_summary.pivot_table(index='scenario', values='Curtailment_GWh_typical_yr', aggfunc = np.sum)
    curtailment_annual_summary.reset_index(inplace=True)
    curtailment_annual_summary['sc_order'] = curtailment_annual_summary['scenario'].map(order)
    curtailment_annual_summary =curtailment_annual_summary .sort_values('sc_order').drop('sc_order',axis=1)
    curtailment_annual_summary['base']=curtailment_annual_summary.loc[curtailment_annual_summary.scenario==baseline_scenario,'Curtailment_GWh_typical_yr'].iloc[0]
    curtailment_annual_summary['relative']=curtailment_annual_summary['Curtailment_GWh_typical_yr']/curtailment_annual_summary['base']

    return curtailment_annual_summary

def curtailment_in_percentage(scenarios_file_name: str, 
                              analysis_scenario: list, 
                              analysis_period: list, 
                              analysis_tech: list):
    #return the curtailment relative to dispatch plus curtailment. Patricia suggested this operation when we were working with Priya.
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario]
    
    curtailment_annual_summary = get_data(scenario, "curtailment_zonal_annual_summary.csv")
    curtailment_annual_summary.replace(short_names, inplace=True)
    curtailment_annual_summary = curtailment_annual_summary.pivot_table(index=['tech_map', 'period', 'scenario'], values = 'Curtailment_GWh_typical_yr', aggfunc = np.sum)
    curtailment_annual_summary.reset_index(inplace=True)
    

    dispatch_annual_summary = get_data(scenario, "dispatch_annual_summary.csv")
    dispatch_annual_summary.replace(short_names, inplace=True)
    dispatch_annual_summary = dispatch_annual_summary.pivot_table(index=['tech_map', 'period', 'scenario'], values = 'Energy_GWh_typical_yr', aggfunc = np.sum)
    dispatch_annual_summary.reset_index(inplace=True)

    curt_and_dispatch = pd.merge(left = curtailment_annual_summary, right = dispatch_annual_summary, on=['tech_map', 'period', 'scenario'])
    curt_and_dispatch = curt_and_dispatch.loc[curt_and_dispatch.tech_map.isin(analysis_tech) & curt_and_dispatch.period.isin(analysis_period)]
    curt_and_dispatch['sc_order'] = curt_and_dispatch['scenario'].map(order)
    curt_and_dispatch = curt_and_dispatch .sort_values('sc_order').drop('sc_order',axis=1)

    curt_and_dispatch_by_sc = curt_and_dispatch.pivot_table(index = ['scenario', 'period'], values = ['Curtailment_GWh_typical_yr', 'Energy_GWh_typical_yr'], aggfunc = np.sum)
    curt_and_dispatch_by_sc.reset_index(inplace=True)

    curt_and_dispatch['Curtailment_TWh'] = curt_and_dispatch['Curtailment_GWh_typical_yr']/10**3
    curt_and_dispatch['Dispatch_TWh'] = curt_and_dispatch['Energy_GWh_typical_yr']/10**3

    curt_and_dispatch_by_sc['Curtailment_TWh'] = curt_and_dispatch_by_sc['Curtailment_GWh_typical_yr']/10**3
    curt_and_dispatch_by_sc['Dispatch_TWh'] = curt_and_dispatch_by_sc['Energy_GWh_typical_yr']/10**3

    curt_and_dispatch_by_sc.drop(['Curtailment_GWh_typical_yr', 'Energy_GWh_typical_yr'], axis=1, inplace=True)
    curt_and_dispatch.drop(['Curtailment_GWh_typical_yr', 'Energy_GWh_typical_yr'], axis=1, inplace=True)

    curt_and_dispatch['Curtailment_relative_to_total_energy'] = curt_and_dispatch['Curtailment_TWh'] / (curt_and_dispatch['Curtailment_TWh'] + curt_and_dispatch['Dispatch_TWh']) * 100
    curt_and_dispatch_by_sc['Curtailment_relative_to_total_energy'] = curt_and_dispatch_by_sc['Curtailment_TWh'] / (curt_and_dispatch_by_sc['Curtailment_TWh'] + curt_and_dispatch_by_sc['Dispatch_TWh']) * 100

    return curt_and_dispatch, curt_and_dispatch_by_sc

# ----------------------------------------------
# ---------------------------------------------- IMPORTS, EXPORTS, RATIOS ------------------------------------------------------------------------------------------
# ----------------------------------------------                          ------------------------------------------------------------------------------------------


def xch_ratio_timeseries(scenarios_file_name: str, analysis_scenarios: list, zones: list):
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 
    
    load_balance = get_data(scenario, "load_balance.csv")
    load_balance  = load_balance  .replace({"scenario": short_names})
    load_balance .columns= load_balance .columns.str.lower()
    load_balance['withdrawal']=-(load_balance['zone_demand_mw'] + load_balance['storagenetcharge'])
    load_balance.rename(columns={'zonetotalcentraldispatch': 'injection'}, inplace=True)
    load_balance.drop(columns=['normalized_energy_balance_duals_dollar_per_mwh', 'txpowernet', 'zone_demand_mw', 'storagenetcharge'], inplace=True)

    load_balance=load_balance.loc[load_balance['load_zone'].isin(zones)]

    ratio_timeseries = load_balance.pivot_table(
    index=["scenario", "timestamp"], values=["injection", "withdrawal"], aggfunc=np.sum )

    ratio_timeseries['ratio'] = ratio_timeseries['injection']/ratio_timeseries['withdrawal']
    ratio_timeseries.reset_index(inplace=True)

    ratio_timeseries['sc_order'] = ratio_timeseries['scenario'].map(order)
    ratio_timeseries = ratio_timeseries .sort_values('sc_order').drop('sc_order',axis=1)

    return ratio_timeseries


def annual_exchange_ratio_zone(scenarios_file_name: str, 
                               analysis_scenario: list,
                               zones: list):

    #Returns annual exchange ratio. We do not use the duration of the timestamps because we assume that the timestamps have same duration.
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 
       
    load_balance = get_data(scenario, "load_balance.csv")
    load_balance  = load_balance  .replace({"scenario": short_names})
    load_balance .columns= load_balance .columns.str.lower()
    load_balance['withdrawal']=-(load_balance['zone_demand_mw'] + load_balance['storagenetcharge'])
    load_balance.rename(columns={'zonetotalcentraldispatch': 'injection'}, inplace=True)
    load_balance.drop(columns=['normalized_energy_balance_duals_dollar_per_mwh', 'txpowernet', 'zone_demand_mw', 'storagenetcharge'], inplace=True)
    zone_power_balance=load_balance.copy()
    zone_power_balance=zone_power_balance.loc[zone_power_balance['load_zone'].isin(zones)]

    annual_zone_power_balance_by_scenario = zone_power_balance.pivot_table(
    index=["scenario"], values=["injection", "withdrawal"], aggfunc=np.sum )
    annual_zone_power_balance_by_scenario['ratio']=annual_zone_power_balance_by_scenario['injection']/annual_zone_power_balance_by_scenario['withdrawal']
    annual_zone_power_balance_by_scenario.reset_index(inplace=True)

    annual_zone_power_balance_by_scenario['sc_order'] = annual_zone_power_balance_by_scenario['scenario'].map(order)
    annual_zone_power_balance_by_scenario = annual_zone_power_balance_by_scenario .sort_values('sc_order').drop('sc_order',axis=1)
    
    return annual_zone_power_balance_by_scenario

def monthly_exchange_ratio_at_a_given_period(scenarios_file_name: str, 
                                             analysis_scenario: list,
                                             zones: list, 
                                             analysis_period: list):
    #Warning:  We do not use the duration of the timestamps because we assume that the timestamps have same duration.
    
    # Output: 
    # 
    # month     |	low_5	low_10	low_25	low_50	low_75	low_100	    mid_5																		
    # 1	            0.85	0.86	0.83	0.80	0.78	0.76	    0.84
    # 2	            0.92	0.92	0.89	0.85	0.83	0.81	    0.90
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    load_balance = get_data(scenario, "load_balance.csv")
    load_balance = load_balance  .replace({"scenario": short_names})
    load_balance .columns= load_balance .columns.str.lower()
    load_balance['withdrawal']=-(load_balance['zone_demand_mw'] + load_balance['storagenetcharge'])
    load_balance.rename(columns={'zonetotalcentraldispatch': 'injection'}, inplace=True)
    load_balance.drop(columns=['normalized_energy_balance_duals_dollar_per_mwh', 'txpowernet', 'zone_demand_mw', 'storagenetcharge'], inplace=True)
    zone_power_balance=load_balance.copy()
    zone_power_balance=zone_power_balance.loc[zone_power_balance['load_zone'].isin(zones)]

    time_info = timestamp_info(scenarios_file_name, analysis_scenario)[1]
    zone_power_balance = pd.merge(left = zone_power_balance, right = time_info[['scenario', 'timestamp', 'ts_period']], on = ['scenario', 'timestamp'])

    zone_power_balance=zone_power_balance.loc[zone_power_balance['ts_period'].isin(analysis_period)]

    zone_power_balance.loc[:,"timestamp"]=zone_power_balance.apply(lambda x: datetime.strptime(str(x['timestamp']), '%Y%m%d%H'),axis=1)
    zone_power_balance.loc[:,"timestamp"]=zone_power_balance.apply(lambda x: x['timestamp'].tz_localize('utc'),axis=1)
    zone_power_balance.loc[:,"timestamp"]=zone_power_balance.apply(lambda x: x['timestamp'].tz_convert('US/Mountain'),axis=1)
    zone_power_balance.loc[:,'month']=zone_power_balance.apply(lambda x: x['timestamp'].month, axis=1)

    zone_power_balance_by_sc=zone_power_balance.pivot_table(index=["scenario", "month"], values=['injection', 'withdrawal'], aggfunc=np.sum)

    zone_power_balance_by_sc['ratio']=zone_power_balance_by_sc['injection']/zone_power_balance_by_sc['withdrawal']

    zone_power_balance_by_sc=zone_power_balance_by_sc.pivot_table(index="month", columns="scenario", values='ratio', aggfunc=np.sum)

    order_cols=[]

    for n in range(0,len(scenario)): #count the number of keys of order dictionary
        order_cols.append(get_key(order,n))

    zone_power_balance_by_sc =  zone_power_balance_by_sc[order_cols]

    return zone_power_balance_by_sc

def annual_exchange_ratio_by_zone(scenarios_file_name: str,
                                  analysis_scenario: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    load_balance = get_data(scenario, "load_balance.csv")
    load_balance  = load_balance  .replace({"scenario": short_names})
    load_balance .columns= load_balance .columns.str.lower()
    load_balance['withdrawal']=-(load_balance['zone_demand_mw'] + load_balance['storagenetcharge'])
    load_balance.rename(columns={'zonetotalcentraldispatch': 'injection'}, inplace=True)
    load_balance.drop(columns=['normalized_energy_balance_duals_dollar_per_mwh', 'txpowernet', 'zone_demand_mw', 'storagenetcharge'], inplace=True)
    zone_power_balance=load_balance.copy()

    annual_zone_power_balance_by_scenario = zone_power_balance.pivot_table(
    index=["scenario", 'load_zone'], values=["injection", "withdrawal"], aggfunc=np.sum )
    annual_zone_power_balance_by_scenario['ratio']=annual_zone_power_balance_by_scenario['injection']/annual_zone_power_balance_by_scenario['withdrawal']
    annual_zone_power_balance_by_scenario.reset_index(inplace=True)

    annual_zone_power_balance_by_scenario['sc_order'] = annual_zone_power_balance_by_scenario['scenario'].map(order)
    annual_zone_power_balance_by_scenario = annual_zone_power_balance_by_scenario .sort_values('sc_order').drop('sc_order',axis=1)
    
    return annual_zone_power_balance_by_scenario

def avg_exchange_ratio_by_zone(scenarios_file_name: str,
                               analysis_scenario: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    load_balance = get_data(scenario, "load_balance.csv")
    load_balance  = load_balance  .replace({"scenario": short_names})
    load_balance .columns= load_balance .columns.str.lower()
    load_balance['withdrawal']=-(load_balance['zone_demand_mw'] + load_balance['storagenetcharge'])
    load_balance.rename(columns={'zonetotalcentraldispatch': 'injection'}, inplace=True)
    load_balance.drop(columns=['normalized_energy_balance_duals_dollar_per_mwh', 'txpowernet', 'zone_demand_mw', 'storagenetcharge'], inplace=True)
    zone_power_balance=load_balance.copy()
    zone_power_balance['ratio'] = zone_power_balance['injection']/zone_power_balance['withdrawal']

    annual_zone_power_balance_by_scenario = zone_power_balance.pivot_table(
    index=["scenario", 'load_zone'], values=["ratio"], aggfunc=np.average )
    annual_zone_power_balance_by_scenario.reset_index(inplace=True)

    annual_zone_power_balance_by_scenario['sc_order'] = annual_zone_power_balance_by_scenario['scenario'].map(order)
    annual_zone_power_balance_by_scenario = annual_zone_power_balance_by_scenario .sort_values('sc_order').drop('sc_order',axis=1)
    
    return annual_zone_power_balance_by_scenario

def annual_exchange_ratio_zone_by_period(scenarios_file_name: str, 
                                         analysis_scenario: list,
                                         zones: list):
    
    # Funtion that returns the annual exchange ratio by scenario and by period.
    # Output:
    #
    #   ts_period |	low_5	low_10	low_25	low_50	low_75	low_100	    mid_5	mid_10	mid_25
    #   --------------------------------------------------------------------------------------
    #   2050	    0.96	0.96	0.94	0.90	0.88	0.86	    0.93	0.93	0.92	
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    load_balance = get_data(scenario, "load_balance.csv")
    load_balance  = load_balance  .replace({"scenario": short_names})
    load_balance .columns= load_balance .columns.str.lower()
    load_balance['withdrawal']=-(load_balance['zone_demand_mw'] + load_balance['storagenetcharge'])
    load_balance.rename(columns={'zonetotalcentraldispatch': 'injection'}, inplace=True)
    load_balance.drop(columns=['normalized_energy_balance_duals_dollar_per_mwh', 'txpowernet', 'zone_demand_mw', 'storagenetcharge'], inplace=True)
    zone_power_balance=load_balance.copy()
    zone_power_balance=zone_power_balance.loc[zone_power_balance['load_zone'].isin(zones)]

    time_info = timestamp_info(scenarios_file_name, analysis_scenario)[1]
    zone_power_balance = pd.merge(left = zone_power_balance, right = time_info[['scenario', 'timestamp', 'ts_period']], on = ['scenario', 'timestamp'])
    
    annual_zone_power_balance_by_scenario = zone_power_balance.pivot_table(
    index=["scenario", 'ts_period'], values=["injection", "withdrawal"], aggfunc=np.sum )
    annual_zone_power_balance_by_scenario['ratio']=annual_zone_power_balance_by_scenario['injection']/annual_zone_power_balance_by_scenario['withdrawal']
    
    annual_zone_power_balance_by_scenario=annual_zone_power_balance_by_scenario.pivot_table(index="ts_period", columns="scenario", values='ratio',aggfunc=np.sum)

    order_cols=[]

    for n in range(0,len(scenario)): #count the number of keys of order dictionary
        order_cols.append(get_key(order,n))

    annual_zone_power_balance_by_scenario=annual_zone_power_balance_by_scenario[order_cols]

    return annual_zone_power_balance_by_scenario   

# ----------------------------------------------                    ------------------------------------------------------------------------------------------------
# ---------------------------------------------- BUILT TRANSMISSION ------------------------------------------------------------------------------------------------
# ----------------------------------------------                    ------------------------------------------------------------------------------------------------

def intertie_expansion(scenarios_file_name: str, 
                       analysis_scenario: list, 
                       only_one_period: list, 
                       baseline_scenario: str, 
                       zones: list):

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario]
    
    transmission = get_data(scenario, "transmission.csv", usecols=['PERIOD', 'trans_lz1', 'trans_lz2', 'TxCapacityNameplate'])
    transmission = transmission  .replace({"scenario": short_names})
    transmission = transmission.loc[transmission.trans_lz1.isin(zones) ^ transmission.trans_lz2.isin(zones)]
    transmission = transmission.loc[transmission.PERIOD.isin(only_one_period)]
    total_transmission = transmission.pivot_table(index=['scenario'], values=['TxCapacityNameplate'], aggfunc=np.sum)
    total_transmission.reset_index(inplace = True)
    total_transmission['base']=total_transmission.loc[total_transmission.scenario== baseline_scenario,'TxCapacityNameplate'].iloc[0]
    total_transmission['relative']=total_transmission['TxCapacityNameplate']/total_transmission['base']
    total_transmission['sc_order'] = total_transmission['scenario'].map(order)
    total_transmission = total_transmission.sort_values('sc_order').drop('sc_order',axis=1)
    
    return total_transmission

def intertie_build_transmission(scenarios_file_name: str, 
                                analysis_scenario: list, 
                                only_one_period: list, 
                                baseline_scenario: str, 
                                zones: list):

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario]
    
    transmission = get_data(scenario, "transmission.csv", usecols=['PERIOD', 'trans_lz1', 'trans_lz2', 'BuildTx'])
    transmission['BuildTx'] = transmission['BuildTx'] .replace('.', 0)
    transmission['BuildTx'] = pd.to_numeric(transmission['BuildTx'])
    transmission = transmission  .replace({"scenario": short_names})
    transmission = transmission.loc[transmission.trans_lz1.isin(zones) ^ transmission.trans_lz2.isin(zones)]
    transmission = transmission.loc[transmission.PERIOD.isin(only_one_period)]
    total_transmission = transmission.pivot_table(index=['scenario'], values=['BuildTx'], aggfunc=np.sum)
    total_transmission.reset_index(inplace = True)
    total_transmission['base']=total_transmission.loc[total_transmission.scenario== baseline_scenario,'BuildTx'].iloc[0]
    total_transmission['relative']=total_transmission['BuildTx']/total_transmission['base']
    total_transmission['sc_order'] = total_transmission['scenario'].map(order)
    total_transmission = total_transmission.sort_values('sc_order').drop('sc_order',axis=1)
    
    return total_transmission

def in_zone_transmission_expansion(scenarios_file_name: str, 
                                   analysis_scenario: list,
                                   only_one_period: list, 
                                   baseline_scenario: str, 
                                   zones: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 
    
    transmission = get_data(scenario, "transmission.csv", usecols=['PERIOD', 'trans_lz1', 'trans_lz2', 'TxCapacityNameplate'])
    transmission = transmission  .replace({"scenario": short_names})
    if len(zones)!= 0:
        transmission = transmission.loc[transmission.trans_lz1.isin(zones) & transmission.trans_lz2.isin(zones)]
    transmission = transmission.loc[transmission.PERIOD.isin(only_one_period)]
    total_transmission = transmission.pivot_table(index=['scenario'], values=['TxCapacityNameplate'], aggfunc=np.sum)
    total_transmission.reset_index(inplace = True)
    total_transmission['base']=total_transmission.loc[total_transmission.scenario== baseline_scenario,'TxCapacityNameplate'].iloc[0]
    total_transmission['relative']=total_transmission['TxCapacityNameplate']/total_transmission['base']
    total_transmission['sc_order'] = total_transmission['scenario'].map(order)
    total_transmission = total_transmission.sort_values('sc_order').drop('sc_order',axis=1)
    
    return total_transmission 

def in_zone_build_transmission(scenarios_file_name: str, 
                                   analysis_scenario: list,
                                   only_one_period: list, 
                                   baseline_scenario: str, 
                                   zones: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 
    
    transmission = get_data(scenario, "transmission.csv", usecols=['PERIOD', 'trans_lz1', 'trans_lz2', 'BuildTx'])
    transmission = transmission.replace({'BuildTx'})
    transmission['BuildTx'] = transmission['BuildTx'] .replace('.', 0)
    transmission['BuildTx'] = pd.to_numeric(transmission['BuildTx'])
    if len(zones)!= 0:
        transmission = transmission.loc[transmission.trans_lz1.isin(zones) & transmission.trans_lz2.isin(zones)]
    transmission = transmission.loc[transmission.PERIOD.isin(only_one_period)]
    total_transmission = transmission.pivot_table(index=['scenario'], values=['BuildTx'], aggfunc=np.sum)
    total_transmission.reset_index(inplace = True)
    total_transmission['base']=total_transmission.loc[total_transmission.scenario== baseline_scenario,'BuildTx'].iloc[0]
    total_transmission['relative']=total_transmission['BuildTx']/total_transmission['base']
    total_transmission['sc_order'] = total_transmission['scenario'].map(order)
    total_transmission = total_transmission.sort_values('sc_order').drop('sc_order',axis=1)
    
    return total_transmission  

def in_zone_build_transmission(scenarios_file_name: str, 
                                   analysis_scenario: list,
                                   only_one_period: list, 
                                   baseline_scenario: str, 
                                   zones: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 
    
    transmission = get_data(scenario, "transmission.csv", usecols=['PERIOD', 'trans_lz1', 'trans_lz2', 'BuildTx'])
    transmission['BuildTx'] = transmission['BuildTx'] .replace('.', 0)
    transmission['BuildTx'] = pd.to_numeric(transmission['BuildTx'])
    if len(zones)!= 0:
        transmission = transmission.loc[transmission.trans_lz1.isin(zones) & transmission.trans_lz2.isin(zones)]
    transmission = transmission.loc[transmission.PERIOD.isin(only_one_period)]
    total_transmission = transmission.pivot_table(index=['scenario'], values=['BuildTx'], aggfunc=np.sum)
    total_transmission.reset_index(inplace = True)
    total_transmission['base']=total_transmission.loc[total_transmission.scenario== baseline_scenario,'BuildTx'].iloc[0]
    total_transmission['relative']=total_transmission['BuildTx']/total_transmission['base']
    total_transmission['sc_order'] = total_transmission['scenario'].map(order)
    total_transmission = total_transmission.sort_values('sc_order').drop('sc_order',axis=1)
    
    return total_transmission 

def in_zone_weighted_average_transmission_expansion(scenarios_file_name: str, 
                                                    analysis_scenario: list, 
                                                    only_one_period: list, 
                                                    baseline_scenario: str, 
                                                    zones: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    transmission = get_data(scenario, "transmission.csv", usecols=['PERIOD', 'trans_lz1', 'trans_lz2', 'trans_length_km', 'TxCapacityNameplate'])
    transmission = transmission  .replace({"scenario": short_names})
    if len(zones)!= 0:
        transmission = transmission.loc[transmission.trans_lz1.isin(zones) & transmission.trans_lz2.isin(zones)]
    transmission = transmission.loc[transmission.PERIOD.isin(only_one_period)]
    transmission['tx_capacity_x_length'] = transmission['trans_length_km'] * transmission['TxCapacityNameplate']
    total_transmission = transmission.pivot_table(index=['scenario'], values=['tx_capacity_x_length', 'trans_length_km'], aggfunc=np.sum)
    total_transmission.reset_index(inplace = True)
    total_transmission['weighted_average_tx_capacity'] = total_transmission['tx_capacity_x_length']/total_transmission['trans_length_km']
    total_transmission['base']=total_transmission.loc[total_transmission.scenario == baseline_scenario,'weighted_average_tx_capacity'].iloc[0]
    total_transmission['relative']=total_transmission['weighted_average_tx_capacity']/total_transmission['base']
    total_transmission['sc_order'] = total_transmission['scenario'].map(order)
    total_transmission = total_transmission.sort_values('sc_order').drop('sc_order',axis=1)
    
    return total_transmission 

def total_tx_capacity(scenarios_file_name: str, 
                      analysis_scenario: list,
                      analysis_period: list, 
                      analysis_zones: list, 
                      interval_length_existing_cap: float, 
                      interval_length_new_cap: float, 
                      geodf_wecc: gpd.GeoDataFrame):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    transmission_cap = get_data(scenario, "transmission.csv", usecols=['PERIOD', 'trans_lz1', 'trans_lz2', 'existing_trans_cap', 'BuildTx'])
    transmission_cap.replace({"scenario": short_names}, inplace=True)

    if len(analysis_period)==0:
        analysis_period = transmission_cap.PERIOD.unique()
    if len(analysis_zones)==0:
        analysis_zones = set(list(transmission_cap.trans_lz1) + list(transmission_cap.trans_lz2))

    transmission_cap = transmission_cap.loc[((transmission_cap.trans_lz1.isin(analysis_zones)) | (transmission_cap.trans_lz2.isin(analysis_zones)))]
    transmission_cap = transmission_cap.loc[transmission_cap.PERIOD.isin(analysis_period)]
    
    transmission_cap['existing_trans_cap']= transmission_cap['existing_trans_cap']/1000 #Tranform capacity from MW to GW
    
    transmission_cap["BuildTx"].replace({'.': 0}, inplace=True)
    transmission_cap["BuildTx"] = transmission_cap["BuildTx"].astype(float)
    transmission_cap['BuildTx'] = transmission_cap['BuildTx']/1000 ##Tranform capacity from MW to GW
    
    if len(geodf_wecc)>0:
        transmission_cap = pd.merge(left=transmission_cap, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'trans_lz1'}), on='trans_lz1')
        transmission_cap.rename(columns={'centroid_partial_can': 'coordinate_trans_lz1'}, inplace=True)
        transmission_cap = pd.merge(left=transmission_cap, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'trans_lz2'}), on='trans_lz2')
        transmission_cap.rename(columns={'centroid_partial_can': 'coordinate_trans_lz2'}, inplace=True)

    #Generate a dataframe with intervals for the existing capacity and create column with linewidth (useful for maps)
    if interval_length_existing_cap>0:
        nintervals = int(math.floor(max(transmission_cap['existing_trans_cap'])/interval_length_existing_cap))

        existing_intervals = []
        for n in range(0, nintervals+1):
            min_interval = interval_length_existing_cap * n
            max_interval = interval_length_existing_cap * (n+1)
            linewidth = (n+1)
            existing_intervals.append([min_interval, max_interval, linewidth])

        existing_intervals = pd.DataFrame(existing_intervals, columns=['min_interval','max_interval', "linewidth"])

        transmission_cap.loc[:, 'existing_linewidth'] = transmission_cap.apply(lambda x: return_interval(x['existing_trans_cap'], existing_intervals), axis=1).linewidth

    #Generate a dataframe with intervals for the new capacity and create column with linewidth (useful for maps)
    if interval_length_new_cap>0:
        nintervals = int(math.floor(max(transmission_cap['BuildTx'])/interval_length_new_cap))

        new_cap_intervals = []
        for n in range(0, nintervals+1):
            min_interval = interval_length_new_cap * n
            max_interval = interval_length_new_cap * (n+1)
            linewidth = (n+1)
            new_cap_intervals.append([min_interval, max_interval, linewidth])

        new_cap_intervals = pd.DataFrame(new_cap_intervals, columns=['min_interval','max_interval', "linewidth"])

        transmission_cap.loc[:, 'new_cap_linewidth'] = transmission_cap.apply(lambda x: return_interval(x['BuildTx'], new_cap_intervals), axis=1).linewidth

    return transmission_cap,  existing_intervals, new_cap_intervals

def total_tx_capacity_special_zones(scenarios_file_name: str, 
                                    analysis_scenario: list,
                                    transmission_cap: pd.DataFrame, 
                                    analysis_period: list, 
                                    analysis_zones: list, 
                                    interval_length_existing_cap: float, 
                                    interval_length_new_cap: float, 
                                    geodf_wecc: gpd.GeoDataFrame):

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    transmission_cap.replace({"scenario": short_names}, inplace=True)

    if len(analysis_period)==0:
        analysis_period = transmission_cap.PERIOD.unique()
    if len(analysis_zones)==0:
        analysis_zones = set(list(transmission_cap.trans_lz1) + list(transmission_cap.trans_lz2))

    transmission_cap = transmission_cap.loc[((transmission_cap.trans_lz1.isin(analysis_zones)) | (transmission_cap.trans_lz2.isin(analysis_zones)))]
    transmission_cap = transmission_cap.loc[transmission_cap.PERIOD.isin(analysis_period)]
    
    transmission_cap['existing_trans_cap']= transmission_cap['existing_trans_cap']/1000 #Tranform capacity from MW to GW
    
    transmission_cap["BuildTx"].replace({'.': 0}, inplace=True)
    transmission_cap["BuildTx"] = transmission_cap["BuildTx"].astype(float)
    transmission_cap['BuildTx'] = transmission_cap['BuildTx']/1000 ##Tranform capacity from MW to GW
    
    if len(geodf_wecc)>0:
        transmission_cap = pd.merge(left=transmission_cap, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'trans_lz1'}), on='trans_lz1')
        transmission_cap.rename(columns={'centroid_partial_can': 'coordinate_trans_lz1'}, inplace=True)
        transmission_cap = pd.merge(left=transmission_cap, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'trans_lz2'}), on='trans_lz2')
        transmission_cap.rename(columns={'centroid_partial_can': 'coordinate_trans_lz2'}, inplace=True)

    #Generate a dataframe with intervals for the existing capacity and create column with linewidth (useful for maps)
    if interval_length_existing_cap>0:
        nintervals = int(math.floor(max(transmission_cap['existing_trans_cap'])/interval_length_existing_cap))

        existing_intervals = []
        for n in range(0, nintervals+1):
            min_interval = interval_length_existing_cap * n
            max_interval = interval_length_existing_cap * (n+1)
            linewidth = (n+1)
            existing_intervals.append([min_interval, max_interval, linewidth])

        existing_intervals = pd.DataFrame(existing_intervals, columns=['min_interval','max_interval', "linewidth"])

        transmission_cap.loc[:, 'existing_linewidth'] = transmission_cap.apply(lambda x: return_interval(x['existing_trans_cap'], existing_intervals), axis=1).linewidth

    #Generate a dataframe with intervals for the new capacity and create column with linewidth (useful for maps)
    if interval_length_new_cap>0:
        nintervals = int(math.floor(max(transmission_cap['BuildTx'])/interval_length_new_cap))

        new_cap_intervals = []
        for n in range(0, nintervals+1):
            min_interval = interval_length_new_cap * n
            max_interval = interval_length_new_cap * (n+1)
            linewidth = (n+1)
            new_cap_intervals.append([min_interval, max_interval, linewidth])

        new_cap_intervals = pd.DataFrame(new_cap_intervals, columns=['min_interval','max_interval', "linewidth"])

        transmission_cap.loc[:, 'new_cap_linewidth'] = transmission_cap.apply(lambda x: return_interval(x['BuildTx'], new_cap_intervals), axis=1).linewidth

    return transmission_cap,  existing_intervals, new_cap_intervals

def generate_new_transmission(scenario_file_name: str, special_zones: pd.DataFrame):

    scenario, short_names, order = read_scenarios(scenario_file_name)
    tx = get_data(scenario, "transmission.csv", usecols=['PERIOD', 'trans_lz1', 'trans_lz2', 'existing_trans_cap', 'BuildTx'])
    tx .replace(short_names, inplace=True)
    new_tx = pd.merge(left = tx, right = special_zones.rename(columns={'load_zone':'trans_lz1'}), on='trans_lz1')
    new_tx = new_tx.merge(special_zones.rename(columns={'load_zone':'trans_lz2'}), on='trans_lz2')
    new_tx.drop(['trans_lz1', 'trans_lz2'], axis=1, inplace=True)
    new_tx.rename(columns={'state_x': 'trans_lz1', 'state_y': 'trans_lz2'}, inplace=True)

    new_tx = new_tx[new_tx['trans_lz1'] != new_tx['trans_lz2']]

    new_tx["BuildTx"].replace({'.': 0}, inplace=True)
    new_tx["BuildTx"] = new_tx["BuildTx"].astype(float)

    new_tx = new_tx.pivot_table(index=['PERIOD', 'trans_lz1', 'trans_lz2', 'scenario'], values=['existing_trans_cap', 'BuildTx'], aggfunc=np.sum)
    new_tx.reset_index(inplace=True)

    new_tx['sc_order'] = new_tx['scenario'].map(order)
    new_tx= new_tx.sort_values('sc_order').drop('sc_order',axis=1)

    return new_tx

def in_zone_built_transmission(scenarios_file_name: str, 
                               analysis_scenario: list,
                               only_one_period: list, 
                               baseline_scenario, 
                               zones: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 
    
    transmission = get_data(scenario, "transmission.csv", usecols=['PERIOD', 'trans_lz1', 'trans_lz2', 'BuildTx'])
    transmission = transmission  .replace({"scenario": short_names})
    transmission["BuildTx"].replace({'.': 0}, inplace=True)
    transmission["BuildTx"] = transmission["BuildTx"].astype(float)
    if len(zones)!= 0:
        transmission = transmission.loc[transmission.trans_lz1.isin(zones) & transmission.trans_lz2.isin(zones)]
    transmission = transmission.loc[transmission.PERIOD.isin(only_one_period)]
    total_transmission = transmission.pivot_table(index='scenario', values='BuildTx', aggfunc=np.sum) 
    total_transmission.reset_index(inplace = True)
    total_transmission['base']=total_transmission.loc[total_transmission.scenario==baseline_scenario,'BuildTx'].iloc[0]
    total_transmission['relative']=total_transmission['BuildTx']/total_transmission['base']
    total_transmission['sc_order'] = total_transmission['scenario'].map(order)
    total_transmission = total_transmission.sort_values('sc_order').drop('sc_order',axis=1)
    total_transmission.reset_index(inplace=True, drop=True)
    
    return total_transmission 


# ----------------------------------------------                     ---------------------------------------------------------------------------------------------
# ---------------------------------------------- USE OF TRANSMISSION ---------------------------------------------------------------------------------------------
# ----------------------------------------------                     ---------------------------------------------------------------------------------------------

def average_tx_loadability_for_timestamps(scenarios_file_name: str, 
                                          analysis_scenario: list,
                                          analysis_zones: list, 
                                          df_timestamps: pd.DataFrame, 
                                          geodf_wecc: gpd.GeoDataFrame, 
                                          df_intervals: pd.DataFrame):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    flows = get_data(scenario, "transmission_dispatch.csv")
    flows.replace(short_names, inplace=True)
    
    if len(analysis_zones)!=0:
        analysis_zones = flows.loc[((flows.load_zone_from.isin(analysis_zones)) & (flows.load_zone_to.isin(analysis_zones)))]

    flows["loadability"] =  flows["transmission_dispatch"]/flows["dispatch_limit"]*100
    

    if len(df_timestamps)!=0:
        flows = pd.merge(left = flows, right = df_timestamps[['scenario', 'timestamp']], on=['scenario', 'timestamp'])
        #flows = flows.loc[flows.timestamp.isin(list(df_timestamps.timestamp))]

    #Get the average of loadability over the selected timestamps
    flows_loadability = flows.pivot_table(index=['load_zone_from', 'load_zone_to', 'scenario'], values='loadability', aggfunc=np.average)
    flows_loadability.reset_index(inplace=True)

    if len(geodf_wecc)>0:
        #Attach the coordinates to the "load_zone_from", and "load_zone_to". This help for the creating of lines in maps
        flows_loadability = pd.merge(left=flows_loadability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_from'}), on='load_zone_from')
        flows_loadability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_from'}, inplace=True)
        flows_loadability = pd.merge(left=flows_loadability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_to'}), on='load_zone_to')
        flows_loadability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_to'}, inplace=True)

    if len(df_intervals)>0:
        #Give linewidth and color to each row according to the loadability. It needs the function return_interval
        flows_loadability.loc[:, ['linewidth', 'color']] = flows_loadability.apply(lambda x: return_interval(x['loadability'], df_intervals), axis=1)[['linewidth','color']]

    #In case if it is necessary, provide default direction.
    default_direction = get_data(scenario, "transmission_lines.csv", fpath='inputs', usecols=['trans_lz1', 'trans_lz2'])
    default_direction.rename(columns={'trans_lz1':'load_zone_from', 'trans_lz2':'load_zone_to'}, inplace=True)
    default_direction.replace(short_names, inplace=True)
    default_direction['direction'] = 1
    flows_loadability = pd.merge(left=flows_loadability, right=default_direction, on=['load_zone_from', 'load_zone_to', 'scenario'], how='left')

    #Give order (though it is not necessary)
    flows_loadability['sc_order'] = flows_loadability['scenario'].map(order)
    flows_loadability = flows_loadability.sort_values('sc_order').drop('sc_order',axis=1)

    return flows_loadability

def average_net_tx_loadability_for_timestamps(scenarios_file_name: str, 
                                              analysis_scenario: list,
                                              analysis_zones: list, 
                                              df_timestamps: pd.DataFrame, 
                                              geodf_wecc: gpd.GeoDataFrame, 
                                              df_intervals: pd.DataFrame):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)

    if len(analysis_scenario) >0:
        scenario = [inv_short_names[i] for i in analysis_scenario] 

    flows = get_data(scenario, "transmission_dispatch.csv")
    flows.replace(short_names, inplace=True)

    default_direction = get_data(scenario, "transmission_lines.csv", fpath='inputs', usecols=['trans_lz1', 'trans_lz2'])
    default_direction.rename(columns={'trans_lz1':'load_zone_from', 'trans_lz2':'load_zone_to'}, inplace=True)
    default_direction.replace(short_names, inplace=True)
    
    if len(analysis_zones)!=0:
        analysis_zones = flows.loc[((flows.load_zone_from.isin(analysis_zones)) & (flows.load_zone_to.isin(analysis_zones)))]

    if len(df_timestamps)!=0:
        flows = pd.merge(left = flows, right = df_timestamps[['scenario', 'timestamp']], on=['scenario', 'timestamp'])

    lines = default_direction.copy()
    lines = lines [['load_zone_from', 'load_zone_to']]
    
    for j in range(0,len(default_direction)):
        one_pair = lines.iloc[[j]]
        reverse_direction = one_pair.copy()
        reverse_direction.rename(columns={'load_zone_from' : 'load_zone_to', 'load_zone_to': 'load_zone_from'}, inplace=True)
    
        aux = pd.merge(left = flows , right = pd.concat([one_pair, reverse_direction]), on = ['load_zone_from', 'load_zone_to'])
    
        greatest_flow=max(aux.transmission_dispatch)
        smallest_flow=min(aux.transmission_dispatch)
        net = greatest_flow - smallest_flow
    
        if smallest_flow > 0:
          if greatest_flow==smallest_flow:
            flows.loc[(flows.load_zone_from.isin(one_pair.load_zone_from)) & (flows.load_zone_to.isin(one_pair.load_zone_to)), "transmission_dispatch"] = net
            flows.loc[(flows.load_zone_from.isin(reverse_direction.load_zone_from)) & (flows.load_zone_to.isin(reverse_direction.load_zone_to)), "transmission_dispatch"] = net
          else: 
            line_with_greatest = aux.loc[aux.transmission_dispatch == greatest_flow]
            line_with_smallest = aux.loc[aux.transmission_dispatch == smallest_flow]
            flows.loc[(flows.load_zone_from.isin(line_with_greatest.load_zone_from)) & (flows.load_zone_to.isin(line_with_greatest.load_zone_to)), "transmission_dispatch"] = net
            flows.loc[(flows.load_zone_from.isin(line_with_smallest.load_zone_from)) & (flows.load_zone_to.isin(line_with_smallest.load_zone_to)), "transmission_dispatch"] = 0.0


    flows["loadability"] =  flows["transmission_dispatch"]/flows["dispatch_limit"]*100
        #flows = flows.loc[flows.timestamp.isin(list(df_timestamps.timestamp))]  

    #Get the average of loadability over the selected timestamps
    flows_loadability = flows.pivot_table(index=['load_zone_from', 'load_zone_to', 'scenario'], values='loadability', aggfunc=np.average)
    flows_loadability.reset_index(inplace=True)

    if len(geodf_wecc)>0:
        #Attach the coordinates to the "load_zone_from", and "load_zone_to". This help for the creating of lines in maps
        flows_loadability = pd.merge(left=flows_loadability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_from'}), on='load_zone_from')
        flows_loadability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_from'}, inplace=True)
        flows_loadability = pd.merge(left=flows_loadability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_to'}), on='load_zone_to')
        flows_loadability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_to'}, inplace=True)

    if len(df_intervals)>0:
        #Give linewidth and color to each row according to the loadability. It needs the function return_interval
        flows_loadability.loc[:, ['linewidth', 'color']] = flows_loadability.apply(lambda x: return_interval(x['loadability'], df_intervals), axis=1)[['linewidth','color']]

    #In case if it is necessary, provide default direction.
    default_direction['direction'] = 1
    flows_loadability = pd.merge(left=flows_loadability, right=default_direction, on=['load_zone_from', 'load_zone_to', 'scenario'], how='left')

    #Give order (though it is not necessary)
    flows_loadability['sc_order'] = flows_loadability['scenario'].map(order)
    flows_loadability = flows_loadability.sort_values('sc_order').drop('sc_order',axis=1)

    return flows_loadability

def annual_tx_loadability_special_lz(scenarios_file_name: str, 
                                     analysis_scenarios: list,
                                     special_txs: pd.DataFrame, 
                                     special_txdis: pd.DataFrame,  
                                     analysis_zones: list, 
                                     geodf_wecc: gpd.GeoDataFrame, 
                                     df_intervals: pd.DataFrame):
    
    scenario, short_names, order = read_scenarios(scenarios_file_name)
    flows =  special_txdis

    if len(analysis_scenarios)!=0:
        flows = flows.loc[flows.scenario.isin(analysis_scenarios)]
    
    if len(analysis_zones)!=0:
        analysis_zones = flows.loc[((flows.load_zone_from.isin(analysis_zones)) & (flows.load_zone_to.isin(analysis_zones)))]

    flows["cargability"] =  flows["transmission_dispatch"]/flows["dispatch_limit"]*100
    
    #Get the average of cargability over the timestamp of the total number of periods (usually one year). You can change "average"
    flows_cargability = flows.pivot_table(index=['load_zone_from', 'load_zone_to', 'scenario'], values='cargability', aggfunc=np.average)
    flows_cargability.reset_index(inplace=True)

    if len(geodf_wecc)>0:
        #Attach the coordinates to the "load_zone_from", and "load_zone_to". This help for the creating of lines in maps
        flows_cargability = pd.merge(left=flows_cargability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_from'}), on='load_zone_from')
        flows_cargability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_from'}, inplace=True)
        flows_cargability = pd.merge(left=flows_cargability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_to'}), on='load_zone_to')
        flows_cargability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_to'}, inplace=True)

    if len(df_intervals)>0:
        #Give linewidth and color to each row according to the cargability. It needs the function return_interval
        flows_cargability.loc[:, ['linewidth', 'color']] = flows_cargability.apply(lambda x: return_interval(x['cargability'], df_intervals), axis=1)[['linewidth','color']]

    #In case if it is necessary, provide default direction.
    default_direction = special_txs[['trans_lz1', 'trans_lz2', 'scenario']]

    if len(analysis_scenarios)!=0:
        default_direction = default_direction.loc[default_direction.scenario.isin(analysis_scenarios)]

    default_direction.rename(columns={'trans_lz1':'load_zone_from', 'trans_lz2':'load_zone_to'}, inplace=True)
    default_direction['direction'] = 1
    flows_cargability = pd.merge(left=flows_cargability, right=default_direction, on=['load_zone_from', 'load_zone_to', 'scenario'], how='left')

    #Give order (though it is not necessary)
    flows_cargability['sc_order'] = flows_cargability['scenario'].map(order)
    flows_cargability = flows_cargability.sort_values('sc_order').drop('sc_order',axis=1)

    return flows_cargability

def average_net_tx_loadability_for_timestamps_special_zones(scenario_file_name: str, 
                                                            analysis_scenarios: list,
                                                            special_txs: pd.DataFrame, 
                                                            special_txdis: pd.DataFrame, 
                                                            analysis_zones: list, 
                                                            df_timestamps: pd.DataFrame, 
                                                            geodf_wecc: gpd.GeoDataFrame, 
                                                            df_intervals: pd.DataFrame):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenario_file_name)

    if len(analysis_scenarios)!=0:
        flows = flows.loc[flows.scenario.isin(analysis_scenarios)]
    
    flows = special_txdis.copy()
    flows.replace(short_names, inplace=True)

    default_direction = special_txs.copy()
    default_direction.rename(columns={'trans_lz1':'load_zone_from', 'trans_lz2':'load_zone_to'}, inplace=True)
    default_direction.replace(short_names, inplace=True)
    
    if len(analysis_zones)!=0:
        analysis_zones = flows.loc[((flows.load_zone_from.isin(analysis_zones)) & (flows.load_zone_to.isin(analysis_zones)))]

    if len(df_timestamps)!=0:
        flows = pd.merge(left = flows, right = df_timestamps[['scenario', 'timestamp']], on=['scenario', 'timestamp'])

    lines = default_direction.copy()
    lines = lines [['load_zone_from', 'load_zone_to']]
    
    for j in range(0,len(default_direction)):
        one_pair = lines.iloc[[j]]
        reverse_direction = one_pair.copy()
        reverse_direction.rename(columns={'load_zone_from' : 'load_zone_to', 'load_zone_to': 'load_zone_from'}, inplace=True)
    
        aux = pd.merge(left = flows , right = pd.concat([one_pair, reverse_direction]), on = ['load_zone_from', 'load_zone_to'])
    
        greatest_flow=max(aux.transmission_dispatch)
        smallest_flow=min(aux.transmission_dispatch)
        net = greatest_flow - smallest_flow
    
        if smallest_flow > 0:
          if greatest_flow==smallest_flow:
            flows.loc[(flows.load_zone_from.isin(one_pair.load_zone_from)) & (flows.load_zone_to.isin(one_pair.load_zone_to)), "transmission_dispatch"] = net
            flows.loc[(flows.load_zone_from.isin(reverse_direction.load_zone_from)) & (flows.load_zone_to.isin(reverse_direction.load_zone_to)), "transmission_dispatch"] = net
          else: 
            line_with_greatest = aux.loc[aux.transmission_dispatch == greatest_flow]
            line_with_smallest = aux.loc[aux.transmission_dispatch == smallest_flow]
            flows.loc[(flows.load_zone_from.isin(line_with_greatest.load_zone_from)) & (flows.load_zone_to.isin(line_with_greatest.load_zone_to)), "transmission_dispatch"] = net
            flows.loc[(flows.load_zone_from.isin(line_with_smallest.load_zone_from)) & (flows.load_zone_to.isin(line_with_smallest.load_zone_to)), "transmission_dispatch"] = 0.0


    flows["loadability"] =  flows["transmission_dispatch"]/flows["dispatch_limit"]*100
        #flows = flows.loc[flows.timestamp.isin(list(df_timestamps.timestamp))]  

    #Get the average of loadability over the selected timestamps
    flows_loadability = flows.pivot_table(index=['load_zone_from', 'load_zone_to', 'scenario'], values='loadability', aggfunc=np.average)
    flows_loadability.reset_index(inplace=True)

    if len(geodf_wecc)>0:
        #Attach the coordinates to the "load_zone_from", and "load_zone_to". This help for the creating of lines in maps
        flows_loadability = pd.merge(left=flows_loadability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_from'}), on='load_zone_from')
        flows_loadability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_from'}, inplace=True)
        flows_loadability = pd.merge(left=flows_loadability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_to'}), on='load_zone_to')
        flows_loadability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_to'}, inplace=True)

    if len(df_intervals)>0:
        #Give linewidth and color to each row according to the loadability. It needs the function return_interval
        flows_loadability.loc[:, ['linewidth', 'color']] = flows_loadability.apply(lambda x: return_interval(x['loadability'], df_intervals), axis=1)[['linewidth','color']]

    #In case if it is necessary, provide default direction.
    default_direction['direction'] = 1
    flows_loadability = pd.merge(left=flows_loadability, right=default_direction, on=['load_zone_from', 'load_zone_to', 'scenario'], how='left')

    #Give order (though it is not necessary)
    flows_loadability['sc_order'] = flows_loadability['scenario'].map(order)
    flows_loadability = flows_loadability.sort_values('sc_order').drop('sc_order',axis=1)

    return flows_loadability

def total_tx_losses_by_period(scenario_file_name: str, 
                              analysis_scenarios: list,
                              analysis_zones: list):
    
    # Dataframe that provides the transmission losses.
    # Output:
    #   month	scenario	    ts_period       losses   
    #   1	    wecc_10_10	    85,865.70	    1,416.05
    #	1	    wecc_140_170	85,865.70	    1,545.16
    #	2	    wecc_10_10	    78,827.29	    1,374.09
    #	2	    wecc_140_170	78,827.29	    1,400.50
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenario_file_name)

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]  

    tx_dispatch = get_data(scenario, "transmission_dispatch.csv", usecols=['load_zone_from', 'load_zone_to', 'timestamp', 'transmission_dispatch'])
    tx_dispatch  = tx_dispatch  .replace({"scenario": short_names})

    time_info = timestamp_info(scenario_file_name, analysis_scenarios)[1]
    tx_dispatch = pd.merge(left=tx_dispatch, right=time_info[['scenario', 'ts_period', 'timestamp', 'tp_weight_in_year']], on=['scenario', 'timestamp'])
    tx_dispatch['flow_gwh']=tx_dispatch['transmission_dispatch']*tx_dispatch['tp_weight_in_year']/10**3

    txs = get_data(scenario, "transmission_lines.csv", fpath='inputs',  usecols=['trans_lz1','trans_lz2','trans_efficiency'])
    txs.replace({"scenario": short_names}, inplace=True)
    txs_copy=txs.copy() #Duplicate the table to have NM in the sending zone and receiving zone
    txs_copy.rename(columns={'trans_lz1':'trans_lz2', 'trans_lz2':'trans_lz1'}, inplace=True)
    txs=pd.concat([txs,txs_copy])
    txs.reset_index(inplace=True, drop=True)
    txs.rename(columns={'trans_lz1': 'load_zone_from', 'trans_lz2': 'load_zone_to'}, inplace=True)

    tx_dispatch = pd.merge(left= tx_dispatch, right= txs, on=['scenario', 'load_zone_from', 'load_zone_to'], how='left')

    if len(analysis_zones)!= 0:
        tx_dispatch = tx_dispatch.loc[tx_dispatch.load_zone_from.isin(analysis_zones) & tx_dispatch.load_zone_to.isin(analysis_zones)]

    tx_dispatch['losses']=tx_dispatch['flow_gwh']*(1-tx_dispatch['trans_efficiency'])

    period_tx_losses_df =  tx_dispatch.pivot_table(index=['ts_period', 'scenario'], values='losses', aggfunc=np.sum)
    period_tx_losses_df.reset_index(inplace=True)

    return period_tx_losses_df

# --------------------------------------------------         -----------------------------------------------------------------------------------------------------
# -------------------------------------------------- STORAGE -----------------------------------------------------------------------------------------------------
# --------------------------------------------------         -----------------------------------------------------------------------------------------------------

def energy_storage_by_scenario(scenarios_file_name: str, 
                               analysis_scenarios: list, 
                               analysis_period: list, 
                               analysis_zones:list, 
                               baseline_scenario: str):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 
    
    storage_capacity = get_data(scenario, "storage_capacity.csv")
    storage_capacity.replace({"scenario": short_names}, inplace=True)
    
    analysis = storage_capacity.copy()
    analysis.rename(columns = {'OnlineEnergyCapacityMWh': 'EnergyCapacity_MWh'}, inplace = True)
    analysis['Storage (energy)'] = analysis['EnergyCapacity_MWh'] / 10**3

    if len(analysis_zones)==0:
        analysis_zones = list(analysis.load_zone.unique())

    if len(analysis_period)==0:
        analysis_zones = list(analysis.period.unique())

    analysis = analysis.loc[(analysis.period.isin(analysis_period)) & (analysis.load_zone.isin(analysis_zones))]
    analysis_by_sc = analysis.pivot_table(index='scenario', values='Storage (energy)', aggfunc=np.sum)
    analysis_by_sc['sc_order'] = analysis_by_sc .index.map(order)
    analysis_by_sc =analysis_by_sc .sort_values('sc_order').drop('sc_order',axis=1)

    if baseline_scenario==0:
        return analysis_by_sc
    else:
        relative_analysis_by_sc =analysis_by_sc .copy()
        relative_analysis_by_sc.loc['base',:]=relative_analysis_by_sc .loc[baseline_scenario,:]

        for r in order:
            relative_analysis_by_sc.loc[r]=relative_analysis_by_sc.loc[r]/relative_analysis_by_sc.loc['base']*100
    
        relative_analysis_by_sc.drop(index=('base'), inplace=True)

        return analysis_by_sc, relative_analysis_by_sc

def get_storage_cost_by_scenario(scenarios_file_name: str, 
                                 analysis_scenarios: list,
                                 projects: list):
    # Function that returns the storage power capacity cost and storage energy capacity cost of the given project
    # for the scenarios under analysis.
    # Output:
    #  	gen_overnight_cost	gen_storage_energy_overnight_cost	scenario
    #  	10	                10	                                wecc_10_10
    #	10	                20	                                wecc_10_20
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    gen_build_costs = get_data(scenario, "gen_build_costs.csv", fpath='inputs')
    gen_build_costs.replace({"scenario": short_names}, inplace=True)
    gen_build_costs =  gen_build_costs[ gen_build_costs.GENERATION_PROJECT.isin(projects)]
    gen_build_costs = gen_build_costs[['gen_overnight_cost', 'gen_storage_energy_overnight_cost', 'scenario']]
    gen_build_costs['gen_overnight_cost'] = gen_build_costs['gen_overnight_cost'].astype(float)/1000
    gen_build_costs.gen_overnight_cost = gen_build_costs.gen_overnight_cost.round()
    gen_build_costs['gen_overnight_cost'] = gen_build_costs['gen_overnight_cost'].astype(int) #to avoid .00 
    gen_build_costs['gen_storage_energy_overnight_cost'] = gen_build_costs['gen_storage_energy_overnight_cost'].astype(float)/1000
    gen_build_costs.gen_storage_energy_overnight_cost = gen_build_costs.gen_storage_energy_overnight_cost.round()
    gen_build_costs['gen_storage_energy_overnight_cost'] = gen_build_costs['gen_storage_energy_overnight_cost'].astype(int) #to avoid .00
    return gen_build_costs

def energy_capacity_storage(scenarios_file_name: str, 
                            analysis_scenarios: list,
                            only_one_period: list, 
                            baseline_scenario, 
                            zones: list):
    # Returns total online energy capacity of the storage assets in the zones.
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    storage = get_data(scenario, "storage_capacity.csv", usecols=['period', 'load_zone', 'OnlineEnergyCapacityMWh'])
    storage = storage  .replace({"scenario": short_names})
    if len(zones) != 0:
        storage = storage.loc[storage.load_zone.isin(zones)]
    storage = storage.loc[storage.period.isin(only_one_period)]
    storage = storage.pivot_table(index=['scenario'], values=['OnlineEnergyCapacityMWh'], aggfunc=np.sum)
    storage.reset_index(inplace = True)
    if baseline_scenario!='':
        storage['base']=storage.loc[storage.scenario==baseline_scenario,'OnlineEnergyCapacityMWh'].iloc[0]
        storage['relative']=storage['OnlineEnergyCapacityMWh']/storage['base']
    storage['sc_order'] = storage['scenario'].map(order)
    storage = storage.sort_values('sc_order').drop('sc_order',axis=1)
    return storage 

def power_capacity_storage(scenarios_file_name: str,
                           analysis_scenarios: list,
                           only_one_period: list, 
                           baseline_scenario, 
                           zones: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    storage = get_data(scenario, "storage_capacity.csv", usecols=['period', 'load_zone', 'OnlinePowerCapacityMW'])
    storage = storage  .replace({"scenario": short_names})
    if len(zones) != 0:
        storage = storage.loc[storage.load_zone.isin(zones)]
    storage = storage.loc[storage.period.isin(only_one_period)]
    storage = storage.pivot_table(index=['scenario'], values=['OnlinePowerCapacityMW'], aggfunc=np.sum)
    storage.reset_index(inplace = True)
    if baseline_scenario!='':
        storage['base']=storage.loc[storage.scenario==baseline_scenario,'OnlinePowerCapacityMW'].iloc[0]
        storage['relative']=storage['OnlinePowerCapacityMW']/storage['base']
    storage['sc_order'] = storage['scenario'].map(order)
    storage = storage.sort_values('sc_order').drop('sc_order',axis=1)
    
    return storage 

def duration_storage(scenarios_file_name: str,
                     analysis_scenarios: list,
                     only_one_period: list, 
                     baseline_scenario, zones: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 
    
    storage = get_data(scenario, "storage_capacity.csv", usecols=['period', 'load_zone', 'OnlineEnergyCapacityMWh', 'OnlinePowerCapacityMW'])
    storage = storage  .replace({"scenario": short_names})
    if len(zones) != 0:
        storage = storage.loc[storage.load_zone.isin(zones)]
    storage = storage.loc[storage.period.isin(only_one_period)]
    storage = storage.pivot_table(index=['scenario'], values=['OnlineEnergyCapacityMWh', 'OnlinePowerCapacityMW'], aggfunc=np.sum)
    storage.reset_index(inplace = True)
    storage['duration'] = storage['OnlineEnergyCapacityMWh']/ storage['OnlinePowerCapacityMW']
    storage.drop(columns=['OnlineEnergyCapacityMWh', 'OnlinePowerCapacityMW'], inplace=True)
    if len(baseline_scenario)>0:
        storage['base']=storage.loc[storage.scenario == baseline_scenario,'duration'].iloc[0]
        storage['relative']=storage['duration']/storage['base']
    storage['sc_order'] = storage['scenario'].map(order)
    storage = storage.sort_values('sc_order').drop('sc_order',axis=1)
    
    return storage 

def avg_duration_storage(scenarios_file_name: str,
                         analysis_scenarios: list,
                         only_one_period: list, 
                         baseline_scenario, 
                         zones: list):
    # Returns the average duration (h) by: average ( duration(zone) for zone in specified list)
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    storage = get_data(scenario, "storage_capacity.csv", usecols=['period', 'load_zone', 'OnlineEnergyCapacityMWh', 'OnlinePowerCapacityMW'])
    storage = storage  .replace({"scenario": short_names})
    if len(zones) != 0:
        storage = storage.loc[storage.load_zone.isin(zones)]
    storage = storage.loc[storage.period.isin(only_one_period)]
    storage.reset_index(inplace = True) # Use reset before dropping with index.
    storage.drop(storage[storage.OnlinePowerCapacityMW ==0].index, inplace=True)
    storage['duration'] = storage['OnlineEnergyCapacityMWh']/ storage['OnlinePowerCapacityMW']
    storage.reset_index(inplace = True)
    storage.drop(columns=['OnlineEnergyCapacityMWh', 'OnlinePowerCapacityMW'], inplace=True)
    storage = storage.pivot_table(index = 'scenario', values='duration', aggfunc = np.average)
    storage.reset_index(inplace = True)
    if baseline_scenario!='':
        storage['base']=storage.loc[storage.scenario==baseline_scenario,'duration'].iloc[0]
        storage['relative']=storage['duration']/storage['base']
    storage['sc_order'] = storage['scenario'].map(order)
    storage = storage.sort_values('sc_order').drop('sc_order',axis=1)

    return storage 

def duration_storage_by_zone(scenarios_file_name: list, 
                             analysis_scenarios: list,
                             only_one_period: list, 
                             zones: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    storage = get_data(scenario, "storage_capacity.csv", usecols=['period', 'load_zone', 'OnlineEnergyCapacityMWh', 'OnlinePowerCapacityMW'])
    storage = storage  .replace({"scenario": short_names})
    if len(zones) != 0:
        storage = storage.loc[storage.load_zone.isin(zones)]
    storage = storage.loc[storage.period.isin(only_one_period)]
    storage.reset_index(inplace = True, drop=True)
    storage.drop(storage[storage.OnlinePowerCapacityMW ==0].index, inplace=True)
    storage['duration'] = storage['OnlineEnergyCapacityMWh']/ storage['OnlinePowerCapacityMW']
    storage['sc_order'] = storage['scenario'].map(order)
    storage = storage.sort_values('sc_order').drop('sc_order',axis=1)

    return storage 

# ---------------------------------------------------        ----------------------------------------------------------------------------------------------------
# --------------------------------------------------- DEMAND -----------------------------------------------------------------------------------------------------
# ---------------------------------------------------        -----------------------------------------------------------------------------------------------------

def get_timestamp_of_peak_demand_in_zone(scenarios_file_name: str, 
                                         analysis_scenarios: list,
                                         analysis_zones: list):

    # Function that returns the timestamp where the highest demand happen for the zones under analysis.
    # 
    # Output
    #   scenario	timestamp	tp_weight_in_year	energy_MWh	zone_demand_mw
    #   ----------------------------------------------------------------------
    #	high_10	    2050072504	4.01	            341,310.72	85,035.66

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    hours_per_year = 8766

    timepoints = get_data(scenario, "timepoints.csv", fpath='inputs')
    timepoints.columns= timepoints .columns.str.lower()
    timepoints.rename(columns={'timepoint_id': 'timepoint'}, inplace=True)
    timepoints = timepoints .replace({"scenario": short_names})

    timeseries = get_data(scenario, "timeseries.csv", fpath='inputs')
    timeseries.columns= timeseries .columns.str.lower()
    timeseries = timeseries .replace({"scenario": short_names})

    periods = get_data(scenario, "periods.csv", fpath='inputs')
    periods.columns= periods .columns.str.lower()
    periods.rename(columns={'investment_period': 'ts_period'}, inplace=True)
    periods = periods .replace({"scenario": short_names})

    time_info=pd.merge(left=timeseries,right=timepoints, on=['timeseries', 'scenario'])
    time_info=pd.merge(left=time_info,right=periods , on=['scenario', 'ts_period'])

    time_info['tp_weight']=time_info['ts_duration_of_tp']*time_info['ts_scale_to_period']

    time_info = time_info[['scenario', 'ts_period', 'timestamp', 'tp_weight']]

    period_info=time_info.pivot_table(index=['scenario','ts_period'], values='tp_weight',aggfunc=np.sum )
    period_info.reset_index(inplace=True)
    period_info.rename(columns={'tp_weight': 'hours_in_period'}, inplace=True)
    period_info = pd.merge(left=period_info, right = periods, on=['scenario', 'ts_period'])
    period_info['err_plain'] = (period_info['period_end'] - period_info['period_start'])*hours_per_year - period_info['hours_in_period']
    period_info['err_add_one'] =  (period_info['period_end'] + 1 - period_info['period_start'])*hours_per_year - period_info['hours_in_period']
    period_info.loc[:, 'add_one_to_period_end_rule']= period_info.apply(lambda x: 1 if np.absolute(x['err_add_one'])<np.absolute(x['err_plain']) else 0, axis=1)
    period_info['period_length_years'] = period_info['period_end'] - period_info['period_start'] + period_info['add_one_to_period_end_rule']

    period_info =  period_info[['scenario', 'ts_period', 'period_length_years']]

    time_df = pd.merge(left = time_info, right = period_info, on=['scenario', 'ts_period'] ).reset_index(drop=True)
    time_df['tp_weight_in_year'] = time_df['tp_weight']/time_df['period_length_years']

    load = get_data(scenario, "load_balance.csv", usecols=['load_zone', 'timestamp', 'zone_demand_mw'])
    load = load .replace({"scenario": short_names})
   
    load ['zone_demand_mw'] = load ['zone_demand_mw']*-1

    load = load.loc[load.load_zone.isin(analysis_zones)]
    
    load = pd.merge(left = load, right = time_df[['scenario', 'timestamp', 'tp_weight_in_year']], on=['scenario', 'timestamp'], how='left')
    load['energy_MWh'] = load['zone_demand_mw'] *  load['tp_weight_in_year'] 

    load_of_zones = load.pivot_table(index = ['scenario', 'timestamp', 'tp_weight_in_year'], values= ['energy_MWh','zone_demand_mw'] , aggfunc=np.sum)
    load_of_zones.reset_index(inplace=True)
    load_of_zones.to_csv('test1.csv')
    max_load = load_of_zones.pivot_table(index = ['scenario'], values = ['zone_demand_mw'], aggfunc=np.max)
    max_load.reset_index(inplace=True)
    get_timestamp = pd.merge(left = load_of_zones, right = max_load , on = ['scenario', 'zone_demand_mw'])

    return get_timestamp

def get_timestamp_of_lowest_xch_ratio_of_zones(scenarios_file_name: str, 
                                               analysis_scenarios: list,
                                               analysis_zones: list):

    # Function that returns the timestamp where the lowest ratio (Injection/Withdrawal) happens for the zones under analysis.
    #
    # Output:
    #   scenario	timestamp	StorageNetCharge	TXPowerNet	ZoneTotalCentralDispatch	withdrawal	zone_demand_mw	ratio	tp_weight_in_year
    #   -------------------------------------------------------------------------------------------------------------------------------------------
    #   high_10	    2050040512	-548.66	            14,591.65	27,923.86	                42,515.07	-41,966.40	    0.66	4.01
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    load_balance = get_data(scenario, 'load_balance.csv', usecols = ['load_zone', 'timestamp', 'ZoneTotalCentralDispatch', 'zone_demand_mw', 'StorageNetCharge', 'TXPowerNet'])
    load_balance['withdrawal']=-(load_balance['zone_demand_mw'] + load_balance['StorageNetCharge'])
    load_balance.replace(short_names, inplace=True)
    
    if len(analysis_zones) != 0 :
        load_balance =  load_balance.loc[load_balance.load_zone.isin(analysis_zones)]

    aggregated_load_balance = load_balance.pivot_table(index=['scenario', 'timestamp'], values =['ZoneTotalCentralDispatch', 'withdrawal', 'zone_demand_mw', 'StorageNetCharge', 'TXPowerNet'], aggfunc=np.sum)
    aggregated_load_balance.reset_index(inplace=True)
    aggregated_load_balance ['ratio'] =  aggregated_load_balance['ZoneTotalCentralDispatch'] / aggregated_load_balance['withdrawal']

    lowest_ratio = aggregated_load_balance[['scenario', 'ratio']].pivot_table(index=['scenario'], values='ratio', aggfunc =np.min)
    lowest_ratio.reset_index(inplace=True)

    aggreg_load_balance_at_low_ratio = pd.merge (left =  aggregated_load_balance, right = lowest_ratio, on=['scenario', 'ratio'])

    time_df = timestamp_info(scenarios_file_name, analysis_scenarios)[1]

    time_df = time_df[['scenario', 'timestamp', 'tp_weight_in_year']]

    aggreg_load_balance_at_low_ratio = pd.merge(left = aggreg_load_balance_at_low_ratio, right = time_df, on = ['scenario', 'timestamp'])

    return aggreg_load_balance_at_low_ratio

def monthly_demand(scenarios_file_name: str, 
                   analysis_scenarios: list,
                   analysis_zones: list, 
                   time_zone: str):
    # It returns the monthly demand (GWh) for the selected zones. If the latter is nothing, then the system load is computed.
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name)
    
    if len(analysis_scenarios) >0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 
    
    load_balance = get_data(scenario, "load_balance.csv", usecols=['load_zone', 'timestamp', 'zone_demand_mw'])
    load_balance  = load_balance  .replace({"scenario": short_names})
    load_balance .columns= load_balance .columns.str.lower()
    load_balance['zone_demand_mw'] = -load_balance['zone_demand_mw']

    if len(analysis_zones)==0:
        analysis_zones = list(load_balance.load_zone.unique())

    load_balance = load_balance.pivot_table(index=['timestamp', 'scenario'], values='zone_demand_mw', aggfunc=np.sum)
    load_balance.reset_index(inplace=True)

    time_info = timestamp_info(scenarios_file_name, analysis_scenarios)[1]
    load_balance = pd.merge(left=load_balance, right=time_info[['scenario', 'timestamp', 'tp_weight_in_year']], on=['scenario', 'timestamp'])
    load_balance['zone_demand_gwh']=load_balance['zone_demand_mw']*load_balance['tp_weight_in_year']/10**3

    load_balance["timestamp"]=pd.to_datetime(load_balance["timestamp"], format='%Y%m%d%H', utc=True)
    load_balance["timestamp"]=load_balance["timestamp"].dt.tz_convert(time_zone)

    load_balance['month']=load_balance['timestamp'].dt.month

    load_balance =  load_balance.pivot_table(index=['month', 'scenario'], values='zone_demand_gwh', aggfunc=np.sum)
    load_balance.reset_index(inplace=True)

    return load_balance

# ------------------------------------------------------------------------       --------------------------------------------------------------
#------------------------------------------------------------------------- COSTS --------------------------------------------------------------
# ------------------------------------------------------------------------       --------------------------------------------------------------

def power_generation_fixed_costs(scenarios_file_name: str, 
                                 analysis_scenarios: list, 
                                 analysis_zones: list, 
                                 analysis_tech: list):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name) 
    period_info = timestamp_info(scenarios_file_name, analysis_scenarios)[0]
    
    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    financials = get_data(scenario, "financials.csv", fpath='inputs')
    financials  = financials  .replace({"scenario": short_names})

    period_info = pd.merge(left=period_info, right=financials, on='scenario')
    period_info.loc[:, 'bring_annual_costs_to_base_year']=period_info.apply(lambda x: uniform_series_to_present_value(x['discount_rate'], x['period_length_years'])
                                                                        *future_to_present_value(x['discount_rate'], x['period_start'] - x['base_financial_year']), axis=1)
    period_info.rename(columns={'ts_period': 'investment_period'}, inplace=True)

    generation_projects_info = get_data(scenario, "generation_projects_info.csv" , fpath='inputs')
    generation_projects_info  = generation_projects_info.replace({"scenario": short_names})
    generation_projects_info  = default_value(generation_projects_info, "gen_connect_cost_per_mw", 0)

    gen_build_costs = get_data(scenario, "gen_build_costs.csv", fpath='inputs')
    gen_build_costs  = gen_build_costs  .replace({"scenario": short_names})
    gen_build_costs  = default_value(gen_build_costs, "gen_overnight_cost", 0)
    gen_build_costs  = default_value(gen_build_costs, "gen_fixed_om", 0)
    gen_build_costs  = default_value(gen_build_costs, "gen_storage_energy_overnight_cost", 0)

    gen_build_costs_extended = pd.merge(left=gen_build_costs , right=financials, on='scenario')
    gen_build_costs_extended = pd.merge(left=gen_build_costs_extended , right=generation_projects_info[['GENERATION_PROJECT', 'gen_load_zone', "gen_connect_cost_per_mw", 'gen_max_age', 'scenario']], on=['GENERATION_PROJECT','scenario'])
    gen_build_costs_extended = gen_build_costs_extended[['GENERATION_PROJECT', 'gen_load_zone', 'build_year', 'gen_overnight_cost', 'gen_fixed_om', 'gen_storage_energy_overnight_cost', 'gen_max_age', "gen_connect_cost_per_mw", "interest_rate", "scenario"]]

    gen_build_costs_extended.loc[:,'gen_capital_cost_annual'] = gen_build_costs_extended.apply(lambda x: (x['gen_overnight_cost'] + x['gen_connect_cost_per_mw']) * crf(x['interest_rate'], x['gen_max_age']), axis=1)
    gen_build_costs_extended.loc[:,'storage_energy_capital_cost_annual'] = gen_build_costs_extended.apply(lambda x: x['gen_storage_energy_overnight_cost'] * crf(x['interest_rate'], x['gen_max_age']), axis=1)

    BuildGen = get_data(scenario, "BuildGen.csv")
    BuildGen  = BuildGen  .replace({"scenario": short_names})
    BuildGen.rename(columns={'GEN_BLD_YRS_1':'GENERATION_PROJECT', "GEN_BLD_YRS_2": "build_year"},inplace=True)

    gen_build_costs_extended = pd.merge(left = gen_build_costs_extended, right=BuildGen, on=['GENERATION_PROJECT', 'build_year', 'scenario'])

    BLD_YRS_FOR_GEN_PERIOD = pd.merge(left=generation_projects_info[['GENERATION_PROJECT', 'gen_load_zone', 'gen_max_age', 'scenario']], right=period_info[['investment_period', 'period_start', 'period_length_years','scenario']], on='scenario')
    BLD_YRS_FOR_GEN_PERIOD = pd.merge(left=BLD_YRS_FOR_GEN_PERIOD, right=gen_build_costs[['GENERATION_PROJECT', 'build_year', 'scenario']], on=['GENERATION_PROJECT', 'scenario'])
    BLD_YRS_FOR_GEN_PERIOD.loc[:,'operation']=BLD_YRS_FOR_GEN_PERIOD.apply(lambda x: gen_build_can_operate_in_period(x['gen_max_age'], x['build_year'], x['investment_period'], x['period_start'], x['period_length_years']), axis=1)    

    gen_costs=pd.merge(left=BLD_YRS_FOR_GEN_PERIOD[['scenario', 'GENERATION_PROJECT', 'gen_load_zone', 'investment_period', 'build_year', 'operation']], 
                       right=gen_build_costs_extended[['scenario', 'gen_load_zone', 'GENERATION_PROJECT','build_year','BuildGen', 'gen_capital_cost_annual', 'gen_fixed_om']]
                       , on=['GENERATION_PROJECT', 'gen_load_zone', 'scenario', 'build_year'])

    gen_costs['GenCapitalCosts'] = gen_costs['BuildGen'] * gen_costs['gen_capital_cost_annual'] *gen_costs['operation']
    gen_costs['GenFixedOMCosts'] = gen_costs['BuildGen'] * gen_costs['gen_fixed_om'] *gen_costs['operation']
    gen_costs['TotalGenFixedCosts'] = gen_costs['GenCapitalCosts'] + gen_costs['GenFixedOMCosts']*gen_costs['operation']

    gen_costs = pd.merge(left=gen_costs, right=generation_projects_info[['GENERATION_PROJECT', 'scenario', 'tech_map']], on=['GENERATION_PROJECT','scenario']) #Here we classify by tech

    if len(analysis_zones)>0:
       gen_costs = gen_costs.loc[gen_costs.gen_load_zone.isin(analysis_zones)]

    TotalGenFixedCosts = gen_costs.pivot_table(
        index=['scenario','investment_period', 'tech_map'], values="TotalGenFixedCosts", aggfunc=np.sum )

    TotalGenFixedCosts.rename(columns={'TotalGenFixedCosts' : 'AnnualCost_Real'}, inplace=True)
    TotalGenFixedCosts.reset_index(inplace=True)
    TotalGenFixedCosts = pd.merge(TotalGenFixedCosts, period_info[['scenario', 'investment_period', 'bring_annual_costs_to_base_year']], on=['scenario', 'investment_period'])
    TotalGenFixedCosts ['AnnualCost_NPV'] = TotalGenFixedCosts ['AnnualCost_Real'] * TotalGenFixedCosts ['bring_annual_costs_to_base_year']
    TotalGenFixedCosts.drop(['bring_annual_costs_to_base_year'], axis=1, inplace=True)
    TotalGenFixedCosts['Component'] = 'TotalGenFixedCosts'
    TotalGenFixedCosts['Component_type'] = 'annual'
    TotalGenFixedCosts = TotalGenFixedCosts[['scenario', 'tech_map' , 'investment_period', 'Component', 'Component_type', 'AnnualCost_NPV', 'AnnualCost_Real']]
    TotalGenFixedCosts['sc_order'] = TotalGenFixedCosts.index.map(order)
    TotalGenFixedCosts = TotalGenFixedCosts.sort_values('sc_order').drop('sc_order',axis=1)

    if len(analysis_tech)>0:
       TotalGenFixedCosts =  TotalGenFixedCosts.loc[TotalGenFixedCosts.tech_map.isin(analysis_tech)]
   
    TotalGenFixedCosts['AnnualCost_NPV'] =  TotalGenFixedCosts['AnnualCost_NPV'] /10**9 # Put the numbers in USD billion 
    TotalGenFixedCosts['AnnualCost_Real'] =  TotalGenFixedCosts['AnnualCost_Real'] /10**9 # Put the numbers in USD billion 

    return TotalGenFixedCosts 

def storage_energy_fixed_costs(scenarios_file_name: str, 
                               analysis_scenarios: list, 
                               analysis_zones: list):

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name) 
    period_info = timestamp_info(scenarios_file_name, analysis_scenarios)[0]
    
    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    financials = get_data(scenario, "financials.csv", fpath='inputs')
    financials  = financials  .replace({"scenario": short_names})

    period_info = pd.merge(left=period_info, right=financials, on='scenario')
    period_info.loc[:, 'bring_annual_costs_to_base_year']=period_info.apply(lambda x: uniform_series_to_present_value(x['discount_rate'], x['period_length_years'])
                                                                        *future_to_present_value(x['discount_rate'], x['period_start'] - x['base_financial_year']), axis=1)
    period_info.rename(columns={'ts_period': 'investment_period'}, inplace=True)

    generation_projects_info = get_data(scenario, "generation_projects_info.csv" , fpath='inputs')
    generation_projects_info  = generation_projects_info.replace({"scenario": short_names})
    generation_projects_info  = default_value(generation_projects_info, "gen_connect_cost_per_mw", 0)

    gen_build_costs = get_data(scenario, "gen_build_costs.csv", fpath='inputs')
    gen_build_costs  = gen_build_costs  .replace({"scenario": short_names})
    gen_build_costs  = default_value(gen_build_costs, "gen_overnight_cost", 0)
    gen_build_costs  = default_value(gen_build_costs, "gen_fixed_om", 0)
    gen_build_costs  = default_value(gen_build_costs, "gen_storage_energy_overnight_cost", 0)

    gen_build_costs_extended = pd.merge(left=gen_build_costs , right=financials, on='scenario')
    gen_build_costs_extended = pd.merge(left=gen_build_costs_extended , right=generation_projects_info[['GENERATION_PROJECT', 'gen_load_zone', "gen_connect_cost_per_mw", 'gen_max_age', 'scenario']], on=['GENERATION_PROJECT','scenario'])
    gen_build_costs_extended = gen_build_costs_extended[['GENERATION_PROJECT', 'gen_load_zone', 'build_year', 'gen_overnight_cost', 'gen_fixed_om', 'gen_storage_energy_overnight_cost', 'gen_max_age', "gen_connect_cost_per_mw", "interest_rate", "scenario"]]

    gen_build_costs_extended.loc[:,'gen_capital_cost_annual'] = gen_build_costs_extended.apply(lambda x: (x['gen_overnight_cost'] + x['gen_connect_cost_per_mw']) * crf(x['interest_rate'], x['gen_max_age']), axis=1)
    gen_build_costs_extended.loc[:,'storage_energy_capital_cost_annual'] = gen_build_costs_extended.apply(lambda x: x['gen_storage_energy_overnight_cost'] * crf(x['interest_rate'], x['gen_max_age']), axis=1)

    BuildGen = get_data(scenario, "BuildGen.csv")
    BuildGen  = BuildGen  .replace({"scenario": short_names})
    BuildGen.rename(columns={'GEN_BLD_YRS_1':'GENERATION_PROJECT', "GEN_BLD_YRS_2": "build_year"},inplace=True)

    gen_build_costs_extended = pd.merge(left = gen_build_costs_extended, right=BuildGen, on=['GENERATION_PROJECT', 'build_year', 'scenario'])

    BLD_YRS_FOR_GEN_PERIOD = pd.merge(left=generation_projects_info[['GENERATION_PROJECT', 'gen_load_zone', 'gen_max_age', 'scenario']], right=period_info[['investment_period', 'period_start', 'period_length_years','scenario']], on='scenario')
    BLD_YRS_FOR_GEN_PERIOD = pd.merge(left=BLD_YRS_FOR_GEN_PERIOD, right=gen_build_costs[['GENERATION_PROJECT', 'build_year', 'scenario']], on=['GENERATION_PROJECT', 'scenario'])
    BLD_YRS_FOR_GEN_PERIOD.loc[:,'operation']=BLD_YRS_FOR_GEN_PERIOD.apply(lambda x: gen_build_can_operate_in_period(x['gen_max_age'], x['build_year'], x['investment_period'], x['period_start'], x['period_length_years']), axis=1)    

    storage_build_costs_extended = gen_build_costs_extended.copy()

    BuildStorageEnergy = get_data(scenario, "BuildStorageEnergy.csv")
    BuildStorageEnergy  = BuildStorageEnergy .replace({"scenario": short_names})
    BuildStorageEnergy.rename(columns={'STORAGE_GEN_BLD_YRS_1':'GENERATION_PROJECT', "STORAGE_GEN_BLD_YRS_2": "build_year"},inplace=True)

    storage_build_costs_extended = pd.merge(left=storage_build_costs_extended, right=BuildStorageEnergy, on=['GENERATION_PROJECT', 'build_year', 'scenario'])

    storage_costs=pd.merge(left=BLD_YRS_FOR_GEN_PERIOD[['scenario', 'GENERATION_PROJECT', 'gen_load_zone', 'investment_period', 'build_year', 'operation']], 
                   right=storage_build_costs_extended[['scenario', 'GENERATION_PROJECT', 'gen_load_zone', 'build_year','BuildStorageEnergy', 'storage_energy_capital_cost_annual']]
                   , on=['GENERATION_PROJECT', 'gen_load_zone', 'scenario', 'build_year'])

    storage_costs['StorageEnergyFixedCost'] = storage_costs['BuildStorageEnergy'] * storage_costs['storage_energy_capital_cost_annual'] *storage_costs['operation']

    if len(analysis_zones)>0:
        storage_costs = storage_costs.loc[storage_costs.gen_load_zone.isin(analysis_zones)]

    StorageEnergyFixedCost = storage_costs.pivot_table(
        index=['scenario','investment_period'], values="StorageEnergyFixedCost", aggfunc=np.sum )
    StorageEnergyFixedCost.rename(columns={'StorageEnergyFixedCost' : 'AnnualCost_Real'}, inplace=True)
    StorageEnergyFixedCost.reset_index(inplace=True)
    StorageEnergyFixedCost = pd.merge(StorageEnergyFixedCost, period_info[['scenario', 'investment_period', 'bring_annual_costs_to_base_year']], on=['scenario', 'investment_period'])
    StorageEnergyFixedCost ['AnnualCost_NPV'] = StorageEnergyFixedCost ['AnnualCost_Real'] * StorageEnergyFixedCost ['bring_annual_costs_to_base_year']
    StorageEnergyFixedCost.drop(['bring_annual_costs_to_base_year'], axis=1, inplace=True)
    StorageEnergyFixedCost['Component'] = 'StorageEnergyFixedCost'
    StorageEnergyFixedCost['Component_type'] = 'annual'
    StorageEnergyFixedCost = StorageEnergyFixedCost[['scenario', 'investment_period', 'Component', 'Component_type', 'AnnualCost_NPV', 'AnnualCost_Real']]

    return StorageEnergyFixedCost

def generation_variable_costs(scenarios_file_name: str, 
                              analysis_scenarios: list, 
                              analysis_zones: list):

    # Function that returns the generation variable costs (NPV and real cost) of the scenarios under analysis.
    #
    # Output: Dataframe with structure:
    #   scenario	investment_period	Component	            Component_type	AnnualCost_NPV	AnnualCost_Real
    #   -------------------------------------------------------------------------------------------------------
    #   high_100	2050	            GenVariableOMCostsInTP	timepoint	    422,265,276.67	214,373,379.87
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name) 
    period_info = timestamp_info(scenarios_file_name, analysis_scenarios)[0]
    
    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios] 

    financials = get_data(scenario, "financials.csv", fpath='inputs')
    financials  = financials  .replace({"scenario": short_names})

    period_info = pd.merge(left=period_info, right=financials, on='scenario')
    period_info.loc[:, 'bring_annual_costs_to_base_year']=period_info.apply(lambda x: uniform_series_to_present_value(x['discount_rate'], x['period_length_years'])
                                                                        *future_to_present_value(x['discount_rate'], x['period_start'] - x['base_financial_year']), axis=1)
    period_info.rename(columns={'ts_period': 'investment_period'}, inplace=True)

    dispatch = get_data(scenario, "dispatch_zonal_annual_summary.csv") #you may want to use dispatch.csv, but the run time is extremely long because of the weigh of the file.
    dispatch  = dispatch  .replace({"scenario": short_names})
    dispatch.rename(columns={'period': 'investment_period', 'VariableOMCost_per_yr': 'GenVariableOMCosts'}, inplace=True)

    if len(analysis_zones)>0:
        dispatch = dispatch.loc[dispatch.gen_load_zone.isin(analysis_zones)]

    GenVariableOMCostsInTP = dispatch.pivot_table(
            index=['scenario','investment_period'], values="GenVariableOMCosts", aggfunc=np.sum )
    GenVariableOMCostsInTP.rename(columns={'GenVariableOMCosts' : 'AnnualCost_Real'}, inplace=True)
    GenVariableOMCostsInTP.reset_index(inplace=True)
    GenVariableOMCostsInTP = pd.merge(GenVariableOMCostsInTP, period_info[['scenario', 'investment_period', 'bring_annual_costs_to_base_year']], on=['scenario', 'investment_period'])
    GenVariableOMCostsInTP ['AnnualCost_NPV'] = GenVariableOMCostsInTP ['AnnualCost_Real'] * GenVariableOMCostsInTP ['bring_annual_costs_to_base_year']
    GenVariableOMCostsInTP.drop(['bring_annual_costs_to_base_year'], axis=1, inplace=True)
    GenVariableOMCostsInTP['Component'] = 'GenVariableOMCostsInTP'
    GenVariableOMCostsInTP['Component_type'] = 'timepoint'
    GenVariableOMCostsInTP = GenVariableOMCostsInTP[['scenario', 'investment_period', 'Component', 'Component_type', 'AnnualCost_NPV', 'AnnualCost_Real']]
    
    return GenVariableOMCostsInTP

def transmission_fixed_costs(scenarios_file_name: str, 
                             analysis_scenarios: list, 
                             analysis_zones: list):
    
    # Function that returns the transmission fixed costs of the scenarios under analysis.
    #
    # Output: Dataframe with structure:
    #
    #   scenario	investment_period	Component	    Component_type	AnnualCost_NPV	    AnnualCost_Real
    #   ------------------------------------------------------------------------------------------------
    #   high_100	2050	            TxFixedCosts	annual	        6,791,580,521.98	3,447,913,318.00

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name) 
    period_info = timestamp_info(scenarios_file_name, analysis_scenarios)[0]

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]     

    financials = get_data(scenario, "financials.csv", fpath='inputs')
    financials  = financials  .replace({"scenario": short_names})

    period_info = pd.merge(left=period_info, right=financials, on='scenario')
    period_info.loc[:, 'bring_annual_costs_to_base_year']=period_info.apply(lambda x: uniform_series_to_present_value(x['discount_rate'], x['period_length_years'])
                                                                        *future_to_present_value(x['discount_rate'], x['period_start'] - x['base_financial_year']), axis=1)
    period_info.rename(columns={'ts_period': 'investment_period'}, inplace=True)

    transmission = get_data(scenario, "transmission.csv")
    transmission = transmission[['PERIOD', 'trans_lz1', 'trans_lz2','TotalAnnualCost', 'scenario']]
    transmission  = transmission  .replace({"scenario": short_names})
    transmission.rename(columns={'TotalAnnualCost': 'TxFixedCosts', 'PERIOD':'investment_period'}, inplace=True)

    if len(analysis_zones)>0:
        transmission  = transmission.loc[(transmission.trans_lz1.isin(analysis_zones)) | (transmission.trans_lz2.isin(analysis_zones))]

    TxFixedCosts = transmission.pivot_table(index=['scenario','investment_period'], values="TxFixedCosts", aggfunc=np.sum )
    TxFixedCosts.rename(columns={'TxFixedCosts' : 'AnnualCost_Real'}, inplace=True)
    TxFixedCosts.reset_index(inplace=True)
    TxFixedCosts = pd.merge(TxFixedCosts, period_info[['scenario', 'investment_period', 'bring_annual_costs_to_base_year']], on=['scenario', 'investment_period'])
    TxFixedCosts ['AnnualCost_NPV'] = TxFixedCosts ['AnnualCost_Real'] * TxFixedCosts ['bring_annual_costs_to_base_year']
    TxFixedCosts.drop(['bring_annual_costs_to_base_year'], axis=1, inplace=True)
    TxFixedCosts['Component'] = 'TxFixedCosts'
    TxFixedCosts['Component_type'] = 'annual'
    TxFixedCosts = TxFixedCosts[['scenario', 'investment_period', 'Component', 'Component_type', 'AnnualCost_NPV', 'AnnualCost_Real']]

    return TxFixedCosts

def fuel_costs(scenarios_file_name: str, 
               analysis_scenarios: list, 
               analysis_zones: list):

    # Function that returns the fuel costs of the scenarios under analysis.
    #
    # Output: Dataframe with structure:
    #   scenario	investment_period	Component	        Component_type	AnnualCost_NPV	    AnnualCost_Real
    #   ------------------------------------------------------------------------------------------------
    #   high_100	2050	            FuelCostsPerPeriod	annual	        1,581,075,880.34	802,672,156.64

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name) 
    period_info = timestamp_info(scenarios_file_name, analysis_scenarios)[0]

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]     

    financials = get_data(scenario, "financials.csv", fpath='inputs')
    financials  = financials  .replace({"scenario": short_names})

    period_info = pd.merge(left=period_info, right=financials, on='scenario')
    period_info.loc[:, 'bring_annual_costs_to_base_year']=period_info.apply(lambda x: uniform_series_to_present_value(x['discount_rate'], x['period_length_years'])
                                                                        *future_to_present_value(x['discount_rate'], x['period_start'] - x['base_financial_year']), axis=1)
    period_info.rename(columns={'ts_period': 'investment_period'}, inplace=True)

    zone_to_rfm = get_data(scenario,"zone_to_regional_fuel_market.csv", fpath='inputs')
    zone_to_rfm  = zone_to_rfm .replace({"scenario": short_names})

    fuel_costs_tier_0 = get_data(scenario,"fuel_cost.csv", fpath='inputs')
    fuel_costs_tier_0['regional_fuel_market'] = fuel_costs_tier_0['load_zone'] + '_' + fuel_costs_tier_0['fuel']
    fuel_costs_tier_0['tier'] = 0
    fuel_costs_tier_0.rename(columns={'fuel_cost':'unit_cost'},inplace=True)
    fuel_costs_tier_0 = fuel_costs_tier_0[['regional_fuel_market','period','tier','unit_cost','scenario','load_zone']]
    fuel_costs_tier_0 = fuel_costs_tier_0 .replace({"scenario": short_names})

    fuel_costs_tiers_1_and_up = get_data(scenario,"fuel_supply_curves.csv", fpath='inputs')
    fuel_costs_tiers_1_and_up = fuel_costs_tiers_1_and_up .replace({"scenario": short_names})
    fuel_costs_tiers_1_and_up = fuel_costs_tiers_1_and_up[ ['regional_fuel_market', 'period', 'tier', 'unit_cost', 'scenario']]
    fuel_costs_tiers_1_and_up = pd.merge(left=fuel_costs_tiers_1_and_up,right=zone_to_rfm , on=['regional_fuel_market', 'scenario'])

    fuel_costs = pd.concat([fuel_costs_tier_0,fuel_costs_tiers_1_and_up],ignore_index=True)
    fuel_costs = fuel_costs .replace({"scenario": short_names})

    consumefuel = get_data(scenario, "ConsumeFuelTier.csv")
    consumefuel  = consumefuel .replace({"scenario": short_names})
    consumefuel.rename(columns={'RFM_SUPPLY_TIERS_1': 'regional_fuel_market', 'RFM_SUPPLY_TIERS_2': 'period', 'RFM_SUPPLY_TIERS_3':'tier'}, inplace='True')
    consumefuel = pd.merge(left=consumefuel,right=fuel_costs , on=['regional_fuel_market', 'period', 'tier' ,'scenario'])
    consumefuel ['FuelCostsPerPeriod'] = consumefuel ['ConsumeFuelTier'] * consumefuel ['unit_cost']
    consumefuel .rename(columns={'period':'investment_period'}, inplace=True)

    if len(analysis_zones)>0:
        consumefuel   = consumefuel.loc[consumefuel.load_zone.isin(analysis_zones)]

    FuelCostsPerPeriod = consumefuel.pivot_table(index=['scenario','investment_period'], values="FuelCostsPerPeriod", aggfunc=np.sum )
    FuelCostsPerPeriod.rename(columns={'FuelCostsPerPeriod' : 'AnnualCost_Real'}, inplace=True)
    FuelCostsPerPeriod.reset_index(inplace=True)
    FuelCostsPerPeriod = pd.merge(FuelCostsPerPeriod, period_info[['scenario', 'investment_period', 'bring_annual_costs_to_base_year']], on=['scenario', 'investment_period'])
    FuelCostsPerPeriod ['AnnualCost_NPV'] = FuelCostsPerPeriod ['AnnualCost_Real'] * FuelCostsPerPeriod ['bring_annual_costs_to_base_year']
    FuelCostsPerPeriod.drop(['bring_annual_costs_to_base_year'], axis=1, inplace=True)
    FuelCostsPerPeriod['Component'] = 'FuelCostsPerPeriod'
    FuelCostsPerPeriod['Component_type'] = 'annual'
    FuelCostsPerPeriod = FuelCostsPerPeriod[['scenario', 'investment_period', 'Component', 'Component_type', 'AnnualCost_NPV', 'AnnualCost_Real']]
    
    return FuelCostsPerPeriod

def total_NPV_cost_zone(scenarios_file_name: str, 
                            analysis_scenarios: list, 
                            analysis_zones: list,
                            analysis_tech=[],
                            consider_tx = 1):
    
    # Output: 

    #   scenario    tech_map  investment_period               Component Component_type  AnnualCost_NPV  AnnualCost_Real
    #   wecc_10_10     Biomass               2035      TotalGenFixedCosts         annual    1.428616e+00     6.859928e-01
    #   wecc_10_10        Coal               2035      TotalGenFixedCosts         annual    0.000000e+00     0.000000e+00
    #   wecc_10_10         Gas               2035      TotalGenFixedCosts         annual    0.000000e+00     0.000000e+00

    power_gen_fixed = power_generation_fixed_costs(scenarios_file_name, analysis_scenarios, analysis_zones, analysis_tech)
    power_gen_fixed['AnnualCost_NPV'] = power_gen_fixed['AnnualCost_NPV']*10**9
    power_gen_fixed['AnnualCost_Real'] = power_gen_fixed['AnnualCost_Real']*10**9

    storage_energy_fixed = storage_energy_fixed_costs(scenarios_file_name, analysis_scenarios, analysis_zones)
    gen_var = generation_variable_costs(scenarios_file_name, analysis_scenarios, analysis_zones)

    fuel = fuel_costs(scenarios_file_name, analysis_scenarios, analysis_zones)
    
    if consider_tx == 1:
        tx_costs = transmission_fixed_costs(scenarios_file_name, analysis_scenarios, analysis_zones)   
        total_cost = pd.concat([power_gen_fixed, storage_energy_fixed, gen_var, tx_costs, fuel])
    else:
        total_cost = pd.concat([power_gen_fixed, storage_energy_fixed, gen_var, fuel])

    return total_cost


def total_NPV_cost(scenarios_file_name: str, 
                   analysis_scenarios: list,
                   analysis_period: list, 
                   baseline_scenario):

    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name) 

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]     

    costs = get_data(scenario, "costs_itemized.csv")
    costs = costs  .replace({"scenario": short_names})
    costs = costs.loc[costs.PERIOD.isin(analysis_period)]
    costs_by_sc = costs.pivot_table(index='scenario', values='AnnualCost_NPV', columns='Component')
    col_list= list(costs_by_sc)
    costs_by_sc['Total_NPV'] = costs_by_sc[col_list].sum(axis=1)
    costs_by_sc.drop(columns=['EmissionsCosts', 'FuelCostsPerPeriod', 'GenVariableOMCostsInTP', 'StorageEnergyFixedCost', 'TotalGenFixedCosts', 'TxFixedCosts'], inplace=True)
    costs_by_sc.reset_index(inplace=True)
    costs_by_sc['base']=costs_by_sc.loc[costs_by_sc.scenario==baseline_scenario,'Total_NPV'].iloc[0]
    costs_by_sc['relative']=costs_by_sc['Total_NPV']/costs_by_sc['base']
    costs_by_sc['sc_order'] = costs_by_sc['scenario'].map(order)
    costs_by_sc = costs_by_sc.sort_values('sc_order').drop('sc_order',axis=1)
    costs_by_sc.reset_index(inplace=True, drop=True)
    costs_by_sc= costs_by_sc.rename_axis(None, axis=1)
    
    return costs_by_sc

def NPV_costs_itemized(scenarios_file_name: str, 
                       analysis_scenarios: list,
                       analysis_period: list):
    scenario, short_names, inv_short_names, order = read_scenarios(scenarios_file_name) 

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]     
        
    costs = get_data(scenario, "costs_itemized.csv")
    costs = costs  .replace({"scenario": short_names})
    costs = costs.loc[costs.PERIOD.isin(analysis_period)]
    costs_by_type = costs.pivot_table(index="scenario", columns="Component", values="AnnualCost_NPV", aggfunc=np.sum)
    costs_by_type.reindex(columns=["TotalGenFixedCosts", "StorageEnergyFixedCost", "FuelCostsPerPeriod", "TxFixedCosts",
                          "EmissionsCosts", "GenVariableOMCostsInTP"])
    costs_by_type.loc[:,'Total']=costs_by_type.apply(lambda x: sum(x[c] for c in costs_by_type.columns), axis=1)

    costs_by_type['sc_order'] = costs_by_type.index.map(order)
    costs_by_type=costs_by_type.sort_values('sc_order').drop('sc_order',axis=1)
    costs_by_type = costs_by_type /10**9 # Put the numbers in USD billion 

    costs_by_type_in_percentage = costs_by_type.div(costs_by_type.Total, axis=0)

    return costs_by_type,  costs_by_type_in_percentage


# -------------------------------------------------------------------------              -------------------------------------------------------
#-------------------------------------------------------------------------- TRANSMISSION -------------------------------------------------------
# -------------------------------------------------------------------------              -------------------------------------------------------

def annual_tx_loadability(scenario_file_name: str, 
                          analysis_scenarios: list, 
                          analysis_zones: list, 
                          geodf_wecc: gpd.GeoDataFrame, 
                          df_intervals: pd.DataFrame):

    scenario, short_names, inv_short_names, order = read_scenarios(scenario_file_name)

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]   

    flows = get_data(scenario, "transmission_dispatch.csv")
    flows.replace(short_names, inplace=True)
    
    if len(analysis_zones)!=0:
        analysis_zones = flows.loc[((flows.load_zone_from.isin(analysis_zones)) & (flows.load_zone_to.isin(analysis_zones)))]

    flows["cargability"] =  flows["transmission_dispatch"]/flows["dispatch_limit"]*100
    
    #Get the average of cargability over the timestamp of the total number of periods (usually one year). You can change "average"
    flows_cargability = flows.pivot_table(index=['load_zone_from', 'load_zone_to', 'scenario'], values='cargability', aggfunc=np.average)
    flows_cargability.reset_index(inplace=True)

    if len(geodf_wecc)>0:
        #Attach the coordinates to the "load_zone_from", and "load_zone_to". This help for the creating of lines in maps
        flows_cargability = pd.merge(left=flows_cargability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_from'}), on='load_zone_from')
        flows_cargability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_from'}, inplace=True)
        flows_cargability = pd.merge(left=flows_cargability, right=geodf_wecc[['centroid_partial_can']].reset_index().rename(columns={'load_zone' : 'load_zone_to'}), on='load_zone_to')
        flows_cargability.rename(columns={'centroid_partial_can': 'coordinate_load_zone_to'}, inplace=True)

    if len(df_intervals)>0:
        #Give linewidth and color to each row according to the cargability. It needs the function return_interval
        flows_cargability.loc[:, ['linewidth', 'color']] = flows_cargability.apply(lambda x: return_interval(x['cargability'], df_intervals), axis=1)[['linewidth','color']]

    #In case if it is necessary, provide default direction.
    default_direction = get_data(scenario, "transmission_lines.csv", fpath='inputs', usecols=['trans_lz1', 'trans_lz2'])
    default_direction.rename(columns={'trans_lz1':'load_zone_from', 'trans_lz2':'load_zone_to'}, inplace=True)
    default_direction.replace(short_names, inplace=True)
    default_direction['direction'] = 1
    flows_cargability = pd.merge(left=flows_cargability, right=default_direction, on=['load_zone_from', 'load_zone_to', 'scenario'], how='left')

    #Give order (though it is not necessary)
    flows_cargability['sc_order'] = flows_cargability['scenario'].map(order)
    flows_cargability = flows_cargability.sort_values('sc_order').drop('sc_order',axis=1)

    return flows_cargability

def average_tx_loading(scenario_file_name: str, 
                          analysis_scenarios: list, 
                          analysis_zones: list):

    scenario, short_names, inv_short_names, order = read_scenarios(scenario_file_name)

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]   

    flows = get_data(scenario, "transmission_dispatch.csv")
    flows.replace(short_names, inplace=True)
    
    if len(analysis_zones)!=0:
        analysis_zones = flows.loc[((flows.load_zone_from.isin(analysis_zones)) & (flows.load_zone_to.isin(analysis_zones)))]

    flows["loading"] =  flows["transmission_dispatch"]/flows["dispatch_limit"]*100
    
    #Get the average of cargability over the timestamp of the total number of periods (usually one year). You can change "average"
    flows_cargability = flows.pivot_table(index=['load_zone_from', 'load_zone_to', 'scenario', 'dispatch_limit'], values='loading', aggfunc=np.average)
    flows_cargability.reset_index(inplace=True)

    flows_cargability['cap_t_loading'] = flows_cargability['dispatch_limit'] * flows_cargability['loading']

    total_flows = flows_cargability.pivot_table(index = 'scenario', values = 'cap_t_loading', aggfunc =np.sum)

    total_capacity = flows.pivot_table(index='scenario', values='dispatch_limit', aggfunc=np.sum)
    total_capacity.reset_index(inplace=True)    

    #Give order (though it is not necessary)
    flows_cargability['sc_order'] = flows_cargability['scenario'].map(order)
    flows_cargability = flows_cargability.sort_values('sc_order').drop('sc_order',axis=1)

    return flows_cargability


def generate_new_txs(scenario_file_name: str, 
                     analysis_scenarios: list, 
                     special_zones: pd.DataFrame):
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenario_file_name)

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]  

    tx_lines = get_data(scenario, "transmission_lines.csv", fpath='inputs', usecols=['trans_lz1', 'trans_lz2'])
    tx_lines.replace(short_names, inplace=True)

    new_txs = pd.merge(left = tx_lines, right = special_zones.rename(columns={'load_zone':'trans_lz1'}), on='trans_lz1')
    new_txs = new_txs.merge(special_zones.rename(columns={'load_zone':'trans_lz2'}), on='trans_lz2')
    new_txs.rename(columns={'state_x': 'trans_st1', 'state_y': 'trans_st2'}, inplace=True)
    new_txs = new_txs[['trans_st1', 'trans_st2', 'trans_lz1', 'trans_lz2', 'scenario']]

    new_txs['sc_order'] = new_txs['scenario'].map(order)
    new_txs= new_txs.sort_values('sc_order').drop('sc_order',axis=1)

    new_txs = new_txs[new_txs['trans_st1'] != new_txs['trans_st2']] #Eliminate internal lines, after this line, you may print the df if you want to see the lines between two special loadzones

    new_txs = new_txs[['trans_st1', 'trans_st2', 'scenario']].rename(columns={'trans_st1': 'trans_lz1', 'trans_st2': 'trans_lz2'}) #We reformat the dataframe to have it in transmission_lines.csv format

    return new_txs

def generate_new_tx_dispatch(scenario_file_name: str, analysis_scenarios: list, special_zones: pd.DataFrame):

    scenario, short_names, inv_short_names, order = read_scenarios(scenario_file_name)

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]  

    tx_dispatch = get_data(scenario, "transmission_dispatch.csv", usecols=['load_zone_from', 'load_zone_to', 'timestamp', 'transmission_dispatch', 'dispatch_limit'])
    tx_dispatch.replace(short_names, inplace=True)

    new_txdis = pd.merge(left = tx_dispatch, right = special_zones.rename(columns={'load_zone':'load_zone_from'}), on='load_zone_from')
    new_txdis = new_txdis.merge(special_zones.rename(columns={'load_zone':'load_zone_to'}), on='load_zone_to')
    new_txdis.drop(['load_zone_from', 'load_zone_to'], axis=1, inplace=True)
    new_txdis.rename(columns={'state_x': 'load_zone_from', 'state_y': 'load_zone_to'}, inplace=True)
    
    new_txdis = new_txdis[new_txdis['load_zone_from'] != new_txdis['load_zone_to']]
    
    new_txdis = new_txdis.pivot_table(index=['load_zone_from', 'load_zone_to', 'timestamp', 'scenario'], values=['transmission_dispatch', 'dispatch_limit'], aggfunc=np.sum)
    new_txdis.reset_index(inplace=True)
    
    new_txdis['sc_order'] = new_txdis['scenario'].map(order)
    new_txdis= new_txdis.sort_values('sc_order').drop('sc_order',axis=1)
    
    return new_txdis

def monthly_tx_losses(scenario_file_name: str, 
                      analysis_scenarios: list,
                      analysis_zones: list, 
                      time_zone: str):
    
    # Dataframe that provides the transmission losses.
    # Output:
    #   month	scenario	    zone_demand_gwh	losses	    load_plus_losses_twh
    #   1	    wecc_10_10	    85,865.70	    1,416.05	87.28
    #	1	    wecc_140_170	85,865.70	    1,545.16	87.41
    #	2	    wecc_10_10	    78,827.29	    1,374.09	80.20
    #	2	    wecc_140_170	78,827.29	    1,400.50	80.23
    
    scenario, short_names, inv_short_names, order = read_scenarios(scenario_file_name)

    if len(analysis_scenarios)>0:
        scenario = [inv_short_names[i] for i in analysis_scenarios]  

    tx_dispatch = get_data(scenario, "transmission_dispatch.csv", usecols=['load_zone_from', 'load_zone_to', 'timestamp', 'transmission_dispatch'])
    tx_dispatch  = tx_dispatch  .replace({"scenario": short_names})

    time_info = timestamp_info(scenario_file_name, analysis_scenarios)[1]
    tx_dispatch = pd.merge(left=tx_dispatch, right=time_info[['scenario', 'timestamp', 'tp_weight_in_year']], on=['scenario', 'timestamp'])
    tx_dispatch['flow_gwh']=tx_dispatch['transmission_dispatch']*tx_dispatch['tp_weight_in_year']/10**3

    txs = get_data(scenario, "transmission_lines.csv", fpath='inputs',  usecols=['trans_lz1','trans_lz2','trans_efficiency'])
    txs.replace({"scenario": short_names}, inplace=True)
    txs_copy=txs.copy() #Duplicate the table to have NM in the sending zone and receiving zone
    txs_copy.rename(columns={'trans_lz1':'trans_lz2', 'trans_lz2':'trans_lz1'}, inplace=True)
    txs=pd.concat([txs,txs_copy])
    txs.reset_index(inplace=True, drop=True)
    txs.rename(columns={'trans_lz1': 'load_zone_from', 'trans_lz2': 'load_zone_to'}, inplace=True)

    tx_dispatch = pd.merge(left= tx_dispatch, right= txs, on=['scenario', 'load_zone_from', 'load_zone_to'], how='left')

    if len(analysis_zones)!= 0:
        tx_dispatch = tx_dispatch.loc[tx_dispatch.load_zone_from.isin(analysis_zones) & tx_dispatch.load_zone_to.isin(analysis_zones)]
    
    tx_dispatch['losses']=tx_dispatch['flow_gwh']*(1-tx_dispatch['trans_efficiency'])

    tx_dispatch["timestamp"]=pd.to_datetime(tx_dispatch["timestamp"], format='%Y%m%d%H', utc=True)
    tx_dispatch["timestamp"]=tx_dispatch["timestamp"].dt.tz_convert(time_zone)

    tx_dispatch['month']=tx_dispatch['timestamp'].dt.month

    monthly_tx_losses_df =  tx_dispatch.pivot_table(index=['month', 'scenario'], values='losses', aggfunc=np.sum)
    monthly_tx_losses_df.reset_index(inplace=True)

    return monthly_tx_losses_df


# -----------------------------------------------------------------------        -------------------------------------------------------------
# ----------------------------------------------------------------------- EXTRAS -------------------------------------------------------------
# -----------------------------------------------------------------------        -------------------------------------------------------------

PLOT_PARAMS = {
    "font.size": 7,
    "font.family": "Source Sans Pro",
    "legend.fontsize": 6,
    "legend.handlelength": 2,
    "figure.dpi": 120,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.5,
    "lines.linewidth": 1.5,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.major.width": 0.6,
    "ytick.minor.width": 0.4,
    "ytick.minor.size": 2.5,
    "ytick.major.size": 5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "grid.linewidth": 0.1,
    "savefig.dpi":300,
    "legend.frameon": False,
    "legend.framealpha": 0.8,
    #"legend.edgecolor": 0.9,
    "legend.borderpad": 0.2,
    "legend.columnspacing": 1.5,
    "legend.labelspacing":  0.4,
}
