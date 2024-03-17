# %%
# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scienceplots
import seaborn as sns
import importlib
import utils
from utils import * 
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pypdf import PdfFileReader, PageObject
import matplotlib.dates as mdates

plt.style.use(['science','ieee'])
pd.options.display.float_format = '{:,.2f}'.format
plt.style.use(['science','ieee'])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 6.7

importlib.reload(utils)

# %%
# ---------------------------------------              ------------------------------------------------------------
# --------------------------------------- INITIAL DATA ------------------------------------------------------------
# ---------------------------------------              ------------------------------------------------------------

analysis_scenarios, reference_storage_project = [], [1191209729]
wecc_storage_costs = get_storage_cost_by_scenario("wecc_scenarios.csv", analysis_scenarios, reference_storage_project)
ca_is_storage_costs = get_storage_cost_by_scenario("ca_scenarios.csv" ,analysis_scenarios, reference_storage_project)

zones=[]
periods = [2035]
start_date="2035 06 01 00"
end_date="2035 06 30 00"
time_zone = 'US/Pacific'
consider_exp_imp = 0

selected_scenarios = ["wecc_10_10", "wecc_140_170"]
ca_zones = ['CA_IID', 'CA_LADWP', 'CA_PGE_BAY', 'CA_PGE_CEN', 'CA_PGE_N', 'CA_PGE_S', 'CA_SCE_CEN', 'CA_SCE_S','CA_SCE_SE','CA_SCE_VLY','CA_SDGE', 'CA_SMUD']


# %%
# --------------------------------------               -----------------------------------------------------------------
# -------------------------------------- LOAD DISPATCH -----------------------------------------------------------------
# --------------------------------------               -----------------------------------------------------------------

s1 = stacked_dispatch('wecc_scenarios.csv', [selected_scenarios[0]], zones, time_zone, start_date, end_date, periods, consider_exp_imp) 

s2 = stacked_dispatch('wecc_scenarios.csv', [selected_scenarios[1]], zones, time_zone, start_date, end_date, periods, consider_exp_imp) 

# %%
# --------------------------------------                            --------------------------------------------------------------
# -------------------------------------- DAILY AND MONTHLY DISPATCH --------------------------------------------------------------
# --------------------------------------                            -------------------------------------------------------------

s1m = s1[1]/1000 #Energy (TWh)
s1d = s1[0]

s2m = s2[1]/1000 #Energy (TWh)
s2d = s2[0]


# %%

s1_relative = s1.copy()
s1_relative = pd.concat([s1_relative, s1_relative.sum(axis=1)], axis=1).rename(columns={0: 'Total'})
for j in s1_relative.columns:
    s1_relative[j]=s1_relative[j]/s1_relative['Total']*100
s1_relative

s2_relative = s2.copy()
s2_relative = pd.concat([s2_relative, s2_relative.sum(axis=1)], axis=1).rename(columns={0: 'Total'})
for j in s2_relative.columns:
    s2_relative[j]=s2_relative[j]/s2_relative['Total']*100
s2_relative

# %%
# ---------------------------------------                 -----------------------------------------------------------------------------------
# --------------------------------------- LOAD AND LOSSES -----------------------------------------------------------------------------------
# ---------------------------------------                 -----------------------------------------------------------------------------------

monthly_load = monthly_demand('wecc_scenarios.csv', selected_scenarios, zones, time_zone)
monthly_losses = monthly_tx_losses('wecc_scenarios.csv', selected_scenarios, zones, time_zone)
monthly_load = pd.merge(left=monthly_load, right=monthly_losses, on=['month', 'scenario'])
monthly_load['load_plus_losses_twh'] = (monthly_load['zone_demand_gwh'] +  monthly_load['losses'])/10**3 
monthly_load

# %%
# --------------------------------------                   ----------------------------------------------------------------------------------
# --------------------------------------- MONTHLY DISPATCH ----------------------------------------------------------------------------------
# ---------------------------------------                  ----------------------------------------------------------------------------------

figure_name = "Monthly dispatch in the wecc.pdf"

fig, ((ax11, ax12)) = plt.subplots(nrows=2, ncols=1, figsize=(3.0,6.0))
dic={1 : 'Jan', 2: 'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

ax11 = s1m[tech_order].plot.area(stacked=True,ax=ax11, color=tech_colors, rot=0, lw=0.1,legend=None)
ax11 = monthly_load.loc[monthly_load.scenario.isin([selected_scenarios[0]])].plot.line(x='month', y='load_plus_losses_twh', ax=ax11, rot=0, lw=1, color='red', linestyle='dashed')

ax11.set_ylim(-40,140)
ax11.set_xlim(s1m.index[0], s1m.index[-1])
ax11.set_title('(a)')
ax11.set_xlabel(' ')
ax11.set_ylabel('Energy (TWh)')
ax11.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
ax11.set_xticklabels(dic.values(), rotation=0)
ax11.xaxis.set_tick_params(which='minor', bottom=False, top=False)
ax11.get_legend().remove()

ax12 = s2m[tech_order].plot.area(stacked=True,ax=ax12, color=tech_colors, rot=0, lw=0.1,legend=None)
ax11 =monthly_load.loc[monthly_load.scenario.isin([selected_scenarios[1]])].plot.line(x='month', y='load_plus_losses_twh',ax=ax12, rot=0, lw=1, color='red', linestyle='dashed')
ax12.set_ylim(-40,140)
ax12.set_xlim(s2m.index[0], s2m.index[-1])
ax12.set_title('(b)')
ax12.set_xlabel('Month')
ax12.set_ylabel('Energy (TWh)')
ax12.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
ax12.set_xticklabels(dic.values(), rotation=0)
ax12.xaxis.set_tick_params(which='minor', bottom=False, top=False)
ax12.get_legend().remove()

plt.subplots_adjust(wspace=0, hspace=0.16)

#Legend of the generation portfolio
handles_gen_tech = [mpatches.Patch( color=tech_colors[k], label=k)  for k in tech_colors.keys()]

fig.legend(handles = handles_gen_tech, title='Generation',   bbox_to_anchor=(0.90,0.06), ncol=3,  handlelength=1, alignment='left')

handles_demand = [Line2D([0], [0], label= 'Load + transmission losses', alpha=1,  lw=2, color='red', linestyle='dashed')]

fig.legend(handles = handles_demand, title=' ', bbox_to_anchor=(0.65,-0.03), ncol=1, handlelength=1.5, handletextpad=1, alignment='left')


folder_to_save_results = "figures/"
plt.savefig(folder_to_save_results+figure_name, facecolor='White', transparent=False)
plt.savefig(folder_to_save_results+'temporal.pdf', facecolor='White', transparent=False)


# %%
# -------------------------------------------                --------------------------------------------------------------
# ------------------------------------------- DAILY DISPATCH --------------------------------------------------------------
# -------------------------------------------                --------------------------------------------------------------

figure_name = "Daily dispatch in the wecc.pdf"

fig, ((ax11, ax12)) = plt.subplots(nrows=2, ncols=1, figsize=(3.0,6.0))

ax11 = s1d[tech_order].plot.area(stacked=True,ax=ax11, color=tech_colors, rot=0, lw=0.1,legend=None)

ax11.set_ylim(-400,1000)
ax11.set_xlim(s1d.index[0], s1d.index[-1])
ax11.set_title('(a)')
ax11.set_xlabel(' ')
ax11.set_ylabel('Energy (GWh)')
ax11.set_xticks([0, 4, 8, 12, 16, 20])
#ax11.set_xticklabels(dic.values(), rotation=0)
ax11.xaxis.set_tick_params(which='minor', bottom=False, top=False)
#ax11.get_legend().remove()

ax11.xaxis.grid(True, linestyle='dashed')

ax12 = s2d[tech_order].plot.area(stacked=True,ax=ax12, color=tech_colors, rot=0, lw=0.1,legend=None)
#ax11 =monthly_load.loc[monthly_load.scenario.isin([selected_scenarios[1]])].plot.line(x='month', y='load_plus_losses_twh',ax=ax12, rot=0, lw=1, color='red', linestyle='dashed')
ax12.set_ylim(-400,1000)
ax12.set_xlim(s2d.index[0], s2d.index[-1])
ax12.set_title('(b)')
ax12.set_xlabel('Hour (h)')
ax12.set_ylabel('Energy (GWh)')
ax12.set_xticks([0, 4, 8, 12, 16, 20])
#ax12.set_xticklabels(dic.values(), rotation=0)
ax12.xaxis.set_tick_params(which='minor', bottom=False, top=False)
#ax12.get_legend().remove()
ax12.xaxis.grid(True, linestyle='dashed')

plt.subplots_adjust(wspace=0, hspace=0.16)

#Legend of the generation portfolio
handles_gen_tech = [mpatches.Patch( color=tech_colors[k], label=k)  for k in tech_colors.keys()]

fig.legend(handles = handles_gen_tech, title='Generation',   bbox_to_anchor=(0.90,0.06), ncol=3,  handlelength=1, alignment='left')

handles_demand = [Line2D([0], [0], label= 'Load + transmission losses', alpha=1,  lw=2, color='red', linestyle='dashed')]

fig.legend(handles = handles_demand, title=' ', bbox_to_anchor=(0.65,-0.03), ncol=1, handlelength=1.5, handletextpad=1, alignment='left')

folder_to_save_results = "figures/"
plt.savefig(folder_to_save_results+figure_name, facecolor='White', transparent=False)
plt.savefig(folder_to_save_results+'temporal.pdf', facecolor='White', transparent=False)


# %%


s1m.to_csv('monthly_dispatch_a.csv')

s2m.to_csv('monthly_dispatch_b.csv')

s1d.to_csv('daily_dispatch_a.csv')

s2d.to_csv('daily_dispatch_b.csv')


# %%

# %%

# -------------------------------------------                --------------------------------------------------------------
# ------------------------------------------- SOC            --------------------------------------------------------------
# -------------------------------------------                --------------------------------------------------------------

selected_scenarios = ['wecc_140_170', 'wecc_10_10']

zones_states = pd.read_csv('zones_states.csv')
rockies = list(zones_states.loc[zones_states.region == 'Rockies','load_zone'])
pacificnorthwest =  list(zones_states.loc[zones_states.region == 'Pacific Northwest','load_zone'])
california = list(zones_states.loc[zones_states.region == 'California','load_zone'])
southwest =  list(zones_states.loc[zones_states.region == 'Southwest','load_zone'])
mexico = ['MEX_BAJA']
canada = ['CAN_ALB', 'CAN_BC']

tech = []
time_zone = 'US/Pacific'

zones = california

get_timestamp_of_peak_demand_in_zone('wecc_scenarios.csv', selected_scenarios, zones)
get_timestamp_of_lowest_xch_ratio_of_zones('wecc_scenarios.csv', selected_scenarios, zones)

load_c= load_timeseries('wecc_scenarios.csv',selected_scenarios, zones)
load_c['zone_demand_gw'] = load_c['zone_demand_mw']/1000

flow_c = poweflow_timeseries('wecc_scenarios.csv',selected_scenarios, zones)
flow_c['Imports'] =  flow_c['Imports']/1000
flow_c['Exports'] =  flow_c['Exports']/1000

stg_c = storage_timeseries('wecc_scenarios.csv',selected_scenarios, zones)
stg_c['StateOfChargeGWh'] = stg_c['StateOfCharge']/1000
stg_c['Storage(C)'] = -stg_c['ChargeMW']/1000
stg_c['Storage(D)'] = stg_c['DischargeMW']/1000

dpch_c = dispatch_timeseries('wecc_scenarios.csv',selected_scenarios, zones, tech)
dpch_c['DispatchGen_GW'] = dpch_c['DispatchGen_MW']/1000
dpch_c['Curtailment'] = dpch_c['Curtailment_MW']/1000
curtailment_c = (dpch_c.loc[dpch_c.tech_map.isin(['Solar', 'Wind'])]).pivot_table(index = ['scenario', 'timestamp'], values = 'Curtailment', aggfunc=np.sum).reset_index()

dpch_tech_c =dpch_c.pivot_table(index = ['scenario', 'timestamp'], columns='tech_map', values = 'DispatchGen_GW', aggfunc=np.sum)
dpch_tech_c.reset_index(inplace=True)
dpch_tech_c.drop(['Storage'], axis=1, inplace=True)
dpch_tech_c =  pd.merge(left = dpch_tech_c, right=stg_c[['scenario', 'timestamp', 'Storage(C)', 'Storage(D)']], on=['scenario', 'timestamp'])
dpch_tech_c =  pd.merge(left = dpch_tech_c, right=flow_c[['scenario', 'timestamp', 'Imports', 'Exports']], on=['scenario', 'timestamp'])
dpch_tech_c =  pd.merge(left = dpch_tech_c, right=curtailment_c[['scenario', 'timestamp', 'Curtailment']], on = ['scenario', 'timestamp'])

# %%
# -------------------------------------------- CALIFORNIA INTERACTIVE PANEL -------------------------------------------------------------

# import graph_objects from plotly package
import plotly.graph_objects as go
 
# import make_subplots function from plotly.subplots
# to make grid of plots
from plotly.subplots import make_subplots

# Dataframes
load = load_c.copy()
stg = stg_c.copy()
dpch_tech = dpch_tech_c.copy()

# Enter dates
def filter_dates(df):
    start_date="2035 01 15 00"
    end_date="2035 12 30 00"
    time_zone='US/Pacific'
    time_1=datetime.strptime(str(start_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    time_2=datetime.strptime(str(end_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    return    df[(time_1 <= df.timestamp) & (df.timestamp < time_2)]  

#Dictionaries 
tech_order_1 = tech_order + ['Storage(C)', 'Storage(D)', 'Imports', 'Exports', 'Curtailment']
tech_order_1.remove('Storage')
tech_colors_1 = tech_colors.copy()
tech_colors_1['Storage(C)'] = "#7fffd4"
tech_colors_1['Storage(D)'] = "#7fffd4"
tech_colors_1['Imports'] = 'lightpink'
tech_colors_1['Exports'] = 'magenta'
tech_colors_1['Curtailment'] = 'Olive'

plot = go.Figure() 

plot = make_subplots(specs=[[{"secondary_y": True}]])

load0 =  load.loc[load.scenario == selected_scenarios[1]].pipe(filter_dates)
stg0 = stg.loc[stg.scenario == selected_scenarios[1]].pipe(filter_dates)
dpch_tech0 = dpch_tech.loc[dpch_tech.scenario == selected_scenarios[1]].pipe(filter_dates)
dpch_tech0.drop(['scenario'], axis=1, inplace=True)
dpch_tech0.set_index('timestamp', inplace=True)


for j in tech_order_1:
    col_name = j
    plot.add_trace(go.Scatter(x = list(dpch_tech0.index), 
                   y=  list(dpch_tech0[j]),
                   mode='lines',
                   line = dict(width = 0.5, color = tech_colors_1[j]),
                   name = col_name, 
                   stackgroup= 'positive' if j not in ['Storage(C)', 'Exports'] else 'negative')
                   )

plot.add_trace(go.Scatter(x = list(load0.timestamp), 
                   y=  load0.zone_demand_gw,
                   mode='lines',
                   line = dict(width = 1.5, color = 'red'),
                   name = 'Load', 
                   ))

plot.add_trace(go.Scatter(  x=list(stg0.timestamp),
                            y=stg0.StateOfChargeGWh,
                            line = dict(width = 1.5, color = 'black'),
                            name="State of Charge",
                            yaxis="y2"
))

plot.update_layout( xaxis=dict( rangeselector=dict( 
                buttons=list([ 
                    dict(count=1,
                     label="1 day",
                     step="day",
                     stepmode="backward"),
                dict(count=7,
                     label="1 week",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="1 month",
                     step="month",
                     stepmode="backward"),

            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
plot.update_traces(mode="lines", hovertemplate='%{y:.2f}')
plot.update_layout(hovermode = 'x unified', spikedistance = -1)

plot.update_layout(plot_bgcolor='white')
plot.update_xaxes(ticks='inside', mirror = True, showline=True, linecolor='black', gridcolor='lightgrey')
plot.update_yaxes(minor_ticks="inside", ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
plot.update_layout(yaxis=dict(title="Generation and Load (GW)"),yaxis2=dict(title="State of charge (GWh)"))
plot.update_layout(font_family="Times New Roman")
#plot.write_html("file.html")

plot.show()
# %%

# -------------------------------------------- CALIFORNIA DISPATCH -------------------------------------------------------------

# Dataframes
load = load_c.copy()
stg = stg_c.copy()
dpch_tech = dpch_tech_c.copy()

# Enter dates
def filter_dates(df):
    start_date="2035 05 15 00"
    end_date="2035 05 30 00"
    time_zone='US/Pacific'
    time_1=datetime.strptime(str(start_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    time_2=datetime.strptime(str(end_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    return    df[(time_1 <= df.timestamp) & (df.timestamp < time_2)]  

#Dictionaries 
tech_order_1 = tech_order + ['Storage(C)', 'Storage(D)', 'Imports', 'Exports', 'Curtailment']
tech_order_1.remove('Storage')
tech_colors_1 = tech_colors.copy()
tech_colors_1['Storage(C)'] = "#7fffd4"
tech_colors_1['Storage(D)'] = "#7fffd4"
tech_colors_1['Imports'] = 'lightpink'
tech_colors_1['Exports'] = 'magenta'
tech_colors_1['Curtailment'] = 'Olive'

# Load, state of charge and dispatch (scenario 0)
load0 =  load.loc[load.scenario == selected_scenarios[0]].pipe(filter_dates)
stg0 = stg.loc[stg.scenario == selected_scenarios[0]].pipe(filter_dates)
dpch_tech0 = dpch_tech.loc[dpch_tech.scenario == selected_scenarios[0]].pipe(filter_dates)
dpch_tech0.drop(['scenario'], axis=1, inplace=True)
dpch_tech0.set_index('timestamp', inplace=True)

# Load, state of charge and dispatch (scenario 1)
load1 =  load.loc[load.scenario == selected_scenarios[1]].pipe(filter_dates)
stg1 = stg.loc[stg.scenario == selected_scenarios[1]].pipe(filter_dates)
dpch_tech1 = dpch_tech.loc[dpch_tech.scenario == selected_scenarios[1]].pipe(filter_dates)
dpch_tech1.drop(['scenario'], axis=1, inplace=True)
dpch_tech1.set_index('timestamp', inplace=True)

# Figures
fig, ((ax0, ax1)) = plt.subplots(nrows=2, ncols=1, figsize=(6.0,3.0))

ax0 = dpch_tech0[tech_order_1].plot.area(stacked=True,ax=ax0, color=tech_colors_1, rot=0, lw=0.1,legend=None)
ax0 = load0.plot(ax = ax0, x ='timestamp', y ='zone_demand_gw', color = 'red', style ='-')
ax01 = ax0.twinx() 
ax01 = stg0.plot(ax = ax01,  x ='timestamp', y ='StateOfChargeGWh', color = 'black')

ax0.set(yticks = [-40, -20, 0, 20, 40, 60, 80, 100, 120])
ax01.set(yticks = [0, 100, 200, 300, 400, 500])
ax0.set_title('')
ax0.set_xlabel(' ')
ax0.get_legend().remove()
ax01.get_legend().remove()


ax1 = dpch_tech1[tech_order_1].plot.area(stacked=True,ax=ax1, color=tech_colors_1, rot=0, lw=0.1,legend=None)
ax1 = load1.plot(ax = ax1, x ='timestamp', y ='zone_demand_gw', color = 'red', style ='-')
ax11 = ax1.twinx() 
ax11 = stg1.plot(ax = ax11, x ='timestamp', y ='StateOfChargeGWh')

ax1.set(yticks = [-40, -20, 0, 20, 40, 60, 80, 100])
ax11.set(yticks = [0, 100, 200, 300, 400, 500])
ax1.set_title('')
ax1.set_xlabel('Time')
ax1.get_legend().remove()
ax11.get_legend().remove()

fig.text(0.07,0.3, "Generation and load (GW)", rotation=90)
fig.text(0.95,0.3, "State of charge (GWh)", rotation=90)
plt.subplots_adjust(wspace=0, hspace=0.13)


#Legend of the generation portfolio
tech_colors_1.pop('Storage(D)')
tech_colors_1.pop('Storage(C)')
handles_gen_tech = [mpatches.Patch( color=tech_colors_1[k], label=k)  for k in tech_colors_1.keys()]
fig.legend(handles = handles_gen_tech, title='',   bbox_to_anchor=(0.65,-0.05), ncol=4,  handlelength=1, alignment='left')

handles_demand = [Line2D([0], [0], label= 'Load', alpha=1,  lw=2, color='red', linestyle='dashed'),
                  Line2D([0], [0], label= 'State of Charge', alpha=1,  lw=2, color='black', linestyle='dashed')]
fig.legend(handles = handles_demand, title=' ', bbox_to_anchor=(0.90,-0.05), ncol=1, handlelength=1.5, handletextpad=1, alignment='left')

folder_to_save_results = "figures/"
#plt.savefig(folder_to_save_results+figure_name, facecolor='White', transparent=False)
plt.savefig(folder_to_save_results+'temporal.pdf', facecolor='White', transparent=False)

# %%

# -------------------------------------------- THE ROCKIES DISPATCH -------------------------------------------------------------

#Select scenarios to plot
selected_scenarios = ['wecc_140_170', 'wecc_10_10']

#Get regions
zones_states = pd.read_csv('zones_states.csv')
rockies = list(zones_states.loc[zones_states.region == 'Rockies','load_zone'])
pacificnorthwest =  list(zones_states.loc[zones_states.region == 'Pacific Northwest','load_zone'])
california = list(zones_states.loc[zones_states.region == 'California','load_zone'])
southwest =  list(zones_states.loc[zones_states.region == 'Southwest','load_zone'])
mexico = ['MEX_BAJA']
canada = ['CAN_ALB', 'CAN_BC']

# Get techs and zones
tech = []
zones = rockies + canada

# Get timestamps when peak demand happens
get_timestamp_of_peak_demand_in_zone('wecc_scenarios.csv', selected_scenarios, zones)

# Load
load_r = load_timeseries('wecc_scenarios.csv',selected_scenarios, zones)
load_r['zone_demand_gw'] = load_r['zone_demand_mw']/1000

# Flow
flow_r = poweflow_timeseries('wecc_scenarios.csv',selected_scenarios, zones)
flow_r['Imports'] =  flow_r['Imports']/1000
flow_r['Exports'] =  flow_r['Exports']/1000

# Storage
stg_r = storage_timeseries('wecc_scenarios.csv',selected_scenarios, zones)
stg_r['StateOfChargeGWh'] = stg_r['StateOfCharge']/1000
stg_r['Storage(C)'] = -stg_r['ChargeMW']/1000
stg_r['Storage(D)'] = stg_r['DischargeMW']/1000

# Dispatch
dpch_r = dispatch_timeseries('wecc_scenarios.csv',selected_scenarios, zones, tech)
dpch_r['DispatchGen_GW'] = dpch_r['DispatchGen_MW']/1000
dpch_r['Curtailment'] = dpch_r['Curtailment_MW']/1000
curtailment_r = (dpch_r.loc[dpch_r.tech_map.isin(['Solar', 'Wind'])]).pivot_table(index = ['scenario', 'timestamp'], values = 'Curtailment', aggfunc=np.sum).reset_index()

dpch_tech_r =dpch_r.pivot_table(index = ['scenario', 'timestamp'], columns='tech_map', values = 'DispatchGen_GW', aggfunc=np.sum)
dpch_tech_r.reset_index(inplace=True)
dpch_tech_r.drop(['Storage'], axis=1, inplace=True)
dpch_tech_r =  pd.merge(left = dpch_tech_r, right=stg_r[['scenario', 'timestamp', 'Storage(C)', 'Storage(D)']], on=['scenario', 'timestamp'])
dpch_tech_r =  pd.merge(left = dpch_tech_r, right=flow_r[['scenario', 'timestamp', 'Imports', 'Exports']], on=['scenario', 'timestamp'])
dpch_tech_r =  pd.merge(left = dpch_tech_r, right=curtailment_r[['scenario', 'timestamp', 'Curtailment']], on = ['scenario', 'timestamp'])



# %%

# -------------------------------------------- THE ROCKIES DISPATCH -------------------------------------------------------------


# Dataframes
load = load_r.copy()
stg = stg_r.copy()
dpch_tech = dpch_tech_r.copy()

# Enter dates
def filter_dates(df):
    start_date="2035 07 15 00"
    end_date="2035 07 28 00"
    time_zone='US/Pacific'
    time_1=datetime.strptime(str(start_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    time_2=datetime.strptime(str(end_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    return    df[(time_1 <= df.timestamp) & (df.timestamp < time_2)]  

#Dictionaries 
tech_order_1 = tech_order + ['Storage(C)', 'Storage(D)', 'Imports', 'Exports', 'Curtailment']
tech_order_1.remove('Storage')
tech_colors_1 = tech_colors.copy()
tech_colors_1['Storage(C)'] = "#7fffd4"
tech_colors_1['Storage(D)'] = "#7fffd4"
tech_colors_1['Imports'] = 'lightpink'
tech_colors_1['Exports'] = 'magenta'
tech_colors_1['Curtailment'] = 'Olive'

# Load, state of charge and dispatch (scenario 0)
load0 =  load.loc[load.scenario == selected_scenarios[0]].pipe(filter_dates)
stg0 = stg.loc[stg.scenario == selected_scenarios[0]].pipe(filter_dates)
dpch_tech0 = dpch_tech.loc[dpch_tech.scenario == selected_scenarios[0]].pipe(filter_dates)
dpch_tech0.drop(['scenario'], axis=1, inplace=True)
dpch_tech0.set_index('timestamp', inplace=True)

# Load, state of charge and dispatch (scenario 1)
load1 =  load.loc[load.scenario == selected_scenarios[1]].pipe(filter_dates)
stg1 = stg.loc[stg.scenario == selected_scenarios[1]].pipe(filter_dates)
dpch_tech1 = dpch_tech.loc[dpch_tech.scenario == selected_scenarios[1]].pipe(filter_dates)
dpch_tech1.drop(['scenario'], axis=1, inplace=True)
dpch_tech1.set_index('timestamp', inplace=True)

# Figures
fig, ((ax0, ax1)) = plt.subplots(nrows=2, ncols=1, figsize=(6.0,3.0))

ax0 = dpch_tech0[tech_order_1].plot.area(stacked=True,ax=ax0, color=tech_colors_1, rot=0, lw=0.1,legend=None)
ax0 = load0.plot(ax = ax0, x ='timestamp', y ='zone_demand_gw', color = 'red', style ='-')
ax01 = ax0.twinx() 
ax01 = stg0.plot(ax = ax01,  x ='timestamp', y ='StateOfChargeGWh', color = 'black')

ax0.set(yticks = [-40, -20, 0, 20, 40, 60, 80])
ax01.set(yticks = [0, 100, 200, 300, 400, 500])
ax0.set_title('')
ax0.set_xlabel(' ')
ax0.get_legend().remove()
ax01.get_legend().remove()


ax1 = dpch_tech1[tech_order_1].plot.area(stacked=True,ax=ax1, color=tech_colors_1, rot=0, lw=0.1,legend=None)
ax1 = load1.plot(ax = ax1, x ='timestamp', y ='zone_demand_gw', color = 'red', style ='-')
ax11 = ax1.twinx() 
ax11 = stg1.plot(ax = ax11, x ='timestamp', y ='StateOfChargeGWh')

ax1.set(yticks = [-40, -20, 0, 20, 40, 60, 80])
ax11.set(yticks = [0, 100, 200, 300, 400, 500])
ax1.set_title('')
ax1.set_xlabel('Time')
ax1.get_legend().remove()
ax11.get_legend().remove()

fig.text(0.07,0.3, "Generation and load (GW)", rotation=90)
fig.text(0.95,0.3, "State of charge (GWh)", rotation=90)
plt.subplots_adjust(wspace=0, hspace=0.13)


#Legend of the generation portfolio
tech_colors_1.pop('Storage(D)')
tech_colors_1.pop('Storage(C)')
handles_gen_tech = [mpatches.Patch( color=tech_colors_1[k], label=k)  for k in tech_colors_1.keys()]
fig.legend(handles = handles_gen_tech, title='Generation',   bbox_to_anchor=(0.65,-0.05), ncol=4,  handlelength=1, alignment='left')

handles_demand = [Line2D([0], [0], label= 'Load', alpha=1,  lw=2, color='red', linestyle='dashed'),
                  Line2D([0], [0], label= 'State of Charge', alpha=1,  lw=2, color='black', linestyle='dashed')]
fig.legend(handles = handles_demand, title=' ', bbox_to_anchor=(0.90,-0.05), ncol=1, handlelength=1.5, handletextpad=1, alignment='left')

folder_to_save_results = "figures/"
figure_name = 'Dispatch in the Rockies.pdf'
plt.savefig(folder_to_save_results+figure_name, facecolor='White', transparent=False)
plt.savefig(folder_to_save_results+'temporal.pdf', facecolor='White', transparent=False)


# %%

# -------------------------------------------- NEW MEXICO DISPATCH -------------------------------------------------------------

#Select scenarios to plot
selected_scenarios = ['wecc_140_170', 'wecc_10_10']

# Get techs and zones
tech = []
zones = ['NM_N']

# Get timestamps when peak demand happens
get_timestamp_of_peak_demand_in_zone('wecc_scenarios.csv', selected_scenarios, zones)

# Load
load_nm = load_timeseries('wecc_scenarios.csv',selected_scenarios, zones, time_zone='US/Mountain')
load_nm['zone_demand_gw'] = load_nm['zone_demand_mw']/1000

# Flow
flow_nm = poweflow_timeseries('wecc_scenarios.csv',selected_scenarios, zones, time_zone='US/Mountain')
flow_nm['Imports'] =  flow_nm['Imports']/1000
flow_nm['Exports'] =  flow_nm['Exports']/1000

# Storage
stg_nm = storage_timeseries('wecc_scenarios.csv',selected_scenarios, zones, time_zone='US/Mountain')
stg_nm['StateOfChargeGWh'] = stg_nm['StateOfCharge']/1000
stg_nm['Storage(C)'] = -stg_nm['ChargeMW']/1000
stg_nm['Storage(D)'] = stg_nm['DischargeMW']/1000

# Dispatch
dpch_nm = dispatch_timeseries('wecc_scenarios.csv',selected_scenarios, zones, tech, time_zone='US/Mountain')
dpch_nm['DispatchGen_GW'] = dpch_nm['DispatchGen_MW']/1000
dpch_nm['Curtailment'] = dpch_nm['Curtailment_MW']/1000
curtailment_nm = (dpch_nm.loc[dpch_nm.tech_map.isin(['Solar', 'Wind'])]).pivot_table(index = ['scenario', 'timestamp'], values = 'Curtailment', aggfunc=np.sum).reset_index()

dpch_tech_nm =dpch_nm.pivot_table(index = ['scenario', 'timestamp'], columns='tech_map', values = 'DispatchGen_GW', aggfunc=np.sum)
dpch_tech_nm.reset_index(inplace=True)
dpch_tech_nm.drop(['Storage'], axis=1, inplace=True)
dpch_tech_nm =  pd.merge(left = dpch_tech_nm, right=stg_nm[['scenario', 'timestamp', 'Storage(C)', 'Storage(D)']], on=['scenario', 'timestamp'])
dpch_tech_nm =  pd.merge(left = dpch_tech_nm, right=flow_nm[['scenario', 'timestamp', 'Imports', 'Exports']], on=['scenario', 'timestamp'])
dpch_tech_nm =  pd.merge(left = dpch_tech_nm, right=curtailment_nm[['scenario', 'timestamp', 'Curtailment']], on = ['scenario', 'timestamp'])


# %%

# -------------------------------------------------- INTERACTIVE PANEL - NEW MEXICO - HIGH STORAGE COSTS ---------------------------------------------------------------------

# import graph_objects from plotly package
import plotly.graph_objects as go
 
# import make_subplots function from plotly.subplots
# to make grid of plots
from plotly.subplots import make_subplots

# Dataframes
load = load_nm.copy()
stg = stg_nm.copy()
dpch_tech = dpch_tech_nm.copy()

# Enter dates
def filter_dates(df):
    start_date="2035 01 15 00"
    end_date="2035 12 30 00"
    time_zone='US/Mountain'
    time_1=datetime.strptime(str(start_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    time_2=datetime.strptime(str(end_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    return    df[(time_1 <= df.timestamp) & (df.timestamp < time_2)]  

#Dictionaries 
tech_order_1 = tech_order + ['Storage(C)', 'Storage(D)', 'Imports', 'Exports', 'Curtailment']
tech_order_1.remove('Storage')
tech_colors_1 = tech_colors.copy()
tech_colors_1['Storage(C)'] = "#7fffd4"
tech_colors_1['Storage(D)'] = "#7fffd4"
tech_colors_1['Imports'] = 'lightpink'
tech_colors_1['Exports'] = 'magenta'
tech_colors_1['Curtailment'] = 'Olive'

plot = go.Figure() 

plot = make_subplots(specs=[[{"secondary_y": True}]])

load0 =  load.loc[load.scenario == selected_scenarios[0]].pipe(filter_dates)
stg0 = stg.loc[stg.scenario == selected_scenarios[0]].pipe(filter_dates)
dpch_tech0 = dpch_tech.loc[dpch_tech.scenario == selected_scenarios[0]].pipe(filter_dates)
dpch_tech0.drop(['scenario'], axis=1, inplace=True)
dpch_tech0.set_index('timestamp', inplace=True)


for j in tech_order_1:
    col_name = j
    plot.add_trace(go.Scatter(x = list(dpch_tech0.index), 
                   y=  list(dpch_tech0[j]),
                   mode='lines',
                   line = dict(width = 0.5, color = tech_colors_1[j]),
                   name = col_name, 
                   stackgroup= 'positive' if j not in ['Storage(C)', 'Exports'] else 'negative')
                   )

plot.add_trace(go.Scatter(x = list(load0.timestamp), 
                   y=  load0.zone_demand_gw,
                   mode='lines',
                   line = dict(width = 1.5, color = 'red'),
                   name = 'Load', 
                   ))

plot.add_trace(go.Scatter(  x=list(stg0.timestamp),
                            y=stg0.StateOfChargeGWh,
                            line = dict(width = 1.5, color = 'black'),
                            name="State of Charge",
                            yaxis="y2"
))

plot.update_layout( xaxis=dict( rangeselector=dict( 
                buttons=list([ 
                    dict(count=1,
                     label="1 day",
                     step="day",
                     stepmode="backward"),
                dict(count=7,
                     label="1 week",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="1 month",
                     step="month",
                     stepmode="backward"),

            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
plot.update_traces(mode="lines", hovertemplate='%{y:.2f}')
plot.update_layout(hovermode = 'x unified', spikedistance = -1)

plot.update_layout(plot_bgcolor='white', title = 'New Mexico - High storage costs')
plot.update_xaxes(ticks='inside', mirror = True, showline=True, linecolor='black', gridcolor='lightgrey')
plot.update_yaxes(minor_ticks="inside", ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
plot.update_layout(yaxis=dict(title="Generation and Load (GW)"),yaxis2=dict(title="State of charge (GWh)"))
plot.update_layout(font_family="Times New Roman")
#plot.write_html("file.html")

plot.show()

# %%
# -------------------------------------------------- INTERACTIVE PANEL - NEW MEXICO - LOW STORAGE COSTS ---------------------------------------------------------------------

# import graph_objects from plotly package
import plotly.graph_objects as go
 
# import make_subplots function from plotly.subplots
# to make grid of plots
from plotly.subplots import make_subplots

# Dataframes
load = load_nm.copy()
stg = stg_nm.copy()
dpch_tech = dpch_tech_nm.copy()

# Enter dates
def filter_dates(df):
    start_date="2035 01 15 00"
    end_date="2035 12 30 00"
    time_zone='US/Mountain'
    time_1=datetime.strptime(str(start_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    time_2=datetime.strptime(str(end_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    return    df[(time_1 <= df.timestamp) & (df.timestamp < time_2)]  

#Dictionaries 
tech_order_1 = tech_order + ['Storage(C)', 'Storage(D)', 'Imports', 'Exports', 'Curtailment']
tech_order_1.remove('Storage')
tech_colors_1 = tech_colors.copy()
tech_colors_1['Storage(C)'] = "#7fffd4"
tech_colors_1['Storage(D)'] = "#7fffd4"
tech_colors_1['Imports'] = 'lightpink'
tech_colors_1['Exports'] = 'magenta'
tech_colors_1['Curtailment'] = 'Olive'

plot = go.Figure() 

plot = make_subplots(specs=[[{"secondary_y": True}]])

load0 =  load.loc[load.scenario == selected_scenarios[1]].pipe(filter_dates)
stg0 = stg.loc[stg.scenario == selected_scenarios[1]].pipe(filter_dates)
dpch_tech0 = dpch_tech.loc[dpch_tech.scenario == selected_scenarios[1]].pipe(filter_dates)
dpch_tech0.drop(['scenario'], axis=1, inplace=True)
dpch_tech0.set_index('timestamp', inplace=True)


for j in tech_order_1:
    col_name = j
    plot.add_trace(go.Scatter(x = list(dpch_tech0.index), 
                   y=  list(dpch_tech0[j]),
                   mode='lines',
                   line = dict(width = 0.5, color = tech_colors_1[j]),
                   name = col_name, 
                   stackgroup= 'positive' if j not in ['Storage(C)', 'Exports'] else 'negative'))

plot.add_trace(go.Scatter(x = list(load0.timestamp), 
                   y=  load0.zone_demand_gw,
                   mode='lines',
                   line = dict(width = 1.5, color = 'red'),
                   name = 'Load', 
                   ))

plot.add_trace(go.Scatter(  x=list(stg0.timestamp),
                            y=stg0.StateOfChargeGWh,
                            line = dict(width = 1.5, color = 'black'),
                            name="State of Charge",
                            yaxis="y2"
))

plot.update_layout( xaxis=dict( rangeselector=dict( 
                buttons=list([ 
                    dict(count=1,
                     label="1 day",
                     step="day",
                     stepmode="backward"),
                dict(count=7,
                     label="1 week",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="1 month",
                     step="month",
                     stepmode="backward"),

            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
plot.update_traces(mode="lines", hovertemplate='%{y:.2f}')
plot.update_layout(hovermode = 'x unified', spikedistance = -1)

plot.update_layout(plot_bgcolor='white', title = 'New Mexico - Low storage costs')
plot.update_xaxes(ticks='inside', mirror = True, showline=True, linecolor='black', gridcolor='lightgrey')
plot.update_yaxes(minor_ticks="inside", ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
plot.update_layout(yaxis=dict(title="Generation and Load (GW)"),yaxis2=dict(title="State of charge (GWh)"))
plot.update_layout(font_family="Times New Roman")
#plot.write_html("file.html")

plot.show()

#%%

# --------------------------------------------------------- LINES --------------------------------------------------------

def filter_dates(df):
    start_date="2035 07 15 00"
    end_date="2035 07 28 00"
    time_zone='US/Pacific'
    time_1=datetime.strptime(str(start_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    time_2=datetime.strptime(str(end_date), '%Y %m %d %H').replace(tzinfo=timezone(time_zone))
    return    df[(time_1 <= df.timestamp) & (df.timestamp < time_2)]  
    

load0 =  load_r.loc[load_r.scenario == selected_scenarios[0]].pipe(filter_dates)
flow0 =  flow_r.loc[flow_r.scenario == selected_scenarios[0]].pipe(filter_dates)
stg0 = stg_r.loc[stg_r.scenario == selected_scenarios[0]].pipe(filter_dates)
solar0 = dpch_r.loc[(dpch_r.tech_map == 'Solar') & (dpch_r.scenario ==selected_scenarios[0])].pipe(filter_dates)
wind0 = dpch_r.loc[(dpch_r.tech_map == 'Wind') & (dpch_r.scenario == selected_scenarios[0])].pipe(filter_dates)

load1 =  load_r.loc[load_r.scenario == selected_scenarios[1]].pipe(filter_dates)
flow1 =  flow_r.loc[flow_r.scenario == selected_scenarios[1]].pipe(filter_dates)
stg1 = stg_r.loc[stg_r.scenario == selected_scenarios[1]].pipe(filter_dates)
solar1 = dpch_r.loc[(dpch_r.tech_map == 'Solar')  & (dpch_r.scenario == selected_scenarios[1])].pipe(filter_dates)
wind1 = dpch_r.loc[(dpch_r.tech_map == 'Wind') & (dpch_r.scenario == selected_scenarios[1])].pipe(filter_dates)


fig, ((ax0, ax1)) = plt.subplots(nrows=2, ncols=1, figsize=(6.0,3.0))

ax0 = solar0.plot(ax = ax0, x ='timestamp', y ='DispatchGen_GW', color = 'orange')
ax0 = wind0.plot(ax = ax0, x ='timestamp', y ='DispatchGen_GW', color = 'blue')
ax0 = flow0.plot(ax = ax0, x ='timestamp', y ='TXPowerNet_GW', color = 'green')
ax0 = load0.plot(ax = ax0, x ='timestamp', y ='zone_demand_gw', color = 'red')

ax01 = ax0.twinx() 

ax01 = stg0.plot(ax = ax01, x ='timestamp', y ='StateOfChargeGWh')

ax0.set_ylim(-10, 70)
ax0.set(yticks = [0, 10, 20, 30, 40, 50, 60, 70])
ax01.set_ylim(-10, 500)
ax01.set(yticks = [0, 100, 200, 300, 400, 500])
ax0.set_title('(a)')
ax0.set_xlabel(' ')
ax0.get_legend().remove()
ax01.get_legend().remove()

ax1 = solar1.plot(ax = ax1, x ='timestamp', y ='DispatchGen_GW', color = 'orange')
ax1 = wind1.plot(ax = ax1, x ='timestamp', y ='DispatchGen_GW', color = 'blue')
ax1 = load1.plot(ax = ax1, x ='timestamp', y ='zone_demand_gw', color = 'red')
ax1 = flow1.plot(ax = ax1, x ='timestamp', y ='TXPowerNet_GW', color = 'green')

ax11 = ax1.twinx() 

ax11 = stg1.plot(ax = ax11, x ='timestamp', y ='StateOfChargeGWh')

ax1.set_ylim(-10, 70)
ax1.set(yticks = [0, 10, 20, 30, 40, 50, 60, 70])
ax11.set_ylim(-10, 500)
ax11.set(yticks = [0, 100, 200, 300, 400, 500])
ax1.set_title('(b)')
ax1.set_xlabel('Time')
ax1.get_legend().remove()
ax11.get_legend().remove()

fig.text(0.07,0.3, "Generation and load (GW)", rotation=90)
fig.text(0.95,0.3, "State of charge (GWh)", rotation=90)
plt.subplots_adjust(wspace=0, hspace=0.30)

folder_to_save_results = "figures/"
#plt.savefig(folder_to_save_results+figure_name, facecolor='White', transparent=False)
plt.savefig(folder_to_save_results+'temporal.pdf', facecolor='White', transparent=False)

# %%


