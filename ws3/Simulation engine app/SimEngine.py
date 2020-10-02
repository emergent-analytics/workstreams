# TITLE: SIMULATION ENGINE APP
# DESCRIPTION: This code outputs the Emergent Alliance's Simulation Engine to an app that can be interacted with using a web browser.  
# AUTHOR: Alvaro Corrales Cano - Data Scientist at IBM
# This is version 8 of the app

# IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.integrate import odeint, simps
from sqlalchemy import create_engine

# CLASS AND FUNCTION DEFINITIONS
# Read data in
@st.cache
def read_data(path = None, engine = None, table = None):
    if (path == None) & (engine != None):
        _df = pd.read_sql(table, engine)
        _df.index = _df['Sectors']
        _df.drop(columns = 'Sectors', inplace = True)
        _df.columns = _df.index

    elif (path != None) & (engine == None):
        _df = pd.read_csv(path + '/A_UK.csv', header = [0, 1], index_col= [0, 1])
        _df.columns = _df.index.get_level_values(1).values
        _df.index = _df.columns
    
    return _df

@st.cache
def read_GVA(path = None, engine = None, table = None):
    if (path == None) & (engine != None):
        _GVA = pd.read_sql(table, engine) # - TBC

    elif (path != None) & (engine == None):
        _GVA = pd.read_csv(path + '/GVA_UK.csv', index_col = [0,1], header = None)
        _GVA.index = _GVA.index.get_level_values(1).values
    
    return _GVA    

# Leontief Model
class LeonTradeModel:
    
    def __init__(self,df_A,demand='unit', type = 'Upstream'):
        self.df_A = df_A
        self.sector_indices = df_A.index.get_level_values(0).values
        self.sectors =  df_A.index.get_level_values(0).unique().values        
        if type == 'Upstream':
            self.A = df_A.values
        elif type == 'Downstream':
            self.A = df_A.T.values
        if (demand == 'unit'):
            self.d_base = np.ones(len(df_A))
        else:
            self.d_base = demand
        self.x_base = np.dot(self.df_A.values,self.d_base)
        self.x_out = self.x_base
        
    def shock_impulse(self, sectors_n_shocks=None, general_shock = [0, 0, 0]):
        shock_vec = {}
        for sector in self.sectors:
            shock_vec[sector] = (general_shock[0]/12, general_shock[1]/12, general_shock[2])
        
        for sector in sectors_n_shocks:
            shock_vec[sector] = sectors_n_shocks[sector]

        return shock_vec

    def recovery_impulse(self, sectors_n_stimuli = None, general_stimulus = [0, 0, 0]):
        recovery_vec = {}
        for sector in self.sectors:
                recovery_vec[sector] = (general_stimulus[0]/12, general_stimulus[1]/12, general_stimulus[2])
        
        for sector in sectors_n_stimuli:
                recovery_vec[sector] = sectors_n_stimuli[sector]

        return recovery_vec


# Dynamic propagation function
def economic_dynamics_ode(y,t, A, sectors, external_shock_vec_dict):
    '''
    Shocked economic dynamics with external shock vector. To be invoked with ode solver
    Args:
         y: array-like output (I/O matrix formulation)
         t: array-like timesteps
         A: I/O matrix
         external_shock_vec_dict: shock vector annotation dictionary with following annotations
                key: sector index (should correspond to position in the vector)
                values:(t0,t1,val) where shock of magnitude val persists between t0 and t1
               
    Returns:
         Return val of ODE
   
    '''
    try:
        economic_dynamics_ode.time_vec.append(t)
    except:
        economic_dynamics_ode.time_vec = [t]
    shock_vec = pd.Series(data = np.zeros(len(A)), index = sectors)
    for _sector_idx, _shock_attrs in external_shock_vec_dict.items():
        if (t>=_shock_attrs[0]) & (t<=_shock_attrs[1]):
            shock_vec[_sector_idx] = _shock_attrs[2]    

    return np.dot(A-np.eye(A.shape[0]),y) + shock_vec

# Propagation dynamics with recovery
def economic_dynamics_ode_rec(y,t, A, sectors, external_shock_vec_dict, recovery_dict):
    '''
    Shocked economic dynamics with external shock vector and recovery strategy (directed). To be invoked with ode solver
    Args:
         y: array-like output (I/O matrix formulation)
         t: array-like timesteps
         A: I/O matrix
         external_shock_vec_dict: shock vector annotation dictionary with following annotations
                key: sector index (should correspond to position in the vector)
                values:(t0,t1,val) where shock of magnitude val persists between t0 and t1
               
    Returns:
         Return val of ODE
   
    '''
    try:
        economic_dynamics_ode.time_vec.append(t)
    except:
        economic_dynamics_ode.time_vec = [t]
    shock_vec = pd.Series(data = np.zeros(len(A)), index = sectors)
    recovery_vec = pd.Series(data = np.zeros(len(A)), index = sectors)
    for _sector_idx, _shock_attrs in external_shock_vec_dict.items():
        if (t>=_shock_attrs[0]) & (t<=_shock_attrs[1]):
            shock_vec[_sector_idx] = _shock_attrs[2]
    
    for _sector_idx, _shock_attrs in recovery_dict.items():
        if (t>=_shock_attrs[0]) & (t<=_shock_attrs[1]):
            recovery_vec[_sector_idx] = _shock_attrs[2]
        
    return np.dot(A-np.eye(A.shape[0]),y) + shock_vec + recovery_vec 

# Total output loss function
def total_out_loss(sol,time_vec, by_sector = True, sectors = None, GVA_vec = None):
    '''
    Total output contraction using simple quadrature
    Args:
        sol: solution matrix
        time_vec: arraylike time steps 
        sectors: list of sectors in our economy
    Returns:
        Total output loss as a vector if sectors is not None, where each entry corresponds to a sector, or a scalar otherwise
    '''
    # num_sectors = sol.shape[1]
    if by_sector == True:
        _sol_df = pd.DataFrame(data = sol, columns = sectors)
        tot_out_loss = pd.Series(data = np.zeros(sol.shape[1]), index = sectors)
        for n in sectors:
            tot_out_loss.loc[n] = simps(_sol_df.loc[:,n],time_vec)
    else:
        num_sectors = sol.shape[1]
        tot_out_loss = 0
        for n in range(num_sectors):
            tot_out_loss += simps(sol[:,n],time_vec) * GVA_vec.iloc[n] / GVA_vec.sum()
    
    return tot_out_loss

# READ DATA IN
# Read A matrix locally
df_lev = read_data(path = '.')
# Read GVA matrix 
GVA_vec = read_GVA(path = '.')

# Can also read from Db2 database if preferred
# string = "------"
# engine = create_engine(string)
# table = 'a_uk'
# df_lev = read_data(engine = engine, table = table)
# table = 'gva_uk'
# GVA_vec = read_GVA(engine = engine, table = table)

# UI SET UP
# Initial header
st.title('Emergent Economic Engine')
st.write('This app allows you to see how a shock to one sector of the UK economy [1] propagates throughout the national economic network and how it is eventually absorbed over time. \
    You can think of it as a *what-if* type of simulation.')
st.write('**:point_left: To start, use the menu on the left to choose the parameters of the shock.**')

# Model parameter selection using UI - Sidebar set up
st.sidebar.markdown("# Model parameters")
st.sidebar.markdown('Use this menu to tailor your *what-if* scenario. You can choose the sectors that you want to shock, the magnitude of the shock and when it happens. \
    You will also be able to choose the parameters of your recovery path.')
st.sidebar.markdown('## Lever 1. Choose your shocks')

# Define time frame and stream of propagation
years = st.sidebar.slider(label = 'For how many years do you want to run the simulation? Note that the simulation will be broken down by months. \
                            Also, bear in mind that the further in time you go, the less precise the simulation.', \
                            min_value = 1, max_value = 8, value = 2, key = 'years')
months = years * 12

type_shock = st.sidebar.selectbox(label = 'Do you want your shock to propagate upstream or downstream the supply chain?', options = ['Upstream', 'Downstream'], index = 0, key = 'updown')

# Economy-wide shock
if st.sidebar.checkbox(label = 'Add an economy-wide shock (e.g. general for all sectors)', key = 'genshock'):
    st.sidebar.markdown("### Economy-wide initial shock")
    general_shock = st.sidebar.slider(label = 'What will be the relative magnitude of the economy-wide shock?', min_value = -100.0, max_value = 100.0, value = 0.0, key = 'genshock')
    start_general, end_general = st.sidebar.slider("Between what months do you want the economy-wide shock to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'genshock')
    general_shock_list = [start_general, end_general, general_shock/100]
else:
     general_shock_list = [0, 0, 0]

# Define shocks (up to 5)
st.sidebar.markdown("### Sector 1")
sector_1 = st.sidebar.selectbox(label = 'What sector do you want to start by shocking?', options = df_lev.index, key = 'sect1')
shock_val_1 = st.sidebar.slider(label = 'What will be the relative magnitude of this shock (percentage)?', min_value = -100.0, max_value = 100.0, value = 1.0, key = 'sect1')
start_sector_1, end_sector_1 = st.sidebar.slider("How many months would you like your initial shock to persist?", min_value = 0, max_value = months, value = [0, 6], key = 'sect1')

shocked_sectors = [sector_1]

sector_2, shock_val_2, start_sector_2, end_sector_2 = 0, 0, 0, 0
sector_3, shock_val_3, start_sector_3, end_sector_3 = 0, 0, 0, 0
sector_4, shock_val_4, start_sector_4, end_sector_4 = 0, 0, 0, 0
sector_5, shock_val_5, start_sector_5, end_sector_5 = 0, 0, 0, 0

if st.sidebar.checkbox(label = 'Add another sector', key = 'sect2'):
    st.sidebar.markdown("### Sector 2")
    sector_2 = st.sidebar.selectbox(label = 'What other sector do you want to shock?', options = df_lev.index, key = 'sect2')
    shock_val_2 = st.sidebar.slider(label = 'What will be the relative magnitude of this shock (percentage)?', min_value = -100.0, max_value = 100.0, value = 1.0, key = 'sect2')
    start_sector_2, end_sector_2 = st.sidebar.slider("Between what months do you want this shock to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'sect2')
    shocked_sectors.append(sector_2)

    if st.sidebar.checkbox(label = 'Add another sector', key = 'sect3'):
        st.sidebar.markdown("### Sector 3")
        sector_3 = st.sidebar.selectbox(label = 'What other sector do you want to shock?', options = df_lev.index, key = 'sect3')
        shock_val_3 = st.sidebar.slider(label = 'What will be the relative magnitude of this shock (percentage)?', min_value = -100.0, max_value = 100.0, value = 1.0, key = 'sect3')
        start_sector_3, end_sector_3 = st.sidebar.slider("Between what months do you want this shock to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'sect3')
        shocked_sectors.append(sector_3)

        if st.sidebar.checkbox(label = 'Add another sector', key = 'sect4'):
            st.sidebar.markdown("### Sector 4")
            sector_4 = st.sidebar.selectbox(label = 'What other sector do you want to shock?', options = df_lev.index, key = 'sect4')
            shock_val_4 = st.sidebar.slider(label = 'What will be the relative magnitude of this shock (percentage)?', min_value = -100.0, max_value = 1.0, value = 100.00, key = 'sect4')
            start_sector_4, end_sector_4 = st.sidebar.slider("Between what months do you want this shock to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'sect4')
            shocked_sectors.append(sector_4)

            if st.sidebar.checkbox(label = 'Add another sector', key = 'sect5'):
                st.sidebar.markdown("### Sector 5")
                sector_5 = st.sidebar.selectbox(label = 'What other sector do you want to shock?', options = df_lev.index, key = 'sect5')
                shock_val_5 = st.sidebar.slider(label = 'What will be the relative magnitude of this shock (percentage)?', min_value = -100.0, max_value = 100.0, value = 1.0, key = 'sect5')
                start_sector_5, end_sector_5 = st.sidebar.slider("Between what months do you want this shock to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'sect5')
                shocked_sectors.append(sector_5)

sectors_n_shocks = {sector_1: (start_sector_1 / 12, end_sector_1 / 12, shock_val_1 / 100),
                    sector_2: (start_sector_2 / 12, end_sector_2 / 12, shock_val_2 / 100),
                    sector_3: (start_sector_3 / 12, end_sector_3 / 12, shock_val_3 / 100),
                    sector_4: (start_sector_4 / 12, end_sector_4 / 12, shock_val_4 / 100),
                    sector_5: (start_sector_5 / 12, end_sector_5 / 12, shock_val_5 / 100)}

# Choose a recovery path
st.sidebar.markdown('## Lever 2. Choose your recovery strategy (optional)')
st.sidebar.markdown('In this lever, you can see how the economy that you modelled in lever 1 would do if you applied a stimulus to boost its recovery. The stimulus can be spread across sectors or \
                    target specific sectors (not necessarily the sames that you shocked previously).')
want_recovery = st.sidebar.checkbox(label = 'I want to model a recovery strategy', key = 'want_recovery')

if want_recovery:
    
    if st.sidebar.checkbox(label = 'Add an economy-wide stimulus (e.g. general for all sectors)', key = 'genstim'):
        st.sidebar.markdown("### Economy-wide stimulus")
        general_stimulus = st.sidebar.slider(label = 'What will be the relative value of the stimulus as a percentage of the **total output of the sector**?', min_value = 0.0, max_value = 100.0, value = 0.0, key = 'genstim')
        start_general_st, end_general_st = st.sidebar.slider("Between what months do you want the stimulus to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'genstim')
        general_stimulus = [start_general_st, end_general_st, general_stimulus/100]
    else:
        general_stimulus = [0, 0, 0]
        
    if st.sidebar.checkbox(label = 'Do you want to target any specific sectors?', key = 'specstim'):
        
        st.sidebar.markdown("### Sector 1")
        rec_1 = st.sidebar.selectbox(label = 'What sector do you want to start by stimulate?', options = df_lev.index, key = 'rec1')
        rec_val_1 = st.sidebar.slider(label = 'What will be the relative magnitude of this stimulus (percentage)?', min_value = 0.0, max_value = 100.0, value = 1.0, key = 'rec1')
        start_rec_1, end_rec_1 = st.sidebar.slider("Between what months do you want this stimulus to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'rec1')
        
        rec_sectors = [rec_1]

        rec_2, rec_val_2, start_rec_2, end_rec_2 = 0, 0, 0, 0
        rec_3, rec_val_3, start_rec_3, end_rec_3 = 0, 0, 0, 0
        rec_4, rec_val_4, start_rec_4, end_rec_4 = 0, 0, 0, 0
        rec_5, rec_val_5, start_rec_5, end_rec_5 = 0, 0, 0, 0

        if st.sidebar.checkbox(label = 'Add another sector', key = 'rec2'):
            st.sidebar.markdown("### Sector 2")
            rec_2 = st.sidebar.selectbox(label = 'What other sector do you want to start by stimulate?', options = df_lev.index, key = 'rec2')
            rec_val_2 = st.sidebar.slider(label = 'What will be the relative magnitude of this stimulus (percentage)?', min_value = 0.0, max_value = 100.0, value = 1.0, key = 'rec2')
            start_rec_2, end_rec_2 = st.sidebar.slider("Between what months do you want this stimulus to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'rec2')
            
            rec_sectors.append(rec_2)

            if st.sidebar.checkbox(label = 'Add another sector', key = 'rec3'):
                st.sidebar.markdown("### Sector 3")
                rec_2 = st.sidebar.selectbox(label = 'What other sector do you want to start by stimulate?', options = df_lev.index, key = 'rec3')
                rec_val_3 = st.sidebar.slider(label = 'What will be the relative magnitude of this stimulus (percentage)?', min_value = 0.0, max_value = 100.0, value = 1.0, key = 'rec3')
                start_rec_3, end_rec_3 = st.sidebar.slider("Between what months do you want this stimulus to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'rec3')
                
                rec_sectors.append(rec_3)    

                if st.sidebar.checkbox(label = 'Add another sector', key = 'rec4'):
                    st.sidebar.markdown("### Sector 4")
                    rec_4 = st.sidebar.selectbox(label = 'What other sector do you want to start by stimulate?', options = df_lev.index, key = 'rec4')
                    rec_val_4 = st.sidebar.slider(label = 'What will be the relative magnitude of this stimulus (percentage)?', min_value = 0.0, max_value = 100.0, value = 1.0, key = 'rec4')
                    start_rec_4, end_rec_4 = st.sidebar.slider("Between what months do you want this stimulus to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'rec4')
                    
                    rec_sectors.append(rec_4) 

                    if st.sidebar.checkbox(label = 'Add another sector', key = 'rec5'):
                        st.sidebar.markdown("### Sector 5")
                        rec_5 = st.sidebar.selectbox(label = 'What other sector do you want to start by stimulate?', options = df_lev.index, key = 'rec5')
                        rec_val_5 = st.sidebar.slider(label = 'What will be the relative magnitude of this stimulus (percentage)?', min_value = 0.0, max_value = 100.0, value = 1.0, key = 'rec5')
                        start_rec_5, end_rec_5 = st.sidebar.slider("Between what months do you want this stimulus to happen?", min_value = 0, max_value = months, value = [0, 6], key = 'rec5')
                        
                        rec_sectors.append(rec_5) 
    
        sectors_n_stimuli = {rec_1: (start_rec_1 / 12, end_rec_1 / 12, rec_val_1 / 100),
                        rec_2: (start_rec_2 / 12, end_rec_2 / 12, rec_val_2 / 100),
                        rec_3: (start_rec_3 / 12, end_rec_3 / 12, rec_val_3 / 100),
                        rec_4: (start_rec_4 / 12, end_rec_4 / 12, rec_val_4 / 100),
                        rec_5: (start_rec_5 / 12, end_rec_5 / 12, rec_val_5 / 100)}
    else:
        sectors_n_stimuli = {}
    

# RUN THE MODEL
# Initialise the class
network_model = LeonTradeModel(df_lev, type = type_shock) 
shock_vec = network_model.shock_impulse(sectors_n_shocks = sectors_n_shocks, general_shock = general_shock_list)
demand_vec = shock_vec #network_model.d_base-shock_vec

if (want_recovery == True):
    recovery_vec = network_model.recovery_impulse(sectors_n_stimuli = sectors_n_stimuli, general_stimulus = general_stimulus)


# Run dynamic shock
sol = odeint(economic_dynamics_ode, np.zeros(len(network_model.df_A)), np.linspace(0,years,months), args=(network_model.A, network_model.sectors, shock_vec))

# Total output change
total_change = total_out_loss(sol, np.linspace(0,years,months), by_sector = True, sectors = network_model.sectors)

# Run dynamic shock plus stimulus
if want_recovery:
    sol_rec = odeint(economic_dynamics_ode_rec, np.zeros(len(network_model.df_A)), np.linspace(0,years,months), args = (network_model.A, network_model.sectors, shock_vec, recovery_vec))

# VISUALISATION
# Time dynamics with no recovery
st.write(' ### ** Impact of shock on the economy ** \
        \n In the chart below we can how the shock is propagated throughout the different sectors of the economy and it is absorbed as time passes. \
        \n \n Hover over the chart to see what sectors are the most affected at each point in time.')

df_viz = pd.DataFrame(data = sol * 100, columns = df_lev.columns)


viz_columns = st.multiselect(label = 'Select the sectors that you want to visualise:', options = df_lev.columns, default = [], key = 'viz_columns')

if (viz_columns != []):
    df_viz = df_viz[viz_columns]


fig = px.line(df_viz, color_discrete_sequence=px.colors.sequential.Redor)
fig.layout.update(showlegend = False, width = 800, height = 800, template = 'simple_white')
fig.update_layout(xaxis = dict(title_text = "Months after initial shock"), yaxis = dict(title_text = "% change"))

st.plotly_chart(fig)

# Overall impact chart
change_no_intervention = round(total_out_loss(sol, np.linspace(0,years,months), by_sector = False, GVA_vec= GVA_vec) * 100, 2)

st.write('**Overall, output changes by**', change_no_intervention.iloc[0], "**%**"
        '\n \n The chart below presents the output change broken down by sectors. \
         It excludes the sectors that we have shocked manually - the fall here would equal what we set in our initial parameters. \
        \n \n Again, you can hover over the chart to see the exact name of the sector and output change.')

x_min = np.abs(total_change.drop(index = shocked_sectors).min())
x_max = np.abs(total_change.drop(index = shocked_sectors).max())
xtick = x_max if x_max > x_min else x_min

fig = px.bar(x = total_change.drop(index = shocked_sectors).sort_values(ascending=True),
            # y = total_change.drop(index = sectors).index,
            color=total_change.drop(index = shocked_sectors).index,
            color_discrete_sequence=px.colors.sequential.Redor
            )
fig.layout.update(showlegend = False, width = 800, height = 800, template = 'simple_white', 
                yaxis = dict(showline = False, showticklabels = False, color = 'white'),
                xaxis = dict(title_text = '% change', range = [-xtick, xtick]))
st.plotly_chart(fig)

if want_recovery:

    change_intervention = round(total_out_loss(sol_rec, np.linspace(0,years,months), by_sector = False, GVA_vec= GVA_vec) * 100, 2)

    st.write(' ### ** Countermeasuring the shock ** ', \
            "\n In case of no intervention, the output change would be", change_no_intervention.iloc[0], "%." 
            "\n The selected intervention strategy would yield a change in total output of", change_intervention.iloc[0], "%.")
    
    st.write('In the chart below, you can see the impact on total output (or the selected sector) of your selected recovery strategy. \
                The two lines represent the change in total output with and without intervention (dark and light line, respectively).')

    counter_sectors_list = ['- Aggregate Economy -']
    counter_sectors_list.extend(list(df_lev.columns))
    sector_viz = st.selectbox(label = 'Which sector do you want to visualise?', options = counter_sectors_list, key = 'counterviz')
    
    df_viz2 = pd.DataFrame()

    if sector_viz =='- Aggregate Economy -':
        df_viz2['No intervention scenario'] = np.sum((sol * (GVA_vec / GVA_vec.sum()).values.T) * 100, axis = 1)
        df_viz2['Intervention scenario'] = np.sum((sol_rec * (GVA_vec / GVA_vec.sum()).values.T) * 100, axis = 1)
    else:
        df_sol = pd.DataFrame(data = sol, columns = df_lev.columns)
        df_sol_rec = pd.DataFrame(data = sol_rec, columns = df_lev.columns)
        
        df_viz2['No intervention scenario'] = df_sol[sector_viz].values * 100
        df_viz2['Intervention scenario'] = df_sol_rec[sector_viz].values * 100

    fig = px.line(df_viz2, color_discrete_sequence=['rgb(245, 183, 142)', 'rgb(202, 82, 104)'])
    fig.layout.update(showlegend = False, width = 800, height = 800, template = 'simple_white')
    fig.update_layout(xaxis = dict(title_text = "Months after initial shock"), 
                    yaxis = dict(title_text = "% change"))
    
    st.plotly_chart(fig)


# Regional-Risk Pulse index text and logos
st.write("\n \n \n \n")
st.markdown("_______")
st.markdown("We are a team of data scientists from [IBM's Data Science & AI Elite Team](https://www.ibm.com/community/datascience/elite/), \
        IBM's Cloud Pak Acceleration Team, and [Rolls-Royce's R2 Data Labs](https://www.rolls-royce.com/products-and-services/r2datalabs.aspx) \
         working on Regional Risk-Pulse Index: forecasting and simulation within [Emergent Alliance](https://emergentalliance.org/). \
         Have a look at our [challenge statement](https://emergentalliance.org/?page_id=1659)!")

st.markdown("_______")
st.markdown("[1] Office for National Statistics (2020), *UK input-output analytical tables - industry by industry*, URL: \
     https://www.ons.gov.uk/economy/nationalaccounts/supplyandusetables/datasets/ukinputoutputanalyticaltablesindustrybyindustry, last accessed: \
     27 August 2020. Contains public sector information licensed under the Open Government License v3.0. http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/")

# Copyright © IBM Corp. 2020. Licensed under the Apache License, Version 2.0. Released as licensed Sample Materials.
