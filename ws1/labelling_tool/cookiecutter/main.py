# Cookiecutter tool
#
# Copyright (C) Dr. Klaus G. Paul 2020 Rolls-Royce Deutschland Ltd & Co KG
# Made on behalf of Emergent Alliance Ltd
#
# Notice to users: This is a Minimum Viable Product designed to elicit user requirements. The code
# quality is write-once, reuse only if deemed necessary
#
from bokeh.models import Button, Plot, TextInput, Legend
from bokeh.palettes import RdYlBu3, RdYlBu11, Turbo256, Plasma256, Colorblind8
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.models.widgets import Select
from bokeh.layouts import row, column, layout
from bokeh.models import Range1d, HoverTool, LinearAxis, Label, NumeralTickFormatter, PrintfTickFormatter, Div, LinearColorMapper, ColorBar, BasicTicker
from bokeh.models import BoxAnnotation, DataTable, DateFormatter, TableColumn, RadioGroup, Spinner, Paragraph, RadioButtonGroup
from bokeh.models import Panel, Tabs, DatePicker, PointDrawTool, DataTable, TableColumn, NumberFormatter, DataRange1d
from bokeh.palettes import Category20, Category10, RdYlBu3, Greys4
from bokeh.client.session import push_session, show_session
from bokeh.events import SelectionGeometry, ButtonClick
from bokeh.transform import transform
from sqlalchemy import create_engine
from bokeh.transform import linear_cmap
import sqlalchemy
import datetime
import pandas as pd
import os
import sys
import gzip
import json
import pickle
import numpy as np
# suppress a pyearth internal issue 
# pyearth/earth.py:1066: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
# To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
#  coef, resid = np.linalg.lstsq(B, weighted_y[:, i])[0:2]
np.warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose
import base64
from pathlib import Path
import pycountry
import socket
import random
import urllib
from pyearth import Earth
from pyearth import export
from operator import add

import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

class THEME():    
    def __init__(self,theme="Emergent"):
        """This is used to set some colors of the UI. Note there is another aspect to this
        in the cookiecutter/templates/index.html file which specifies CSS styles for
        UI elements that are trickier to override.
        """
        if theme == "Emergent":
            self.background_color="#252425"
            self.background_fill_color="#252425"
            self.plot_color="#F1F1F1"
            self.text_font="NotoSans"
            self.text_color="#F1F1F1"
            self.tick_line_color="#F1F1F1"
            self.axis_label_text_color="#F1F1F1"
            self.axis_line_color="#F1F1F1"
            self.label_text_color="#F1F1F1"
            self.tick_line_color="#F1F1F1"
            self.good = "green"
            self.bad = "red"
            self.amber = "orange"
            self.indet = "silver"
        else:
            pass


    def apply_theme_defaults(self,bokeh_tingy):
        """To declutter the code, we try to override theme color settings in bokeh widgets.
        This is brute force and will try to set values for elements that have no such attributes,
        which is why everything is enclosed in try/excepts.
        """
        try:
            bokeh_tingy.background_fill_color = self.background_color
        except:
            pass

        try:
            bokeh_tingy.border_fill_color = self.background_color
        except:
            pass

        try:
            bokeh_tingy.color = self.plot_color
        except:
            pass

        try:
            bokeh_tingy.axis.axis_label_text_color=self.plot_color
        except:
            pass

        try:
            bokeh_tingy.axis.axis_line_color=self.plot_color
        except:
            pass

        try:
            bokeh_tingy.axis.major_label_text_color=self.plot_color
        except:
            pass

        try:
            bokeh_tingy.axis.major_label_text_font=self.text_font
        except:
            pass

        try:
            bokeh_tingy.xaxis.major_label_text_font=self.text_font
        except:
            pass

        try:
            bokeh_tingy.yaxis.major_label_text_font=self.text_font
        except:
            pass

        try:
            bokeh_tingy.yaxis.major_label_text_color=self.text_color
        except:
            pass

        try:
            bokeh_tingy.title.text_font=self.text_font
        except:
            pass

        try:
            if 'label_text_font' in bokeh_tingy.legend.__dict__.keys():
                bokeh_tingy.legend.label_text_font = self.text_font
        except:
            pass

        try:
            if 'label_text_line_height' in bokeh_tingy.legend.__dict__.keys():
                bokeh_tingy.legend.label_text_line_height = 10
        except:
            pass

        try:
            if 'glyph_height' in bokeh_tingy.legend.__dict__.keys():
                bokeh_tingy.legend.glyph_height = 20
        except:
            pass

        try:
            if 'label_text_color' in bokeh_tingy.legend.__dict__.keys():
                bokeh_tingy.legend.label_text_color = self.text_color
        except:
            pass

        try:
            if 'background_fill_color' in bokeh_tingy.legend.__dict__.keys():
                bokeh_tingy.legend.background_fill_color = self.background_color
        except:
            pass

        try:
            if 'label_standoff' in bokeh_tingy.legend.__dict__.keys():
                bokeh_tingy.legend.label_standoff = 0
        except:
            pass

        try:
            if 'border_line_color' in bokeh_tingy.legend.__dict__.keys():
                bokeh_tingy.legend.border_line_color = self.background_color
        except:
            pass

        try:
            bokeh_tingy.axis.major_tick_line_color=self.plot_color
        except:
            pass

        try:
            if 'margin' in bokeh_tingy.legend.__dict__.keys():
                bokeh_tingy.legend.margin = 0
        except:
            pass

        try:
            if 'padding' in bokeh_tingy.legend.__dict__.keys():
                bokeh_tingy.legend.padding = 0
        except:
            pass

        try:
            bokeh_tingy.title.text_color=self.plot_color
        except:
            pass

        try:
            bokeh_tingy.xaxis.minor_tick_line_color=None
        except:
            pass

        try:
            bokeh_tingy.yaxis.minor_tick_line_color=None
        except:
            pass

        try:
            bokeh_tingy.toolbar.logo=None
        except:
            pass

        try:
            bokeh_tingy.xgrid.grid_line_alpha = 0.5
        except:
            pass

        try:
            bokeh_tingy.xgrid.grid_line_dash = [2,4]
        except:
            pass

        try:
            bokeh_tingy.ygrid.grid_line_alpha = 0.5
        except:
            pass

        try:
            bokeh_tingy.ygrid.grid_line_dash = [2,4]
        except:
            pass

        try:
            bokeh_tingy.xgrid.grid_line_color = "#000000"
        except:
            pass

        try:
            bokeh_tingy.xgrid.grid_line_alpha = 0.5
        except:
            pass

        try:
            bokeh_tingy.xgrid.grid_line_dash = [2,4]
        except:
            pass

        try:
            bokeh_tingy.ygrid.grid_line_color = "#000000"
        except:
            pass

        try:
            bokeh_tingy.ygrid.grid_line_alpha = 0.5
        except:
            pass

        try:
            bokeh_tingy.ygrid.grid_line_dash = [2,4]
        except:
            pass

        try:
            bokeh_tingy.toolbar.logo=None
        except:
            pass

        #try:

        return bokeh_tingy


def compute_gumbel_waves(df,region="",mincases=2.5,maxwaves=25):
    def gumpdf(x, beta, mu):
        """Return PDF value according to Gumbel"""
        expon = - ((x - mu) / beta)
        return(np.exp(expon) * np.exp(- (np.exp(expon))) / beta)

    def gumcdf(x, beta, mu):
        """Return CDF value according to Gumbel"""
        expon = - ((x - mu) / beta)
        return(np.exp(- (np.exp(expon))))

    #print(df)
    
    lastdate = df["datetime_date"].max() - pd.Timedelta('7 days')
    #print(df["datetime_date"])
    
    measure  = "new_cases" #cases
    smeasure = 'week_window' # smoothed
    rmeasure = 'rcases'      # remaining
    pmeasure = 'model'       # predicted
    wmeasure = 'wave_'       # waves

    df_geo = df.pivot_table(index="datetime_date",values=[measure],aggfunc="sum").fillna(0)
    df_geo['daynum'] = (df_geo.index - df_geo.index.min()).days
    
    alldata = []

    wave = 1

    df_geo[pmeasure] = 0
    df_geo[smeasure] = df_geo[measure].loc[:lastdate].rolling(7).mean()
    df_geo[rmeasure] = df_geo[smeasure]

    plotlist = [smeasure, pmeasure]

    mincases = 2.5
   
    while True:
        #print("Wave {} ".format(wave))
        curwave = wmeasure + str((wave) + 1000)[-2:]
        df_geo[curwave] = 0

        df_pred = pd.DataFrame({'daynum':df_geo['daynum'],measure:df_geo[rmeasure]})
        
        df_pred['gumdiv'] = df_pred[measure] / df_pred[measure].cumsum()
        df_pred = df_pred[(df_pred['gumdiv'] > 0) & (df_pred[measure] > mincases)]

        df_pred['linear'] = np.log(df_pred['gumdiv'])

        df_pred = df_pred[(df_pred['linear'] < -0.5) &
                          (df_pred['linear'] > -4.5)]

        if len(df_pred) <= 1:
            #print('--- no data left')
            break

        eax = df_pred['daynum'].values.reshape(-1, 1)
        eay = df_pred['linear'].values.reshape(-1, 1)

        eamodel = Earth(minspan=1, penalty=0, endspan=0, thresh=1e-9, check_every=1)
        eamodel.fit(eax, eay)

        df_pred['earth'] = eamodel.predict(eax)

        daymin = df_pred['daynum'].min()
        daymax = df_pred['daynum'].max()

        df_pred['gbgrad'] = df_pred['linear'] - df_pred['linear'].shift(1)
        df_pred['eagrad'] = df_pred['earth'] - df_pred['earth'].shift(1)

        fitmod = export.export_python_function(eamodel)

        df_pred['knot'] = ((abs(df_pred['eagrad'] - df_pred['eagrad'].shift(1)) > 1e-6) |
                           (df_pred['daynum'] == (daymin + 1)) |
                           (df_pred['daynum'] == daymax))
        df_pred['daycount'] = df_pred.reset_index().index

        df_knot = df_pred[df_pred['knot']][['daynum', 'daycount', 'eagrad']]
        df_knot['daysdata'] = df_knot['daycount'].shift(-1) - df_knot['daycount']
        df_knot['daystime'] = df_knot['daynum'].shift(-1) - df_knot['daynum']

        df_knot['cand'] = ((df_knot['eagrad'] < -1/33) &
                           (df_knot['daysdata'] >= 3))
        
        df_knot['since'] = df_knot['daynum'] - daymin
        df_knot['score'] = (df_knot['eagrad'] ** 2) * np.sqrt(df_knot['daysdata'] / np.sqrt(df_knot['since']))
        #df_knot['score'] = - df_knot['eagrad'] * df_knot['daysdata']
        df_knot['choice'] = df_knot['score'] == df_knot[df_knot['cand']]['score'].max()

        choice = df_knot[df_knot['choice']]
        if len(choice) == 0:
            #print('--- no data for wave')
            break

        lower = choice['daynum'].values[0]
        upper = choice['daysdata'].values[0] + lower

        df_pred = df_pred[(df_pred['daynum'] >= lower) &
                          (df_pred['daynum'] <= upper)].copy()

        slope = (fitmod([[upper]])[0] - fitmod([[lower]])[0]) / (upper - lower)
        intercept = fitmod([[lower]])[0] - (lower * slope)

        beta = - 1 / slope
        mu = beta * (intercept + np.log(beta))

        df_pred['pgumb'] = gumpdf(df_pred['daynum'], beta, mu)
        df_pred['scale'] = df_pred[measure] / df_pred['pgumb']

        final = df_pred['scale'].mean()
        fincv = df_pred['scale'].std() / final

        df_geo[curwave] = final * gumpdf(df_geo['daynum'], beta, mu)     

        peak = df_geo[df_geo[curwave] == df_geo[curwave].max()].index.min()
        start = df_geo[(df_geo[curwave] >= 1) &
                       (df_geo[curwave].index < peak)].index.min()
        floor = df_geo[(df_geo[curwave] < 1) &
                       (df_geo[curwave].index > peak)].index.min()

        peak = pd.to_datetime(peak) # suppress an all-of-a-sudden issue between machines
        start = pd.to_datetime(start)
        floor = pd.to_datetime(floor)

        alldata.append({"region":region,"wave":wave,"beta":beta,"mu":mu,"fit":(1 - fincv) ** 2,
                        "date_peak":peak.date(), "date_start":start.date(), "date_floor":floor.date(), "size":final,
                       "datetime_date":df_geo[curwave].index,"wave_values":df_geo[curwave].values,
                       "mincases":mincases})

        df_geo[pmeasure] += df_geo[curwave]
        df_geo[rmeasure] -= df_geo[curwave]
        plotlist += [curwave]
        wave += 1
        if wave > maxwaves:
            break

    return alldata


def compute_changepoint_waves(sCountry,ADM0_A3="",country=""):
    # computing waves and "periods of calmness" using a very manual Schmitt-Trigger style detection of gradients up and down
    all_verdicts = []
    
    THRESHOLD = 1
    THRESHOLD_UP = 14
    THRESHOLD_DOWN = 28
    
    data = sCountry.rolling(center=True,window=7).mean().dropna()
        
    datum = data.values[0]
    increasing = 0
    decreasing = 0
    wave_no = 0
    for i,v in data.items():
        if v > datum:
            if increasing == 0:
                start_date = i
            increasing += 1
            if increasing > 3:
                decreasing = 0
        elif v < datum:
            decreasing += 1
            if decreasing > 3:
                increasing = 0

        if increasing == THRESHOLD_UP:
            wave_no += 1
            if len(all_verdicts)>0 and all_verdicts[-1]["kind"] == "begin":
                pass
            else:
                all_verdicts.append({"name":country,"datetime_date":i,"kind":"begin","wave_no":wave_no,"adm0_a3":ADM0_A3})
        if decreasing == THRESHOLD_DOWN:
            if len(all_verdicts)>0 and all_verdicts[-1]["kind"] == "end":
                all_verdicts.pop()
                all_verdicts.append({"name":country,"datetime_date":i,"kind":"end","wave_no":wave_no,"adm0_a3":ADM0_A3})
            else:
                all_verdicts.append({"name":country,"datetime_date":i,"kind":"end","wave_no":wave_no,"adm0_a3":ADM0_A3})
        datum = v

    if len(all_verdicts) > 0:
        dfWaves = pd.DataFrame(all_verdicts)
        dfWaves = dfWaves.sort_values(["name","datetime_date"])
        return dfWaves
    else:
        return pd.DataFrame({"name":[],"datetime_date":[],"kind":[],"wave_no":[]})



class GUIHealth():
    """Base class for the bokeh server application.

    The logic flow is 
        GUI.init()
        GUI.create()
        GUI.compute_data_status()
        GUI.load_data()
    Most other aspects are handled via callback of widgets triggered by user interactions.
    """
    def __init__(self):
        """Set some basics like needing to load data and the theming color.
        """
        #self.set_theme("RR")
        self.theme = THEME()
        self.data_status = "no_data" # "stale", "current"
        if "SQL_CONNECT" not in list(os.environ.keys()):
            sql_url = "sqlite:///database.sqlite"
            sql_url = "postgresql://cookiecutter:cookiecutter@database:5432/cookiec"
            #sql_url = "db2+ibm_db://db2inst1:cookiecutter@192.168.100.197:50000/cookiec"
        else:
            print("HAVE CONNECT {}".format(os.environ["SQL_CONNECT"]))
            sql_url = os.environ["SQL_CONNECT"]
        #print("SQL_CONNECT {}".format(sql_url))
        self.engine = create_engine(sql_url)
        print(self.engine)
        try:
            self.max_gumbel_waves = int(os.environ["GUMBEL_MAX_WAVES"])
        except:
            self.max_gumbel_waves = 25
        self.dataset_selected = ""


    def refresh_data_callback(self,event):
        """This handles downloading data and recomputing some of the derived values.
        The idea was to provide user feedback via a progress bar, unfortunately, bokeh
        does not allow for UI updates while handling a callback (which makes a lot of sense,
        but is unfortunate in our case).
        """
        print("REFRESH")
        self.download_data()
        return
        print("PROCESS")
        self.process_data()
        print("OK")
        self.data_status = "current"
        self.cds.selected.on_change('indices',self.on_selection_change_callback)
        self.country_select.options=sorted(self.adfCountryData.keys())
        self.compute_data_status()
        self.country_select.value = "Germany"


    def on_selection_change_callback(self,attr,old,new):
        """Handling of (de)selecting data in the time series plots.
        If a selection is made, it unlocks the SAVE button.

        Also, the heatmap display would not sync itself to the range selection so this is done here,
        """

        # (un)lock Save button
        if len(self.cds.selected.indices) > 0:
            self.save.disabled = False
        else:
            self.save.disabled = True

        # make selection in the heatmap
        dates = []
        for i in self.cds.selected.indices:
            dates.append(self.cds.data["datetime_date"][i])
        selection = []
        i = 0
        for d in self.cds_OxCGRTHeatmap.data["datetime_date"]:
            if d in dates:
                selection.append(i)
            i += 1
        self.cds_OxCGRTHeatmap.selected.indices = selection

    def compute_metrics(self,bins=25):
        """using self.dfVotesContent, this computes stats for display
        """
        conn = self.engine.connect()
        try:
            ddf = pd.read_sql("select DISTINCT from_dt,to_dt,user,kind,vote_id,rel_peak_new_cases,duration from cookiecutter_verdicts",conn)
        except:
            ddf = pd.DataFrame()
        if len(ddf) > 1:
            #ddf = self.dfVotesContent[["from","to","user","kind","filename","rel_peak_new_cases","duration"]].drop_duplicates()

            sWaveDurations = ddf[ddf["kind"] == "Wave"].duration
            y, x_tmp = np.histogram(sWaveDurations,bins=bins)
            width = np.diff(x_tmp)
            x = [x_tmp[0]+i*width[i] for i in range(len(y))]
            self.cds_wave_duration_histogram.data = {"x":x,"y":y,"color":["tomato" for i in x],"width":width}

            sCalmDurations = ddf[ddf["kind"] == "Calm"].duration
            y, x_tmp = np.histogram(sCalmDurations,bins=bins)
            width = np.diff(x_tmp)
            x = [x_tmp[0]+i*width[i] for i in range(len(y))]
            self.cds_calm_duration_histogram.data = {"x":x,"y":y,"color":["mediumseagreen" for i in x],"width":width}

            peak_cutoff = np.quantile(ddf.rel_peak_new_cases,0.95)
            ddf = ddf[ddf["kind"] == "Wave"]
            dfHeatmap_tmp = ddf[ddf.rel_peak_new_cases < peak_cutoff][["rel_peak_new_cases","duration"]].copy()
            dfHeatmap_tmp["n"] = 1
            try:
                dfTmp = dfHeatmap_tmp.groupby([pd.cut(dfHeatmap_tmp.rel_peak_new_cases, bins),pd.cut(dfHeatmap_tmp.duration, bins)]).n.sum().unstack()
                dfHeatmapData = dfTmp.stack().reset_index().rename(columns={0:"n"}).dropna()
                dfHeatmapData["rpc"] = [i.mid for i in dfHeatmapData.rel_peak_new_cases.values]
                dfHeatmapData["d"] = [i.mid for i in dfHeatmapData.duration.values]
                h = dfHeatmapData.loc[0].duration.length
                w = dfHeatmapData.loc[0].rel_peak_new_cases.length
                self.cds_votes_heatmap.data = {"n":dfHeatmapData.n.values,"rpc":dfHeatmapData.rpc.values,"d":dfHeatmapData.d.values,
                                                "h":[h for i in dfHeatmapData.index],"w":[w for i in dfHeatmapData.index]}    
            except:
                pass
           


    def save_callback(self,event):
        """Saves the currently made selection in a csv file, updates the status bar
        with a corresponding message, and resets the selection (which implicitly will
        lock the save button again.
        """
        # Save selection
        conn = self.engine.connect()
        result = conn.execute("select max(vote_id) from cookiecutter_verdicts")
        vote_id = result.fetchone()[0]+1
        print("VOTE ID = {}".format(vote_id))

        df = self.cds.to_df().iloc[self.cds.selected.indices]
        df["rel_peak_new_cases"] = max(df["new_cases_rel"])
        df["kind"] = self.scenario_type.labels[self.scenario_type.active]
        df["kind_counter"] = int(self.scenario_number.value)
        df["from_dt"] = df.datetime_date.min()
        df["to_dt"] = df.datetime_date.max()
        df["duration"] = (pd.to_datetime(df.datetime_date.max())-pd.to_datetime(df.datetime_date.min())).total_seconds()/86400
        df["user"] = self.user_id.value
        #ADM0_A3 = self.adm0_a3 #dfMapping[self.dfMapping.name == self.country_select.value].adm0_a3.values[0]
        df["adm0_a3"] = self.adm0_a3
        df["vote_datetime"] = datetime.datetime.now()
        #filename = "data/{}.{}.{}.{}.{:%Y%m%d}.{:%Y%m%d}.{}.csv".format(ADM0_A3,
        #        self.country_select.value.replace("*","").replace("'","_"), # Taiwan* Cote d'Ivoire
        #        self.scenario_type.labels[self.scenario_type.active],self.scenario_number.value,
        #        pd.to_datetime(df.date.values[0]),
        #        pd.to_datetime(df.date.values[-1]),
        #        self.user_id.value)
        #df.to_csv(filename,index=False)
        df.to_sql("cookiecutter_verdicts", conn, if_exists='append', dtype={"from_dt":sqlalchemy.types.Date,
                                                                         "to_dt":sqlalchemy.types.Date,
                                                                         "datetime_date":sqlalchemy.types.DateTime,
                                                                         "vote_datetime":sqlalchemy.types.DateTime,
                                                                         "kind":sqlalchemy.types.String(10),
                                                                         "user":sqlalchemy.types.String(50),
                                                                         "adm0_a3":sqlalchemy.types.String(10)},index=False)
        df["vote_id"] = vote_id
        # reset selection
        self.cds.selected.indices = []
        # update message field
        #self.progress_bar_info_message.text = "Saved selection to {}".format(filename)
        self.progress_bar_info_message.text = "Saved selection to table cookiecutter_verdicts, vote_id {}".format(vote_id)
        self.progress_bar_data.data["color"] = ["limegreen"]
        # reset scenario field
        self.scenario_name.value = self.country_select.value + " wave calm #"

        ##self.dfVotesContent = self.dfVotesContent.append(df)
        #self.dfVotesContent[self.dfVotesContent.infection_rate_7 > 1000.] = 0.
        ##self.dfVotesContent["filename"] = filename
        ##self.dfVotesContent.to_pickle("./data/votes.pickle",protocol=3)
        conn.close()
        #print(self.dfVotesContent)
        self.compute_metrics()
        i = 0
        for r in self.gumbel_wave_renderers:
            print("Wave {} visible {}".format(i,r.visible))


    def change_dataset(self,attr,old,new):
        sql_query = "SELECT DISTINCT name FROM COOKIECUTTER_CASE_DATA WHERE data_source='{}';".format(new)
        conn = self.engine.connect()
        df = pd.read_sql(sql_query,conn)
        self.country_select.options = sorted(df["name"].values)
        if "Germany" in df["name"].values:
            self.country_select.value = "Germany"
        elif "US-NC North Carolina" in df["name"].values:
            self.country_select.value = "US-NC North Carolina"
        else:
            self.country_select.value = sorted(df["name"].values)[0]
        #self.dataset_select.options = ["Johns Hopkins global","Johns Hopkins US States","ECDC","D RKI"]
        #self.dataset_select.values = "Johns Hopkins global"
        #self.dataset_selected = "Johns Hopkins global"


    def change_country(self,attr,old,new):
        """Handle change of country to be displayed.

        This generates a dict style data structure that can be used to overwrite the various ColumnDataSource s
        that are used to display the values. This is the common bokeh pattern to update a display.

        Commonly made mistake is to re-create a new ColumnDataSource and try to squeeze in a pandas
        dataframe directly, this, however, will not update the plot.
        """
        # fresh data for the time series plots

        sql_query = """SELECT johns_hopkins_data.*,oxford_stringency_index.*
    FROM johns_hopkins_data
    INNER JOIN oxford_stringency_index ON johns_hopkins_data.adm0_a3 = oxford_stringency_index.countrycode AND
    johns_hopkins_data.datetime_date = oxford_stringency_index.datetime_date 
    WHERE johns_hopkins_data.name='{}' AND oxford_stringency_index.regionname IS NULL
    ORDER BY johns_hopkins_data.datetime_date;""".format(new)


        sql_query = """SELECT cookiecutter_case_data.*,oxford_stringency_index.*
    FROM cookiecutter_case_data
    INNER JOIN oxford_stringency_index ON cookiecutter_case_data.identifier = oxford_stringency_index.countrycode AND
    cookiecutter_case_data.datetime_date = oxford_stringency_index.datetime_date 
    WHERE cookiecutter_case_data.name='{}' AND oxford_stringency_index.regionname IS NULL
    AND cookiecutter_case_data.data_source ='{}'
    ORDER BY cookiecutter_case_data.datetime_date;""".format(new,self.dataset_select.value)

        conn = self.engine.connect()
        dfData = pd.read_sql(sql_query, conn)
        # new we have a clone of datetime_date
        dfRubbish = dfData.datetime_date
        del dfData["datetime_date"]
        dfRubbish.columns=["rubbish","datetime_date"]
        #print(dfRubbish)
        #dfData.index = dfRubbish.datetime_date
        dfData["datetime_date"] = dfRubbish.datetime_date
        dfData.index.name = None
        newdata = {}
        #print(self.cds.data.keys())
        for column in self.cds.data.keys():
            #print(column,end=".->.")
            if column == "index" or column == "Timestamp" or column == "datetime_date":
                #newdata[column] = self.adfCountryData[new].index
                newdata[column] = dfData.datetime_date #datetime_date
            else:
                #newdata[column] = np.nan_to_num(self.adfCountryData[new][column])
                newdata[column] = np.nan_to_num(dfData[column])
        self.cds.data = newdata
        #print(dfData.datetime_date)
        #print(self.cds.data["new_cases"])
        #print(self.cds.data)
        # reset scenario text with country name
        #self.scenario_name.value = new+" wave calm #"
        self.scenario_number.value = 1
        self.scenario_type.active = 0
        # rescale the y axes that require it
        #df = self.cds.to_df()
        #self.p_top.extra_y_ranges["active"].end = dfData.active.dropna().values.max()*1.1
        self.p_top.extra_y_ranges["new_cases"].end = dfData.new_cases.dropna().values.max()*1.1

        # now the same thing for the OxCGRT heatmap
        ddf = pd.read_sql("SELECT * FROM oxford_stringency_index WHERE countrycode='{}' AND regionname IS NULL;".format(dfData.identifier.unique()[0]),conn,index_col="datetime_date")
        dfMeasures = pd.DataFrame(ddf[self.fields_of_interest].stack(), columns=["level"]).reset_index().rename(columns={"level_1":"class"})
        newdata = {}
        for column in self.cds_OxCGRTHeatmap.data.keys():
            if column == "index" or column == "Timestamp":
                newdata[column] = dfMeasures.index
                #newdata[column] = dfData.datetime_date
            else:
                newdata[column] = np.nan_to_num(dfMeasures[column])
                #try: ####
                #    newdata[column] = np.nan_to_num(dfData[column])
                #except:
                #    print("MISSING COLUMN {}".format(column))
                #    newdata[column] = [0. for i in dfData.index]
        self.cds_OxCGRTHeatmap.data = newdata

        # now revisit the background boxes, initially make them all invisible
        for b in self.wave_boxes:
            b.visible = False
            b.fill_color = "#F1F1F1"

        # each background box is a BoxAnnotation. Loop through the episode data and color/display them as required.
        ############
        #dfWaves = self.adfWaves[new]
        conn = self.engine.connect()
        dfWaves = pd.read_sql("SELECT * from cookiecutter_computed_waves_chgpoint WHERE name='{}'".format(new),conn)
        conn.close()
        last = "new"
        box_no = 0
        for i,row in dfWaves.iterrows():
            if row["kind"] == "begin":
                left = row["datetime_date"]
                if last == "end":
                    self.wave_boxes[box_no].left = right
                    self.wave_boxes[box_no].right = left
                    self.wave_boxes[box_no].visible = True
                    self.wave_boxes[box_no].fill_color = "#00FF00"
                    self.wave_boxes[box_no].fill_alpha = 0.05
                    box_no += 1
                last = "begin"
            elif row["kind"] == "end":
                right = row["datetime_date"]
                self.wave_boxes[box_no].left = left
                self.wave_boxes[box_no].right = right
                self.wave_boxes[box_no].visible = True
                self.wave_boxes[box_no].fill_color = "#FF0000"
                self.wave_boxes[box_no].fill_alpha = 0.05
                box_no += 1
                last = "end"
        if last == "begin":
            self.wave_boxes[box_no].left = left
            self.wave_boxes[box_no].right = None
            self.wave_boxes[box_no].visible = True
            self.wave_boxes[box_no].fill_color = "#FF0000"
            self.wave_boxes[box_no].fill_alpha = 0.05
        elif last == "end":
            self.wave_boxes[box_no].left = right
            self.wave_boxes[box_no].right = None
            self.wave_boxes[box_no].visible = True
            self.wave_boxes[box_no].fill_color = "#00FF00"
            self.wave_boxes[box_no].fill_alpha = 0.05

        # Ensure possible previous selection is reset
        self.cds.selected.indices = []

        # Compute previous votes
        self.identifier = dfData.identifier.unique()[0]
        conn = self.engine.connect()
        try:
            ddf = pd.read_sql("select DISTINCT from_dt,to_dt,user,kind,vote_id,rel_peak_new_cases,duration from cookiecutter_verdicts WHERE adm0_a3='{}'".format(self.identifier),conn)
        except:
            ddf = pd.DataFrame()
        conn.close()
        #if len(dfVotesContent)>0:
        if len(ddf)>0:
            #ADM0_A3 = self.dfMapping[self.dfMapping.name == new].ADM0_A3.values[0]
            #ddf = dfVotesContent[self.dfVotesContent.ADM0_A3 == ADM0_A3][["from","to","kind"]].copy().drop_duplicates()
            ddf["from"] = pd.to_datetime(ddf["from_dt"])
            ddf["to"] = pd.to_datetime(ddf["to_dt"])
            ddf["color"] = None
            ddf.loc[ddf["kind"] == "Wave","color"] = "tomato"
            ddf.loc[ddf["kind"] == "Calm","color"] = "mediumseagreen"
            ddf["height"] = [random.random()/3+0.1 for i in ddf.index]
            ddf["y"] = [0.5 for i in ddf.index]
            self.cds_votes_ranges.data = {"from":ddf["from"].values,"to":ddf.to.values,"y":ddf.y.values,
                                            "height":ddf.height.values,"color":ddf.color.values}

        
        ddf = dfData[["new_cases","datetime_date"]].copy().fillna(0)
        #ddf.index = ddf.datetime_date
        #ddf.index.name = None

        all_waves = compute_gumbel_waves(ddf,maxwaves=self.max_gumbel_waves)

        newdata = {}#self.cds_gumbel_waves.data
        empty_list = [None for i in range(len(ddf))]
        for k,v in self.cds_gumbel_waves.data.items():
            newdata[k] = empty_list
        newdata["summed"] = [0 for i in range(len(ddf))]
        if len(all_waves)>0:
            i = 0
            for wave in all_waves:
                newdata["datetime"] = wave["datetime_date"]
                newdata["wave_{:02d}".format(i)] = wave["wave_values"]
                newdata["summed"] = list(map(add,newdata["summed"],wave["wave_values"])) # https://stackoverflow.com/questions/18713321/element-wise-addition-of-2-lists
                i += 1
            try:
                newdata["trend"] = dfData[(min(newdata["datetime"])<=dfData["datetime_date"]) & (dfData["datetime_date"] <= max(newdata["datetime"]))]["trend"]
            except:
                pass # strange error
        self.cds_gumbel_waves.data = newdata

        i = 0
        for r in self.gumbel_wave_renderers:
            r.visible = True



    def create(self):
        """The main GUI magic happens here, where the widgets are created and attached to ColumnDataSources, which in turn
        may get updated by user interactions.

        Returns:

        a layout ready to be applied to the bokeh server canvas
        """
        # The progress bar at the top (which cannot function as a progress bar, its really only a status bar)
        self.progress_bar = figure(width=1200,height=25,tools="",x_range=[0.,1.],y_range=[0.,1.])
        self.progress_bar.toolbar.logo = None
        self.progress_bar.axis.visible = False
        self.progress_bar.xgrid.visible = False
        self.progress_bar.ygrid.visible = False
        self.progress_bar_data = ColumnDataSource({"top":[1.],"bottom":[0.],"left":[0.],"right":[1.],"color":["#FF0000"]})
        self.progress_bar.quad(top="top",bottom="bottom",left="left",right="right",source=self.progress_bar_data,color="color")
        self.progress_bar_info_message = Label(x=0.5,y=0.5,text_color="white",text_align="center",text_font="NotoSans",text_baseline="middle",text_font_size="14px",
                                                text="Need to download and process data")
        self.progress_bar.add_layout(self.progress_bar_info_message)

        # the buttons and dropdowns, connecting them to callbacks as required
        self.blank = Paragraph(text=" ",width=175)
        self.refresh_data = Button(label="Load Data",disabled=False,width=150)
        self.refresh_data.on_event(ButtonClick, self.refresh_data_callback)
        conn = self.engine.connect()
        df = pd.read_sql("SELECT DISTINCT data_source FROM cookiecutter_case_data",conn)
        conn.close()
        self.dataset_select=Select(options=sorted(df.data_source.values))
        self.dataset_select.on_change("value",self.change_dataset)
        self.country_select = Select(options=[""])
        self.country_select.on_change("value",self.change_country)
        self.scenario_name = TextInput(value="country wave 2")
        self.scenario_type_label = Paragraph(text="Select Type")
        #self.scenario_type = RadioGroup(labels=["Wave","Calm"],width=100)
        self.scenario_type = RadioButtonGroup(labels=["Wave","Calm"],width=100,active=0)
        self.scenario_number_label = Paragraph(text=" Occurence #")
        self.scenario_number = Spinner(low=1, step=1, value=1,width=50)
        self.save = Button(label="Save",disabled=True,width=150)
        self.save.on_event(ButtonClick, self.save_callback)
        
        # fields of the OxCGRT dataset we actually use
        self.fields_of_interest = ['c1_school closing','c2_workplace closing','c3_cancel public events','c4_restrictions on gatherings',
                              'c6_stay at home requirements','c7_restrictions on internal movement',
                              'c8_international travel controls', 'e1_income support', 'e2_debt/contract relief', 'e3_fiscal measures',
                              'e4_international support', 'h1_public information campaigns','h2_testing policy', 'h3_contact tracing',
                              'h4_emergency investment in healthcare', 'h5_investment in vaccines']
        self.cds_OxCGRTHeatmap = ColumnDataSource(pd.DataFrame({"datetime_date":[],"class":[],"level":[]}))

        # the main time series data source, created empty initially
        empty_data = dict(zip(self.fields_of_interest,[[] for i in range(len(self.fields_of_interest))]))
        for c in ['datetime_date','new_cases','trend','stringencyindex']: 
            empty_data[c] = []
        #self.cds = ColumnDataSource(pd.DataFrame({'date':[], 'confirmed':[], 'deaths':[], 'recovered':[], 'active':[], 'new_cases':[],
        #       'trend':[], 'stringency_index':[],'infection_rate_7':[], 'new_cases_rel':[], 
        #       }))
        #print("self.cds = ColumnDataSource(empty_data)")
        self.cds = ColumnDataSource(empty_data)
        #print(empty_data)
        #print(self.cds.data)

        # May need a different name for "votes", this is the data structure that captures the labelling activity
        self.cds_votes = ColumnDataSource({"name":[],"waves":[],"episodes":[],"votes":[],"need_vote":[]})

        # The backdrops
        self.wave_boxes = [BoxAnnotation(left=None,right=None,visible=False,fill_alpha=0.1,fill_color="#F1F1F1") for i in range(10)]

        # Previous votes
        self.cds_votes_ranges = ColumnDataSource({"from":[],"to":[],"y":[],"height":[],"color":[]})

        # The wave and calm histograms
        self.cds_wave_duration_histogram = ColumnDataSource({"x":[],"y":[],"color":[],"width":[]})
        self.cds_calm_duration_histogram = ColumnDataSource({"x":[],"y":[],"color":[],"width":[]})

        # Heatmap
        self.cds_votes_heatmap = ColumnDataSource({"n":[],"rpc":[],"d":[],"h":[],"w":[]})

        # Gumbel waves
        empty_data = dict(zip(["wave_{:02d}".format(i) for i in range(self.max_gumbel_waves)],[[] for i in range(100)]))
        empty_data["datetime"] = []
        empty_data["trend"] = []
        empty_data["summed"] = []
        self.cds_gumbel_waves = ColumnDataSource(empty_data)


        # The top time series vbar and line plots
        self.p_top = figure(plot_width=1200, plot_height=225,x_axis_type='datetime',title="Select Wave/Calm regions here",
            tools="pan,box_zoom,box_select,reset",active_drag="box_select",
            output_backend="webgl")
        self.p_top.toolbar.logo=None
        self.p_top.yaxis.visible=False

        self.p_top.extra_y_ranges = {"new_cases": Range1d(start=0,end=100),
                                     #"active": Range1d(start=0,end=100)
                                     }
        self.p_top.vbar(x="datetime_date",top="new_cases",source=self.cds,y_range_name="new_cases",
            #width=86400*750,alpha=0.75,color='#ffbb78',legend_label="New Cases")
            width=86400*750,alpha=0.75,color='darkblue',legend_label="New Cases")
        self.p_top.cross(x="datetime_date",y="trend",source=self.cds,y_range_name="new_cases",
                            alpha=0.75,color="#a85232",legend_label="Trend")
        self.p_top.add_layout(LinearAxis(y_range_name="new_cases"), 'left')

        #self.p_top.line(x="datetime_date",y="active",source=self.cds,y_range_name="active",
        #    color="#1f77b4",legend_label="Active Cases")
        #self.p_top.add_layout(LinearAxis(y_range_name="active"), 'left')
        self.p_top.legend.location="top_left"
        self.p_top.legend.click_policy="hide"
        self.p_top.yaxis[1].formatter=PrintfTickFormatter(format="%i")
        try:
            self.p_top.yaxis[2].formatter=PrintfTickFormatter(format="%i")
        except:
            pass

        tooltips_top = [("datetime_date","@date{%Y-%m-%d}"),
                    ("New Cases","@new_cases{0}"),
                    ("New Cases Trend","@trend{0}"),
                    ("Active Cases","@active{0}"),
                    ("Deaths","@deaths{0.00}"),]
        hover_top = HoverTool(tooltips = tooltips_top,
                          formatters={
                              "@date": "datetime"
                              }
                         )
        self.p_top.add_tools(hover_top)

        for i in range(len(self.wave_boxes)):
            self.p_top.add_layout(self.wave_boxes[i])

        # The centre OxCGRT stringency-for-display line plot
        self.p_stringency = figure(plot_width=1200, plot_height=200,x_axis_type='datetime',title="Summary Stringency Index",
            tools="pan,box_zoom,box_select,reset",active_drag="box_select",
            x_range = self.p_top.x_range,y_range=(0,100.),
            output_backend="webgl")
        self.p_stringency.toolbar.logo = None
        self.p_stringency.step(x="datetime_date",y="stringencyindex",source=self.cds,color="#000000",legend_label="Stringency Index")
        #self.p_stringency.legend.location = "top_left"
        #self.p_stringency.legend.label_text_font="NotoSans"
        self.p_stringency.add_tools(hover_top)

        for i in range(len(self.wave_boxes)):
            self.p_stringency.add_layout(self.wave_boxes[i])

        # The OxCGRT detail heatmap
        colors = list(reversed(Greys4)) 
        mapper = LinearColorMapper(palette=colors, low=0, high=4)
        self.p_hmap = figure(plot_width=1200, plot_height=300, x_axis_type="datetime",title="Stringency/Measures Detail",
                            x_range = self.p_top.x_range,
                            y_range=list(reversed(self.fields_of_interest)), toolbar_location=None, tools="", 
                            output_backend="webgl")

        self.p_hmap.rect(y="class", x="datetime_date", width=86400000, height=1, source=self.cds_OxCGRTHeatmap,
                            line_color=None, fill_color=transform('level', mapper))

        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=4),
                             formatter=PrintfTickFormatter(format="%d"))
        color_bar.background_fill_color = self.theme.background_color
        color_bar.major_label_text_color = "#F1F1F1"

        self.p_hmap.add_layout(color_bar, 'right')
        self.p_hmap.axis.axis_line_color = None
        self.p_hmap.axis.minor_tick_line_color = None

        self.p_votes = figure(plot_width=1200, plot_height=75, x_axis_type="datetime",title="Previous Selections/Votes",
                                x_range = self.p_top.x_range,toolbar_location=None, tools="",output_backend="webgl")
        self.p_votes.hbar(left="from",right="to",y="y",height="height",source=self.cds_votes_ranges,
                            color="color",fill_alpha=0.2,line_color=None)
        self.p_votes.yaxis.visible = False
        self.p_votes.ygrid.visible = False
        #self.p_votes.xaxis.visible = False


        self.p_gumbel = figure(plot_width=1200, plot_height=275,x_axis_type='datetime',title="Possible Waves (Gumbel fits)",
            tools="pan,box_zoom,box_select,reset",active_drag="box_select",
            x_range = self.p_top.x_range,output_backend="webgl")
        self.p_gumbel.toolbar.logo=None
        #self.p_gumbel.yaxis.visible=False
        r0 = self.p_gumbel.cross(x="datetime_date",y="trend",source=self.cds,alpha=0.75,color="#a85232")
        r1 = self.p_gumbel.x(x="datetime",y="summed",source=self.cds_gumbel_waves,alpha=0.75,color="#000000")
        self.p_gumbel.add_layout(Legend(items=[("trend", [r0]),
            ("summed gumbel waves", [r1])],location="top_left"))

        #r = [None for i in range(self.max_gumbel_waves)]
        legend_items = []
        stack_y = []
        colors = []
        for i in range(self.max_gumbel_waves):
            stack_y.append("wave_{:02d}".format(i))
            colors.append(Colorblind8[i%len(Colorblind8)])
        self.gumbel_wave_renderers = self.p_gumbel.varea_stack(stack_y,x="datetime",source=self.cds_gumbel_waves,color=colors,fill_alpha=0.25)
        for i in range(self.max_gumbel_waves):
            legend_items.append(("{:02d}".format(i), [self.gumbel_wave_renderers[i]]))
        self.p_gumbel.add_layout(Legend(items=legend_items,click_policy="hide",orientation="horizontal",label_text_font_size="6pt",
            label_height=1,label_standoff=1,label_text_line_height=1,margin=1,spacing=0,
            background_fill_color="#252425",label_text_color="#FFFFFF",padding=0),"below")


        for i in range(len(self.wave_boxes)):
            self.p_hmap.add_layout(self.wave_boxes[i])

        # The data table that displays stats for countries and how many datasets have been labelled
        columns = [
            TableColumn(field="name",title="Country Name"),
            TableColumn(field="waves",title="Waves"),
            TableColumn(field="episodes",title="Eposides"),
            TableColumn(field="votes",title="Votes"),
            TableColumn(field="need_vote",title="Need Vote"),
            ]

        self.voting_table = DataTable(source=self.cds_votes,columns=columns,width=400,height=800)

        # Votes metrics: Wave Durations
        self.p_histogram_wave = figure(tools="", background_fill_color="#efefef", toolbar_location=None, width=300, height=300,
                                        title="Wave Durations",x_axis_label="Days",y_axis_label="Votes",output_backend="webgl")
        self.p_histogram_wave.vbar(x="x",top="y",width="width",color="color",source=self.cds_wave_duration_histogram)

        # Votes metrics: Calm Durations
        self.p_histogram_calm = figure(tools="", background_fill_color="#efefef", toolbar_location=None, width=300, height=300,
                                        title="Calm Durations",x_axis_label="Days",y_axis_label="Votes",output_backend="webgl")
        self.p_histogram_calm.vbar(x="x",top="y",width="width",color="color",source=self.cds_calm_duration_histogram)

        # Heatmap
        self.p_duration_heatmap = figure(plot_width=300, plot_height=300,x_axis_label="Rel. Peak of New Cases [#/100000]",
                                        y_axis_label="Peak Duration [d]",toolbar_location=None, tools="",
                                        title="Relation of Wave Peak to Duration",output_backend="webgl")
        colors = Plasma256
        mapper = LinearColorMapper(palette=colors)
        self.p_duration_heatmap.rect("rpc","d",source=self.cds_votes_heatmap,fill_color=transform("n", mapper),height="h",width="w",line_color=None)


        # squeezing the color scheme into the widgets
        self.progress_bar = self.theme.apply_theme_defaults(self.progress_bar)
        self.country_select = self.theme.apply_theme_defaults(self.country_select)
        self.scenario_name = self.theme.apply_theme_defaults(self.scenario_name)
        self.save = self.theme.apply_theme_defaults(self.save)
        self.p_top = self.theme.apply_theme_defaults(self.p_top)
        self.p_top.background_fill_color = "#ffffff"
        self.p_stringency = self.theme.apply_theme_defaults(self.p_stringency)
        self.p_stringency.background_fill_color = "#ffffff"
        self.p_hmap = self.theme.apply_theme_defaults(self.p_hmap)
        self.p_hmap.background_fill_color = "#ffffff"
        self.p_gumbel = self.theme.apply_theme_defaults(self.p_gumbel)
        self.p_gumbel.background_fill_color = "#ffffff"
        self.voting_table = self.theme.apply_theme_defaults(self.voting_table)
        self.p_votes = self.theme.apply_theme_defaults(self.p_votes)
        self.p_votes.background_fill_color = "#ffffff"


        self.p_histogram_wave = self.theme.apply_theme_defaults(self.p_histogram_wave)
        self.p_histogram_wave.background_fill_color = "lightsteelblue"
        self.p_histogram_calm = self.theme.apply_theme_defaults(self.p_histogram_calm)
        self.p_histogram_calm.background_fill_color = "lightsteelblue"
        self.p_duration_heatmap = self.theme.apply_theme_defaults(self.p_duration_heatmap)
        self.p_duration_heatmap.background_fill_color = "lightsteelblue"

        self.user_id = TextInput(value="nobody@{}".format(socket.gethostname()),title="Name to save your results")

        # The return value is a row/column/widget array that defines the arrangement of the various elements.
        return(row([column([self.progress_bar,
                            row([self.blank,
                                self.refresh_data,
                                self.dataset_select,
                                self.country_select,
                                self.scenario_type_label,
                                self.scenario_type,
                                self.scenario_number_label,
                                self.scenario_number,
                                self.save]),
                        self.p_top,
                        self.p_stringency,
                        self.p_hmap,
                        self.p_gumbel,
                        self.p_votes,
                            row([Div(text='<div class="horizontalgap" style="width:200px"><h2>Statistics</h2></div>'),
                                self.p_histogram_wave,self.p_histogram_calm,self.p_duration_heatmap]),
                        ]),
                    column([self.user_id,
                        self.voting_table]),
                    ]))


    def check_and_create_reference_data(self):
        if not self.engine.dialect.has_table(self.engine,"johns_hopkins_country_mapping"):
            conn = self.engine.connect()
            self.dfMapping = pd.read_csv("https://github.com/rolls-royce/EMER2GENT/raw/master/data/sun/geo/country_name_mapping.csv",low_memory=False)
            self.dfMapping.to_sql("johns_hopkins_country_mapping", conn, if_exists='replace',dtype={'ADM0_A3':sqlalchemy.types.String(3),
                                                                                  'name':sqlalchemy.types.String(150),
                                                                                  'ISO_3_code_i':sqlalchemy.types.Integer},index=False)
            #print(dfMapping)
            conn.close()

        if not self.engine.dialect.has_table(self.engine,"un_population_data_2020_estimates"):
            conn = self.engine.connect()
            dfPopulationRaw = pd.read_excel("https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/1_Population/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx",
                            sheet_name="ESTIMATES",skiprows=16,usecols="E,BZ")
            alldata = []
            for i,row in dfPopulationRaw.iterrows():
                try:
                    result = pycountry.countries.get(numeric="{:03d}".format(row["Country code"]))
                except:
                    print(row["Country code"],end="..")
                    continue
                if result:
                    alldata.append({"ADM0_A3":result.alpha_3,"population":row["2020"]*1000,"name":result.name})
                else:
                    try:
                        result = pycountry.countries.search_fuzzy(row["Region, subregion, country or area *"])
                        print(row["Country code"],result,end="..")
                        alldata.append({"ADM0_A3":result.alpha_3,"population":round(row["2020"]*1000),"name":result.name})
                    except:
                        continue
            self.dfPopulation = pd.DataFrame(alldata)
            self.dfPopulation.to_sql("un_population_data_2020_estimates", conn, if_exists='replace',dtype={'ADM0_A3':sqlalchemy.types.String(3),
                                                                                  'name':sqlalchemy.types.String(150),
                                                                                  'ISO_3_code_i':sqlalchemy.types.Integer},index=False)
            #print(dfPopulation)
            conn.close()



    def download_data(self):
        """This downloads directly, from github, the Johns Hopkins and Oxford university data sets. Not error handling is performed
        as it is unclear as to how to proceed in these cases (and no existing data would be overwritten)

        Note this code still contains progress bar code which will not cause visual feedback, as this code
        is called during a callback handling routine which prevents GUI updates. I left it in as this could change
        in a future version breaking the agile rule of no provision for future growth and the sensible rule
        of not having untested code.
        """
        print("check_and_create_reference_data")
        self.check_and_create_reference_data()

        conn = self.engine.connect()


        print("Johns Hopkins Global...")
        dfJH = pd.read_csv("https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv?raw=true",
               encoding="utf-8")

        dfMapping = pd.read_sql("SELECT * FROM johns_hopkins_country_mapping",conn)

        dfJHcountries = dfJH[dfJH["Province/State"].isnull()]
        dfJHcountries = dfJHcountries.merge(dfMapping,left_on="Country/Region",right_on="name")
        dfJHcountries = dfJHcountries[dfJHcountries["iso_3_code_i"]>0]
        del dfJHcountries["Lat"]
        del dfJHcountries["Long"]
        del dfJHcountries["Province/State"]
        del dfJHcountries["Country/Region"]
        del dfJHcountries["adm0_a3"]
        del dfJHcountries["iso_3_code_i"]

        dfJHcountries.index = dfJHcountries.name
        dfJHcountries.index.name = None
        del dfJHcountries["name"]
        dfJHcountries = dfJHcountries.transpose()
        dfJHcountries.index = pd.to_datetime(dfJHcountries.index)
        dfJHcountries = dfJHcountries.diff(1).dropna()

        dfJHcountries_trend = dfJHcountries.copy()
        for c in dfJHcountries_trend.columns:
            dfJHcountries_trend[c] = seasonal_decompose(dfJHcountries_trend[c],period=7).trend

        dfnew_cases = dfJHcountries.stack().reset_index().rename(columns={"level_0":"datetime_date","level_1":"name",0:"new_cases"})
        dftrend = dfJHcountries_trend.stack().reset_index().rename(columns={"level_0":"datetime_date","level_1":"name",0:"trend"})
        df = pd.merge(dfnew_cases,dftrend,how="left",left_on=["datetime_date","name"],right_on=["datetime_date","name"])
        df["new_cases"] = df["new_cases"].astype(int)
        df["data_source"] = "Johns Hopkins global"

        try:
            conn.execute("DELETE FROM cookiecutter_case_data WHERE source='Johns Hopkins global'")
        except:
            pass
        df.to_sql("cookiecutter_case_data",conn,index=False,dtype={"datetime_date":sqlalchemy.types.DateTime,
                                                                  "name":sqlalchemy.types.VARCHAR(100),
                                                                   "data_source":sqlalchemy.types.VARCHAR(50)},
                 if_exists="append")

        allwaves = []
        for c in dfJHcountries.columns:
            allwaves.append(compute_changepoint_waves(dfJHcountries[c],country=c,ADM0_A3=dfMapping[dfMapping.name == c].adm0_a3.unique()[0]))
        dfWaves = pd.DataFrame().append(allwaves)
        dfWaves["wave_no"] = dfWaves["wave_no"].astype(int)
        dfWaves["data_source"] = "Johns Hopkins global"
        try:
            conn.execute("DELETE FROM cookiecutter_computed_waves_chgpoint WHERE source='Johns Hopkins global'")
        except:
            pass
        print(dfWaves)
        dfWaves.to_sql("cookiecutter_computed_waves_chgpoint",conn,index=False,dtype={"name":sqlalchemy.types.VARCHAR(100),
                                                                                     "datetime_date":sqlalchemy.types.DateTime,
                                                                                     "kind":sqlalchemy.types.VARCHAR(10),
                                                                                     "adm0_a3":sqlalchemy.types.VARCHAR(10),
                                                                                     "data_source":sqlalchemy.types.VARCHAR(50)},
                 if_exists="append")


        print("Johns Hopkins US States...")

        dfJHUS = pd.read_csv("https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv?raw=true",
                       encoding="utf-8")
        dfJHUS = dfJHUS[dfJHUS.FIPS.notnull()]

        # chosen to hard code this as it is a very limited dataset and very stable
        state_to_postal = dict(zip(['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia',
                  'Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts',
                  'Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey',
                  'New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island',
                  'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia',
                  'Wisconsin','Wyoming','American Samoa','Guam','Northern Mariana Islands','Puerto Rico','Virgin Islands'],
                 ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
                  'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
                  'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','AS','GU','MP','PR','VI']))

        dfJHUS["name"] = ""
        for i,row in dfJHUS.iterrows():
            try:
                dfJHUS.at[i,"name"] = "US-"+state_to_postal[row["Province_State"]]+" "+row["Province_State"]
            except:
                continue

        dfJHUS = dfJHUS[dfJHUS.name > ""].groupby("name").sum()
        del dfJHUS["UID"]
        del dfJHUS["code3"]
        del dfJHUS["FIPS"]
        del dfJHUS["Lat"]
        del dfJHUS["Long_"]
        #del dfJHUS["Province_State"]
        #dfJHUS.index = dfJHUS.name
        dfJHUS.index.name = None
        dfJHUS = dfJHUS.transpose()
        dfJHUS.index = pd.to_datetime(dfJHUS.index)
        dfJHUS = dfJHUS.diff(1).dropna()

        dfJHUS_trend = dfJHUS.copy()
        for c in dfJHUS_trend.columns:
            dfJHUS_trend[c] = seasonal_decompose(dfJHUS_trend[c],period=7).trend
        df = pd.merge(dfnew_cases,dftrend,how="left",left_on=["datetime_date","name"],right_on=["datetime_date","name"])


        dfnew_cases = dfJHUS.stack().reset_index().rename(columns={"level_0":"datetime_date","level_1":"name",0:"new_cases"})
        dftrend = dfJHUS_trend.stack().reset_index().rename(columns={"level_0":"datetime_date","level_1":"name",0:"trend"})
        df = pd.merge(dfnew_cases,dftrend,how="left",left_on=["datetime_date","name"],right_on=["datetime_date","name"])
        df["new_cases"] = df["new_cases"].astype(int)
        df["data_source"] = "Johns Hopkins US States"

        try:
            conn.execute("DELETE FROM cookiecutter_case_data WHERE source='Johns Hopkins US States'")
        except:
            pass
        df.to_sql("cookiecutter_case_data",conn,index=False,dtype={"datetime_date":sqlalchemy.types.DateTime,
                                                                  "name":sqlalchemy.types.VARCHAR(100),
                                                                   "data_source":sqlalchemy.types.VARCHAR(50)},
                 if_exists="append")

        allwaves = []
        for c in dfJHUS.columns:
            allwaves.append(compute_changepoint_waves(dfJHUS[c],country=c,ADM0_A3=c.split(" ")[0]))
        dfWaves = pd.DataFrame().append(allwaves)
        dfWaves["wave_no"] = dfWaves["wave_no"].astype(int)
        dfWaves["data_source"] = "Johns Hopkins US States"

        try:
            conn.execute("DELETE FROM cookiecutter_computed_waves_chgpoint WHERE source='Johns Hopkins US States'")
        except:
            pass
        dfWaves.to_sql("cookiecutter_computed_waves_chgpoint",conn,index=False,dtype={"name":sqlalchemy.types.VARCHAR(100),
                                                                                     "datetime_date":sqlalchemy.types.DateTime,
                                                                                     "kind":sqlalchemy.types.VARCHAR(10),
                                                                                     "adm0_a3":sqlalchemy.types.VARCHAR(10),
                                                                                     "data_source":sqlalchemy.types.VARCHAR(50)},
                 if_exists="append")

        conn.close()
        print("Finished download")

                

    def process_data(self):
        """The heavy lifting of  processing the infection numbers of Johns Hopkins and the OxCGRT data.
        """
        # This look slike a duplicate
        # TODO: cleanup
        self.fields_of_interest = ['C1_School closing','C2_Workplace closing','C3_Cancel public events','C4_Restrictions on gatherings',
        'C6_Stay at home requirements','C7_Restrictions on internal movement', 
        'C8_International travel controls', 'E1_Income support', 'E2_Debt/contract relief', 'E3_Fiscal measures',
        'E4_International support', 'H1_Public information campaigns','H2_Testing policy', 'H3_Contact tracing',
        'H4_Emergency investment in healthcare', 'H5_Investment in vaccines']
        monetary_fields = ['E3_Fiscal measures','E4_International support','H4_Emergency investment in healthcare', 'H5_Investment in vaccines']

        # To reuse some old code, the data are stored in dicts of dataframes, one per country.
        self.adfCountryData = {}
        self.adfOxCGRT = {}
        self.adfMeasures = {}
        self.adfWaves = {}
        counter = 0
        num_countries = len(self.dfConfirmed["Country/Region"].unique())
        country_number = 1
        #next line is troubleshooting/speedup code that could be removed but I leave it in for now
        #TODO consider removal
        #for country in ["Germany","Australia","Austria","United Kingdom","France","Italy"]:
        for country in self.dfConfirmed["Country/Region"].unique():
            try:
                ADM0_A3 = self.dfMapping[self.dfMapping.name == country].ADM0_A3.values[0]
            except:
                continue # cannot use data we have no ISO3 country code for
            if ADM0_A3 == "***": # invalid as per https://github.com/rolls-royce/EMER2GENT/blob/master/data/sun/geo/country_name_mapping.csv 
                continue
            #print("{} {}".format(ADM0_A3,country)) # poor man's progress bar

            # Step 1 Confirmed cases, data comes in an odd time series format with the columns being the time series
            dfCountry = self.dfConfirmed[self.dfConfirmed["Country/Region"] == country].transpose()
            columns = list(dfCountry.columns)
            dfCountry["date"] = pd.to_datetime(dfCountry.index,errors="coerce")
            dfCountry = dfCountry.dropna()
            dfCountry["confirmed"] = dfCountry[columns].sum(axis=1).astype(int)
            for c in columns:
                del dfCountry[c]
        
            # Step 2, recovered, also needs to be transposed
            ddf = self.dfDeaths[self.dfDeaths["Country/Region"] == country].transpose()
            columns = list(ddf.columns)
            ddf["date"] = pd.to_datetime(ddf.index,errors="coerce")
            ddf = ddf.dropna()
            ddf["deaths"] = ddf[columns].sum(axis=1).astype(int)
        
            for c in columns:
                del ddf[c]
        
            dfCountry = dfCountry.join(ddf,rsuffix = "_tmp")
            del dfCountry["date_tmp"]
        
            # Step 3, recovered, also needs to be transposed
            ddf = self.dfRecovered[self.dfRecovered["Country/Region"] == country].transpose()
            columns = list(ddf.columns)
            ddf["date"] = pd.to_datetime(ddf.index,errors="coerce")
            ddf = ddf.dropna()
            ddf["recovered"] = ddf[columns].sum(axis=1).astype(int)
        
            for c in columns:
                del ddf[c]
        
            dfCountry = dfCountry.join(ddf,rsuffix = "_tmp")
            del dfCountry["date_tmp"]
       
            # Cleanup to improve numerical stability, although the semantics of NaN and 0 are, of course, different.
            # Else, scaling of axes will not work as min/max gets confused with NaN. 
            dfCountry.replace([np.inf, -np.inf], np.nan).fillna(0.,inplace=True)
        
            # Some basic computations
            dfCountry["active"] = dfCountry.confirmed-dfCountry.deaths-dfCountry.recovered
            dfCountry["active"] = dfCountry["active"].clip(lower=0)
            dfCountry["new_cases"] = dfCountry.confirmed.diff()
            dfCountry["infection_rate_7"] = dfCountry[["active"]].pct_change(periods=7)
            dfCountry["trend"] = seasonal_decompose(dfCountry[["new_cases"]].fillna(0).values,period=7).trend

            # ADM0_A3 is a synonym for ISO 4166 Alpha 3
            dfCountry["ADM0_A3"] = ADM0_A3
            population = self.dfPopulation[self.dfPopulation.ADM0_A3==ADM0_A3].population.values[0]
            #print("{} POPULATION {}".format(ADM0_A3,population))
            dfCountry["active_rel"] = dfCountry["active"]/population*100000
            dfCountry["new_cases_rel"] = dfCountry["new_cases"]/population*100000

            ddfOxCGRT = self.dfOxCGRT[self.dfOxCGRT.CountryCode == ADM0_A3].copy().rename(columns={"StringencyLegacyIndexForDisplay":"stringency_index"})
            if len(ddfOxCGRT) <= 0:
                continue
            ddfOxCGRT.index = pd.to_datetime(ddfOxCGRT.Date,format="%Y%m%d")
            ddfOxCGRT = ddfOxCGRT[ddfOxCGRT.RegionCode.isnull()]
            ddfOxCGRT = ddfOxCGRT.fillna(0)

            # This step is actually only computing state changes in OxCGRT which is not used right now
            alldata = []
            previous_data = dict(zip(self.fields_of_interest,[0 for i in range(len(self.fields_of_interest))]))
            previous_data["datetime"] = pd.to_datetime("2019-12-31")
            for i,row in ddfOxCGRT.iterrows():
                new_data = dict(zip(self.fields_of_interest,[0 for i in range(len(self.fields_of_interest))]))
                for column in self.fields_of_interest:
                    if row[column]:
                        data = int(row[column])
                    else:
                        data = 0
                    if column in monetary_fields:
                        if data != 0:
                            data = 1
                    if previous_data[column] != data:
                        alldata.append({"country":country,"ADM0_A3":ADM0_A3,"measure":column,"new_measure_value":data,"new":"{}#{}".format(column[:2],data),
                                        "old_measure_value":previous_data[column],"old":"{}#{}".format(column[:2],previous_data[column]),
                                        #"Timestamp":"{:%Y-%m-%d}".format(i),"stringency_index":row["StringencyLegacyIndexForDisplay"],
                                        "Timestamp":"{:%Y-%m-%d}".format(i),"stringency_index":row["stringency_index"],
                                        "id":country_number})
                    new_data[column] = data
                new_data["datetime"] = i
                previous_data = new_data
            country_number += 1
            dfTemp = pd.DataFrame(alldata)
            self.adfOxCGRT[country] = dfTemp

            # For the monetary value fields in OxCGRT use 1 as a non-zero flag (else the heatmap gets confused by $$$$llions
            # of support to economy and the public
            for i,row in ddfOxCGRT.iterrows():
                for m in monetary_fields:
                    if row[m] != 0:
                        ddfOxCGRT.at[i,m] = 1

            # the data for the heatmap
            dfMeasures = pd.DataFrame(ddfOxCGRT[self.fields_of_interest].stack(), columns=["level"]).reset_index().rename(columns={"Date":"date","level_1":"class"})
            self.adfMeasures[country] = dfMeasures
            # the numerical data
            dfCountry = dfCountry.join(ddfOxCGRT[["stringency_index"]])
            dfCountry = dfCountry.join(ddfOxCGRT[self.fields_of_interest])
            # storing it in the dict of datafranes
            self.adfCountryData[country] = dfCountry

            # computing waves and "periods of calmness" using a very manual Schmitt-Trigger style detection of gradients up and down
            all_verdicts = []
            field = "trend"
            THRESHOLD = 1
            THRESHOLD_UP = 14
            THRESHOLD_DOWN = 28
            dfCountry.index = dfCountry["date"]

            ddf = dfCountry[[field]].rolling(center=True,window=7).mean().dropna()
            ddf["pct_change"] = ddf.pct_change()

            datum = ddf[field].values[0]
            increasing = 0
            decreasing = 0
            wave_no = 0
            for i,row in ddf[1:].iterrows():
                if row[field] > datum:
                    if increasing == 0:
                        start_date = i
                    increasing += 1
                    if increasing > 3:
                        decreasing = 0
                elif row[field] < datum:
                    decreasing += 1
                    if decreasing > 3:
                        increasing = 0
                    
                if increasing == THRESHOLD_UP:
                    wave_no += 1
                    if len(all_verdicts)>0 and all_verdicts[-1]["kind"] == "begin":
                        pass
                    else:
                        all_verdicts.append({"country":country,"date":i,"kind":"begin","wave_no":wave_no})
                if decreasing == THRESHOLD_DOWN:
                    if len(all_verdicts)>0 and all_verdicts[-1]["kind"] == "end":
                        all_verdicts.pop()
                        all_verdicts.append({"country":country,"date":i,"kind":"end","wave_no":wave_no})
                    else:
                        all_verdicts.append({"country":country,"date":i,"kind":"end","wave_no":wave_no})
                datum = row[field]

            if len(all_verdicts) > 0:
                dfWaves = pd.DataFrame(all_verdicts)
                dfWaves = dfWaves.sort_values(["country","date"])
                self.adfWaves[country] = dfWaves
            else:
                self.adfWaves[country] = pd.DataFrame({"country":[],"date":[],"kind":[],"wave_no":[]})

        # The data are stored in a somewhat horrifying format which I want to apologize for. This is due to the
        # dict of dataframes idea that came from very early work which allowed for some code reuse.
        """self.blob = {}
        with gzip.open("data/datafile.pckld.gz","w+t") as f:
            self.blob["data"] =base64.b64encode(pickle.dumps(self.adfCountryData,protocol=4)).decode("ascii")
            self.blob["mapping"] = base64.b64encode(pickle.dumps(self.dfMapping,protocol=4)).decode("ascii")
            self.blob["stringency"] = base64.b64encode(pickle.dumps(self.adfOxCGRT,protocol=4)).decode("ascii")
            self.blob["measures"] = base64.b64encode(pickle.dumps(self.adfMeasures,protocol=4)).decode("ascii")
            self.blob["waves"] = base64.b64encode(pickle.dumps(self.adfWaves,protocol=4)).decode("ascii")
            self.blob["population"] = base64.b64encode(pickle.dumps(self.dfPopulation,protocol=4)).decode("ascii")
            data = json.dumps(self.blob)
            f.write(data)"""


    def compute_data_status(self):
        """Determine if data are stale, or don't even exist. Staleness is determined by a "last download" item.
        """
        #if not self.engine.dialect.has_table(self.engine,"johns_hopkins_data"):
        #    self.data_status = "no_data"
        #    message_text = "Could not find data file, press Load Data"
        #    color = "#FFi0000"
        #else:
        try:
            conn = self.engine.connect()
            result = conn.execute("SELECT MAX(datetime_date) FROM cookiecutter_case_data")
            latest_datapoint = result.fetchone()[0]
            data_age = datetime.datetime.now()-pd.to_datetime(latest_datapoint)
            if data_age.days > 2:
                self.data_status = "stale"
                message_text = "Your data are {:.1f} days old, consider reloading it".format(data_age.days+data_age.seconds/86400)
                color = "darkorange"
            else:
                self.data_status = "current"
                message_text = "You have current data, which are {:.1f} days old".format(data_age.days+data_age.seconds/86400)
                color = "limegreen"
        except:
            self.data_status = "no_data"
            message_text = "Could not find data file, press Load Data"
            color = "tomato"

        """datafile = "data/datafile.pckld.gz"
        if os.path.exists(datafile):
            with gzip.open(datafile,"rb") as f:
                self.blob = json.loads(f.read())
            data_age = datetime.datetime.now()-pd.to_datetime(self.blob["last_update"])
            if data_age.days > 1:
                self.data_status = "stale"
                message_text = "Your data are {:.1f} days old, consider reloading it".format(data_age.days+data_age.seconds/86400)
                color = "#FFBF00"
            else:
                self.data_status = "current"
                message_text = "You have current data, which are {:.1f} days old".format(data_age.days+data_age.seconds/86400)
                color = "#00FF00"
        else:
            self.data_status = "no_data"
            message_text = "Could not find data file, press Load Data"
            color = "#FFi0000"
        """

        self.progress_bar_info_message.text = message_text
        self.progress_bar_data.data["color"] = [color]

    def sort_countries_by_relevance(self):
        """Work in progress on how to display the countries in a relevant oder in the dropdown. Alphabetically may cause only the countries A-G to
        ever be voted on.... Here we use the relative percentage growth of infections compared to a week ago.
        """
        score = {}
        for country in self.country_select.options:
            score[country] = self.adfCountryData[country].infection_rate_7.values[-1]
        score_sorted = {k: v for k, v in sorted(score.items(), key=lambda item: item[1],reverse=True)}
        self.country_select.options = list(score_sorted.keys())
        print("SORTED")
        

    def load_data(self):
        """Loading the data but also, as a temprary fix, checking which user files can be found in cookiecutter/data/*.csv
        TODO: The latter needs to be ported to SQL
        """
        print(self.data_status)
        if self.data_status == "no_data":
            self.dfVotesContent = pd.DataFrame()
            return
        else:
            """with gzip.open("data/datafile.pckld.gz","rt") as f:
                data = f.read()
                data = json.loads(data)
                self.adfCountryData = pickle.loads(base64.b64decode(data["data"]))
                self.dfMapping = pickle.loads(base64.b64decode(data["mapping"]))
                self.adfOxCGRT = pickle.loads(base64.b64decode(data["stringency"]))
                self.adfMeasures = pickle.loads(base64.b64decode(data["measures"]))
                self.adfWaves = pickle.loads(base64.b64decode(data["waves"]))
                self.dfPopulation = pickle.loads(base64.b64decode(data["population"]))"""

            conn = self.engine.connect()

            df = pd.read_sql("select distinct data_source from cookiecutter_case_data;",conn)
            self.dataset_select.options = sorted(df.data_source.values)
            self.dataset_select.value = "Johns Hopkins global"
            print("DATASETS {}".format(self.dataset_select.options ))

            #df = pd.read_sql("select distinct name,infection_rate_7 from johns_hopkins_data order by infection_rate_7 DESC NULLS LAST;",conn)
            #df = pd.read_sql("""SELECT distinct name,infection_rate_7 FROM johns_hopkins_data AS a WHERE datetime_date = ( SELECT MAX(datetime_date) FROM johns_hopkins_data AS b ) order by infection_rate_7 DESC NULLS LAST""",conn)
            df = pd.read_sql("SELECT DISTINCT name FROM cookiecutter_case_data  WHERE data_source='Johns Hopkins global' ORDER BY name;",conn)
            #print(df)
            self.country_select.options = list(df.name.values)
            #self.country_select.options=sorted(self.adfCountryData.keys())
            #self.sort_countries_by_relevance()

            dfComputed_waves = pd.read_sql("SELECT * FROM cookiecutter_computed_waves_chgpoint  WHERE data_source='Johns Hopkins global'", conn)
            dfComputed_waves_stats = pd.DataFrame(dfComputed_waves[dfComputed_waves.kind=="begin"].groupby("identifier").size()).rename(columns={0:"waves"})
            dfComputed_waves_stats = dfComputed_waves_stats.join(pd.DataFrame(dfComputed_waves[dfComputed_waves.kind=="end"].groupby("identifier").size()).rename(columns={0:"episodes"}))
            dfComputed_waves_stats = dfComputed_waves_stats.fillna(0)
            dfComputed_waves_stats.episodes = dfComputed_waves_stats.episodes.astype("int")

            try:
                dfVerdicts = pd.read_sql("SELECT * FROM cookiecutter_verdicts  WHERE data_source='Johns Hopkins global'", conn)
            except:
                dfVerdicts = pd.DataFrame({"kind":[],"kind_counter":[],"identfier":[],"user":[]})
            dfVerdicts_stats = pd.DataFrame(dfVerdicts[["kind","kind_counter","identfier","user"]].drop_duplicates().groupby("identfier").size()).rename(columns={0:"votes"})
            dfVotesContent = dfVerdicts_stats.join(dfComputed_waves_stats).fillna(0)
            dfVotesContent.waves = dfVotesContent.waves.astype("int")
            dfVotesContent.episodes = dfVotesContent.episodes.astype("int")
            dfVotesContent["need_vote"] = dfVotesContent.waves+dfVotesContent.episodes > dfVotesContent.votes

            dfMapping = pd.read_sql("SELECT DISTINCT name,identifier from cookiecutter_case_data  WHERE data_source='Johns Hopkins global'",conn,index_col="identifier")
            conn.close()

            if len(dfVotesContent)>0:
                dfVotesContent = dfVotesContent.join(dfMapping)
                self.cds_votes.data = {"name":dfVotesContent.name.values,
                    "waves":dfVotesContent.waves.values,
                    "episodes":dfVotesContent.episodes.values,
                    "votes":dfVotesContent.votes.values,
                    "need_vote":dfVotesContent.need_vote.values}
            else:
                self.cds_votes.data = {"name":[],
                    "waves":[],
                    "episodes":[],
                    "votes":[],
                    "need_vote":[]}

            self.compute_metrics()
            #self.country_select.value = "Germany"
            self.cds.selected.on_change('indices',self.on_selection_change_callback)
        pass


class GUIEconomy():
    def __init__(self):
        self.theme = THEME()
        self.data_status = "no_data" # "stale", "current"
        if "SQL_CONNECT" not in list(os.environ.keys()):
            sql_url = "postgresql://cookiecutter:cookiecutter@database:5432/cookiec" #"sqlite:///database.sqlite"
            #sql_url = "db2+ibm_db://db2inst1:cookiecutter@192.168.100.197:50000/cookiec"
        else:
            sql_url = os.environ["SQL_CONNECT"]
        #print("SQL_CONNECT {}".format(sql_url))
        self.engine = create_engine(sql_url)
        print(self.engine)
        self.add_point_guard = False
        pass


    def change_category(self,attr,old,new):
        #print(old,new)
        self.get_keys(category=new)
        if self.key_select.disabled:
            self.key_select.value = ""
        else:
            self.key_select.value = self.key_select.options[0]
        #print(self.key_select.options)


    def get_categories(self):
        conn = self.engine.connect()
        result = conn.execute("SELECT DISTINCT category FROM economic_indicators;")
        categories = []
        categories.extend([c[0] for c in result.fetchall()])
        conn.close()
        #print(categories)
        self.category_select.options=categories
        self.cds_proxy_data.data = {"datetime":[],"value":[]}


    def change_key(self,attr,old,new):
        if len(new) <= 0:
            print("Zero length key")
            return
        else:
            print("CHANGE KEY TO ",new)
        category = self.category_select.value
        key = new
        conn = self.engine.connect()
        df = pd.read_sql("SELECT datetime_date,parameter_value FROM economic_indicators WHERE category='{}' and parameter_name='{}' ORDER BY datetime_date;".format(category,key),conn)
        print("CHANGE_KEY from '{}' to '{}'".format(old,new))
        #print(df)
        self.cds_proxy_data.data = {"datetime":df["datetime_date"].values,"value":df["parameter_value"].values}
        self.p_values.title.text = "{} - {}".format(category,key)
        self.p_values.x_range=DataRange1d(pd.to_datetime(self.start_date.value).date(),pd.to_datetime(self.end_date.value).date())
        #self.value_axis.bounds=DataRange1d(df.value.min(),df.value.max())
        value_range = df["parameter_value"].max()-df["parameter_value"].min()
        self.p_values.extra_y_ranges["value"].start = df["parameter_value"].min()-value_range*0.05
        self.p_values.extra_y_ranges["value"].end = df["parameter_value"].max()+value_range*0.05
        self.p_values.yaxis[1].axis_label = new
        df = pd.read_sql("SELECT DISTINCT explanation FROM economic_indicators WHERE category='{}' and parameter_name='{}';".format(category,key),conn)
        conn.close()
        url_shown = df["explanation"].values[0]
        url = "https://translate.google.com/translate?hl=en&sl=auto&tl=en&u={}".format(urllib.parse.quote_plus(url_shown))
        self.explanation.text="<H1>{}</H1><H2>{}</H2>See <A HREF=\"{}\" style=\"color:#DDDDDD;\">{}</A> for more details".format(category,key,url,url_shown)


    def get_keys(self,category=""):
        if category == "":
            category = self.category_select.value
        conn = self.engine.connect()
        result = conn.execute("SELECT DISTINCT parameter_name FROM economic_indicators WHERE category='{}';".format(category))
        keys = []
        keys.extend([k[0] for k in result.fetchall()])
        print("KEYS {}".format(keys))
        conn.close()
        if len(keys) <= 0:
            self.key_select.options=["<select catgory first>"]
            #self.key_select.value=["<select catgory first>"]
            self.key_select.disabled = True
        else:
            self.key_select.options = keys
            self.key_select.disabled = False


    def load_data(self):
        if "TSAPAX" in self.category_select.options:
            self.category_select.value = "TSAPAX"
            #print('self.change_category(None,"","TSAPAX")')
        else:
            self.category_select.value = ""
            #print('self.change_category(None,"","")')
        ## self.category_select.value = "TSAPAX"
        #self.change_category(None,"","TSAPAX")
        if "PERCENTAGE" in self.key_select.options:
            self.key_select.value = "PERCENTAGE"
            #print('self.key_select.value = "PERCENTAGE"')
        else:
            self.key_select.value = ""
            #print('self.key_select.value = ""')
        ## self.key_select.value = "PERCENTAGE"
        pass


    def add_point(self,attr,old,new):
        if self.add_point_guard:
            return
        self.add_point_guard = True
        ddf = pd.DataFrame(new)
        #if pd.Index(ddf["datetime"]).is_monotonic:
        #    return
        ddf = ddf.sort_values("datetime").reset_index()
        del ddf["index"]
        ddf["coloridx"] = -1
        ddf["class"] = "shock"
        ddf.at[ddf[ddf["value"].diff(1)>0].index.min():,"coloridx"]=1
        ddf.at[ddf[ddf["value"].diff(1)>0].index.min():,"class"]="recovery"
        #print(ddf)
        self.cds_drawn_polyline.data = {"datetime":ddf.datetime.values,"value":ddf.value.values,"coloridx":ddf.coloridx.values,"class":ddf["class"].values}
        #print(self.cds_drawn_polyline.data)
        self.add_point_guard = False
        self.clear_drawing.disabled = False


    def save_scenario_callback(self,event):
        df = self.cds_drawn_polyline.to_df()
        df["user_id"] = self.user_id.value
        df["datetime_vote"] = datetime.datetime.now()
        df["scenario_name"] = self.scenario_name.value
        df["category"] = self.category_select.value
        df["parameter_name"] = self.key_select.value
        df = df.rename(columns={"value":"parameter_value","datetime":"datetime_date","class":"shock_recovery"}) # db2 does not like value in SQL statements
        df["datetime_date"] = pd.to_datetime(df["datetime_date"]*1E6)
        df.to_csv("doobee.csv",index=False)
        conn = self.engine.connect()
        df.to_sql("cookiecutter_scenarios",conn,if_exists="append",dtype={'user_id':sqlalchemy.types.String(50),
                                                         'datetime_vote': sqlalchemy.types.DateTime,
                                                         'scenario_name':sqlalchemy.types.String(100),
                                                         'category':sqlalchemy.types.String(100),
                                                         'datetime_date': sqlalchemy.types.DateTime,
                                                         'shock_recovery':sqlalchemy.types.String(20),
                                                         'parameter_name':sqlalchemy.types.String(100),
                                                         'parameter_value':sqlalchemy.types.Float },index=False)
        conn.close()
        pass

    def clear_drawing_callback(self,event):
        self.cds_drawn_polyline.data = {"datetime":[],"value":[],"coloridx":[],"class":[]}


    def delete_selected_point_callback(self,event):
        data = self.cds_drawn_polyline.data
        newdata = {}
        for k in data.keys():
            newdata[k] = [i for j, i in enumerate(data[k]) if j not in self.cds_drawn_polyline.selected.indices] # https://stackoverflow.com/questions/497426/deleting-multiple-elements-from-a-list
        self.cds_drawn_polyline.selected.indices = []
        self.cds_drawn_polyline.data = newdata


    def drawn_polyline_selection_change_callback(self,attr,old,new):
        print("drawn_polyline_selection_change_callback old {} new {}".format(old,new))
        if len(new)>0:
            self.delete_selected_point.disabled = False
        else:
            self.delete_selected_point.disabled = True
        pass


    def scenario_name_callback(self,attr,old,new):
        if len(new)>0:
            self.save_scenario.disabled = False
        else:
            self.save_scenario.disabled = True


    def create(self):
        self.cds_proxy_data = ColumnDataSource(pd.DataFrame({"datetime":[],"value":[]}))
        self.cds_drawn_polyline = ColumnDataSource({"datetime":[],"value":[],"coloridx":[],"class":[]})

        self.heading = Div(text="<H1>Economic Data</H1>")
        self.category_select = Select(title="Category",options=[""])
        self.get_categories()
        self.category_select.value = ""

        self.key_select = Select(title="Key",options=[""])

        self.start_date = DatePicker(title="Start Date",value=pd.to_datetime("2020-01-01").date())
        self.end_date = DatePicker(title="End Date",value=datetime.date.today())

        self.p_values = figure(plot_width=1200, plot_height=400,x_axis_type='datetime',title="",
            y_range=(0,1.05),
            tools="pan,box_zoom,box_select,reset", #active_drag="point_draw",
            output_backend="webgl")
        self.p_values.extra_y_ranges = {"value":Range1d()}
        self.p_values = self.theme.apply_theme_defaults(self.p_values)
        self.p_values.background_fill_color = "#ffffff"
        self.p_values.line(x="datetime",y="value",source=self.cds_proxy_data,y_range_name="value")
        #self.value_axis = LinearAxis(y_range_name="value")
        self.p_values.add_layout(LinearAxis(y_range_name="value",
            axis_label_text_color=self.theme.text_color,
            major_label_text_color=self.theme.text_color,
            axis_line_color=self.theme.plot_color), 'right')
        #print(self.start_date.value,type(self.start_date.value))
        #print(pd.to_datetime(self.start_date.value).date(),type(pd.to_datetime(self.start_date.value).date()))
        #print(Range1d(pd.to_datetime(self.start_date.value).date(),pd.to_datetime(self.end_date.value).date()))
        #self.p_values.x_range=Range1d(pd.to_datetime(self.start_date.value).date(),pd.to_datetime(self.end_date.value).date())
        #print(self.p_values.x_range)
        #self.p_values.y_range=Range1d(0,1)

        editor = self.p_values.line(x="datetime",y="value",source=self.cds_drawn_polyline,line_color="darkgrey",line_width=3)
        mapper = linear_cmap(field_name="coloridx",palette=["tomato","grey","seagreen"],low=-1,high=1)
        self.p_values.circle(x="datetime",y="value",source=self.cds_drawn_polyline,size=25,fill_color=mapper)#,fill_color="color",size=25)

        draw_tool = PointDrawTool(renderers=[editor], empty_value='black')
        self.p_values.add_tools(draw_tool)
        self.p_values.toolbar.active_tap = draw_tool


        columns = [TableColumn(field="datetime", title="Datetime", formatter=DateFormatter(),width=100),
                   TableColumn(field="value", title="Value",formatter=NumberFormatter(format="0.00"), width=100),
                   TableColumn(field="class", title="Shock/Recovery", width=100)]
        self.proxy_table = DataTable(source=self.cds_drawn_polyline, columns=columns, editable=True, height=500,selectable='checkbox',index_position=None)

        self.user_id = TextInput(value="nobody@{}".format(socket.gethostname()),title="Name to save your results")
        self.scenario_region = Select(title="Region the scenario applies to",options=["Global","UK","Germany"])
        self.scenario_sector = Select(title="Sector the scenario applies to",options=["Air Transport","Hotel"])
        self.scenario_name = TextInput(title="Title to save your scenario")
        self.scenario_name.on_change("value_input",self.scenario_name_callback) # Yay! This is not really documented anywhere

        self.save_scenario = Button(label="Save Scenario",disabled=True)
        self.save_scenario.on_event(ButtonClick, self.save_scenario_callback)

        self.explanation = Div(text="<H2>Explanation of dataset</H2>Select a category and parameter. You can draw an approximation using the mouse.")

        self.clear_drawing = Button(label="Clear",disabled=True)
        self.clear_drawing.on_event(ButtonClick, self.clear_drawing_callback)

        self.delete_selected_point = Button(label="Delete Selected Point(s)",disabled=True)
        self.delete_selected_point.on_event(ButtonClick, self.delete_selected_point_callback)

        self.cds_drawn_polyline.on_change("data",self.add_point)
        self.cds_drawn_polyline.selected.on_change('indices',self.drawn_polyline_selection_change_callback)
        self.category_select.on_change("value",self.change_category)
        self.key_select.on_change("value",self.change_key)

        self.load_data()

        return row([column([self.heading,
                    row([self.category_select,self.key_select,self.start_date,self.end_date]),
                    row([self.p_values]),
                    row([self.proxy_table,
                        column([self.clear_drawing,self.delete_selected_point])])]),            
                    column([self.user_id,self.scenario_region,self.scenario_sector,self.scenario_name,self.save_scenario,self.explanation])])


# just to be safe
Path("./data").mkdir(exist_ok=True,parents=True)
# instantiate main class
gui_health = GUIHealth()
# create widgets which will not have data to be displayed just yet
content_health = Panel(child=gui_health.create(),title="Health and Stringency")

gui_economy = GUIEconomy()
content_economy = Panel(child=gui_economy.create(),title="Economy and Economic Proxies")

content = Tabs(tabs=[content_health,content_economy])

#print("gui_economy.load_data()")
gui_economy.load_data()
# put it on the canvas/web browser
curdoc().add_root(content)
# check if we have data...
gui_health.compute_data_status()
# load it and map it onto the plots. The latter is done by selecting a default country, which triggers the country callback
gui_health.load_data()

