# Cookiecutter tool
#
# Copyright (C) Dr. Klaus G. Paul 2020 Rolls-Royce Deutschland Ltd & Co KG
# Made on behalf of Emergent Alliance Ltd
#
# Notice to users: This is a Minimum Viable Product designed to elicit user requirements. The code
# quality is write-once, reuse only if deemed necessary
#
from bokeh.models import Button, Plot, TextInput, Legend, TextAreaInput, FactorRange
from bokeh.palettes import RdYlBu3, RdYlBu11, Turbo256, Plasma256, Colorblind8
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.models.widgets import Select
from bokeh.layouts import row, column, layout
from bokeh.models import Range1d, HoverTool, LinearAxis, Label, NumeralTickFormatter, PrintfTickFormatter, Div, LinearColorMapper, ColorBar, BasicTicker
from bokeh.models import BoxAnnotation, DataTable, DateFormatter, TableColumn, RadioGroup, Spinner, Paragraph, RadioButtonGroup, DatePicker
from bokeh.models import Panel, Tabs, DatePicker, PointDrawTool, DataTable, TableColumn, NumberFormatter, DataRange1d, MultiChoice
from bokeh.palettes import Category20, Category10, RdYlBu3, Greys4
from bokeh.client.session import push_session, show_session
from bokeh.events import SelectionGeometry, ButtonClick, Press, PlotEvent
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
import logging


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
            sql_url = "postgresql://cookiecutter:cookiecutter@database:5432/cookiec"
        else:
            print("HAVE CONNECT {}".format(os.environ["SQL_CONNECT"]))
            sql_url = os.environ["SQL_CONNECT"]
        #print("SQL_CONNECT {}".format(sql_url))
        logging.error(sql_url)
        self.engine = create_engine(sql_url, pool_size=10, max_overflow=20)
        print(self.engine)
        try:
            self.max_gumbel_waves = int(os.environ["GUMBEL_MAX_WAVES"])
        except:
            self.max_gumbel_waves = 25
        self.dataset_selected = ""
        self.gumbel_visibility_lock = False # interlock to avoid ping-pong of callbacks


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


    def gumbel_choices_callback(self,attr,old,new):
        removed = list(set(old)-set(new))
        added = list(set(new)-set(old))
        self.gumbel_visibility_lock = True # interlock to avoid ping-pong of callbacks
        for v in removed:
            idx = int(v)
            self.gumbel_wave_renderers[idx].visible = False
        for v in added:
            idx = int(v)
            self.gumbel_wave_renderers[idx].visible = True

        num_datapoints = len(self.cds_gumbel_waves.data["summed_waves"])
        summed_data = [0. for i in range(num_datapoints)]
        chosen_waves = []
        for w in new:
            wave = "wave_{}".format(w)
            for j in range(num_datapoints):
                summed_data[j] += self.cds_gumbel_waves.data[wave][j]
        self.cds_gumbel_waves.data["summed_waves"] = summed_data

        self.gumbel_visibility_lock = False # interlock to avoid ping-pong of callbacks


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
        try:
            result = conn.execute("SELECT MAX(vote_id) FROM cookiecutter_verdicts")
            vote_id = result.fetchone()[0]+1
            have_verdict_table = True
        except:
            print("CANNOT RETRIEVE VOTE_ID")
            have_verdict_table = False
            vote_id = 1
        #print("VOTE ID = {}".format(vote_id))

        df = self.cds.to_df().iloc[self.cds.selected.indices]
        if max(df["new_cases_rel"]) < 0:
            df["rel_peak_new_cases"] = 0.
        else:
            df["rel_peak_new_cases"] = max(df["new_cases_rel"])
        max_selected = max(self.cds.selected.indices)
        if max_selected >= len(self.cds.data["new_cases"])-1:
            df["kind"] = self.scenario_type.labels[self.scenario_type.active]+"_act"
        else:
            df["kind"] = self.scenario_type.labels[self.scenario_type.active]

        df["kind_counter"] = int(self.scenario_number.value)
        df["from_dt"] = df.datetime_date.min()
        df["to_dt"] = df.datetime_date.max()
        df["duration"] = (pd.to_datetime(df.datetime_date.max())-pd.to_datetime(df.datetime_date.min())).total_seconds()/86400
        df["user"] = self.user_id.value ### change to user_name
        #ADM0_A3 = self.adm0_a3 #dfMapping[self.dfMapping.name == self.country_select.value].adm0_a3.values[0]
        df["identifier"] = self.identifier
        df["vote_datetime"] = datetime.datetime.now()
        df["vote_id"] = vote_id
        df.datetime_date = pd.to_datetime(df.datetime_date)
        #print(df.datetime_date)

        ddf = self.cds_gumbel_waves.to_df()
        ddf = ddf[(df.datetime_date.min() <= ddf.datetime)&(ddf.datetime <= df.datetime_date.max())]
        ddf.datetime = pd.to_datetime(ddf.datetime)
        i = 0
        for r in self.gumbel_wave_renderers:
            if not r.visible:
                ddf["wave_{:02d}".format(i)] = -1
            elif ddf["wave_{:02d}".format(i)].sum() == 0:
                del ddf["wave_{:02d}".format(i)]
            i += 1
        df = df.merge(ddf,left_on="datetime_date",right_on="datetime").rename(columns={"trend_x":"trend"})
        del df["trend_y"]
        del df["datetime"]

        # amend schema if necessary
        if have_verdict_table:
            meta = sqlalchemy.MetaData()
            schema = sqlalchemy.Table("cookiecutter_verdicts",meta, autoload=True, autoload_with=conn)
            existing_wave_columns = []
            for c in schema.columns:
                if "wave" in c.name.lower():
                    existing_wave_columns.append(c.name.lower())
            for c in df.columns:
                if "wave" in c:
                    if c not in existing_wave_columns:
                        print("ALTER TABLE cookiecutter_verdicts ADD COLUMN {} FLOAT;".format(c))
                        conn.execute("ALTER TABLE cookiecutter_verdicts ADD COLUMN {} FLOAT;".format(c))


        # Avoid loss of precision SQL errors for tiny numbers
        for c in df.columns:
            if "wave" in c:
                df[c] = df[c].clip(lower=0.1)

        df.to_sql("cookiecutter_verdicts", conn, if_exists='append', dtype={"from_dt":sqlalchemy.types.Date,
                                                                         "to_dt":sqlalchemy.types.Date,
                                                                         "datetime_date":sqlalchemy.types.DateTime,
                                                                         "vote_datetime":sqlalchemy.types.DateTime,
                                                                         "kind":sqlalchemy.types.String(10),
                                                                         "user":sqlalchemy.types.String(50),
                                                                         "identifier":sqlalchemy.types.String(10)},
                                                                         index=False,chunksize=25,method=None)
        #print(df[["kind"]])
        """for i in df.index:
            print(i)
            ddf = df[i:i+1]
            ddf.to_pickle("uaaah.pckld")
            ddf.to_sql("cookiecutter_verdicts", conn, if_exists='append', dtype={"from_dt":sqlalchemy.types.Date,
                                                                         "to_dt":sqlalchemy.types.Date,
                                                                         "datetime_date":sqlalchemy.types.DateTime,
                                                                         "vote_datetime":sqlalchemy.types.DateTime,
                                                                         "kind":sqlalchemy.types.String(10),
                                                                         "user":sqlalchemy.types.String(50),
                                                                         "identifier":sqlalchemy.types.String(10)},
                                                                         index=False)
            conn.close()
            conn = self.engine.connect()"""

        conn.close()
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
        #print(self.dfVotesContent)
        self.compute_metrics()


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


    def gumbel_plot_callback(self,attr,old,new):
        if self.gumbel_visibility_lock: # interlock to avoid ping-pong of callbacks
            return 
        num_datapoints = len(self.cds_gumbel_waves.data["summed_waves"])
        summed_data = [0. for i in range(num_datapoints)]
        chosen_waves = []

        i = 0
        for r in self.gumbel_wave_renderers:
            wave = "wave_{:02d}".format(i)
            if r.visible:
                chosen_waves.append("{:02d}".format(i))
                for j in range(num_datapoints):
                    summed_data[j] += self.cds_gumbel_waves.data[wave][j]
            i += 1

        self.gumbel_choices.value = chosen_waves
        self.cds_gumbel_waves.data["summed_waves"] = summed_data


    def change_country(self,attr,old,new):
        """Handle change of country to be displayed.

        This generates a dict style data structure that can be used to overwrite the various ColumnDataSource s
        that are used to display the values. This is the common bokeh pattern to update a display.

        Commonly made mistake is to re-create a new ColumnDataSource and try to squeeze in a pandas
        dataframe directly, this, however, will not update the plot.
        """
        # fresh data for the time series plots

        sql_query = """SELECT cookiecutter_case_data.*,oxford_stringency_index.*
    FROM cookiecutter_case_data
    INNER JOIN oxford_stringency_index ON cookiecutter_case_data.identifier = oxford_stringency_index.countrycode AND
    cookiecutter_case_data.datetime_date = oxford_stringency_index.datetime_date 
    WHERE cookiecutter_case_data.name='{}' AND oxford_stringency_index.regionname IS NULL
    AND cookiecutter_case_data.data_source ='{}'
    ORDER BY cookiecutter_case_data.datetime_date;""".format(new,self.dataset_select.value)

        conn = self.engine.connect()
        dfData = pd.read_sql(sql_query, conn)

        if len(dfData) == 0: # most likely no OxCGRT dataset
            have_OxCGRT = False
            sql_query = "SELECT * FROM cookiecutter_case_data WHERE cookiecutter_case_data.name='{}' AND cookiecutter_case_data.data_source ='{}' ORDER BY cookiecutter_case_data.datetime_date;".format(new,self.dataset_select.value)
            dfData = pd.read_sql(sql_query, conn)
            print("sql_query {}".format(sql_query))
        else:
            have_OxCGRT = True
            # new we have a clone of datetime_date
            dfRubbish = dfData.datetime_date
            del dfData["datetime_date"]
            dfRubbish.columns=["rubbish","datetime_date"]
            #print(dfRubbish)
            #dfData.index = dfRubbish.datetime_date
            dfData["datetime_date"] = dfRubbish.datetime_date
        dfData.index.name = None

        sql_query = "SELECT population FROM population_data WHERE identifier='{}'".format(dfData.identifier.unique()[0])
        try:
            population = int(conn.execute(sql_query).fetchone()[0])
            dfData['new_cases_rel'] = dfData['new_cases']/population
        except:
            dfData['new_cases_rel'] = -1.
            print("NO POPULATION DATA FOR identifer ({}) name ({})".format(dfData.identifier.unique()[0],new))

        newdata = {}
        #print(self.cds.data.keys())
        for column in self.cds.data.keys():
            if column == "index" or column == "Timestamp" or column == "datetime_date":
                #newdata[column] = self.adfCountryData[new].index
                newdata[column] = dfData.datetime_date #datetime_date
            elif "wave_" in column or column == "summed_waves":
                continue
            elif column in dfData.columns: # cater for reduced oxCGRT missing queries
                newdata[column] = np.nan_to_num(dfData[column])
            else:
                #newdata[column] = np.nan_to_num(self.adfCountryData[new][column])
                if have_OxCGRT:
                    newdata[column] = np.nan_to_num(dfData[column])
                else:
                    newdata[column] = [-1 for i in range(len(dfData["new_cases"]))]
        self.cds.data = newdata
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
        print("HAVE OXCGRT {}".format(have_OxCGRT))
        for column in self.cds_OxCGRTHeatmap.data.keys():
            if column == "index" or column == "Timestamp":
                newdata[column] = dfMeasures.index
                #newdata[column] = dfData.datetime_date
            else:
                if have_OxCGRT:
                    newdata[column] = np.nan_to_num(dfMeasures[column])
                else:
                    newdata[column] = []
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
        conn = self.engine.connect()
        dfWaves = pd.read_sql("SELECT * from cookiecutter_computed_waves_chgpoint WHERE name='{}'".format(new),conn)
        conn.close()
        last = "new"
        box_no = 0
        for i,row in dfWaves.iterrows():
            if row["kind"] == "begin":
                left = row["datetime_date"]
                if last == "end":
                    try:
                        self.wave_boxes[box_no].left = right
                    except:
                        self.wave_boxes[box_no].left = left
                    self.wave_boxes[box_no].right = left
                    self.wave_boxes[box_no].visible = True
                    self.wave_boxes[box_no].fill_color = "#00FF00"
                    self.wave_boxes[box_no].fill_alpha = 0.05
                    box_no += 1
                last = "begin"
            elif row["kind"] == "end":
                right = row["datetime_date"]
                try:
                    self.wave_boxes[box_no].left = left
                except:
                    self.wave_boxes[box_no].left = right
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
            ddf = pd.read_sql("select DISTINCT from_dt,to_dt,user,kind,vote_id,rel_peak_new_cases,duration from cookiecutter_verdicts WHERE identifier='{}'".format(self.identifier),conn)
        except:
            ddf = pd.DataFrame()
        #if len(dfVotesContent)>0:
        if len(ddf)>0:
            #ADM0_A3 = self.dfMapping[self.dfMapping.name == new].ADM0_A3.values[0]
            #ddf = dfVotesContent[self.dfVotesContent.ADM0_A3 == ADM0_A3][["from","to","kind"]].copy().drop_duplicates()
            ddf["from"] = pd.to_datetime(ddf["from_dt"])
            ddf["to"] = pd.to_datetime(ddf["to_dt"])
            ddf["color"] = None
            ddf.loc[ddf["kind"] == "Wave","color"] = "tomato"
            ddf.loc[ddf["kind"] == "Wave_act","color"] = "orange"
            ddf.loc[ddf["kind"] == "Calm","color"] = "mediumseagreen"
            ddf.loc[ddf["kind"] == "Calm_act","color"] = "lightgreen"
            ddf["height"] = [random.random()/3+0.1 for i in ddf.index]
            ddf["y"] = [0.5 for i in ddf.index]
            self.cds_votes_ranges.data = {"from":ddf["from"].values,"to":ddf.to.values,"y":ddf.y.values,
                                            "height":ddf.height.values,"color":ddf.color.values}
        else:
            self.cds_votes_ranges.data = {"from":[],"to":[],"y":[],
                                            "height":[],"color":[]}

        
        ddf = dfData[["new_cases","datetime_date"]].copy().fillna(0)
        #ddf.index = ddf.datetime_date
        #ddf.index.name = None

        all_waves = compute_gumbel_waves(ddf,maxwaves=self.max_gumbel_waves)
        num_waves = len(all_waves)
        self.gumbel_choices.options = ["{:02d}".format(i) for i in range(num_waves)]
        self.gumbel_choices.value = ["{:02d}".format(i) for i in range(num_waves)]

        newdata = {}#self.cds_gumbel_waves.data
        empty_list = [0 for i in range(len(ddf))]
        for k,v in self.cds_gumbel_waves.data.items():
            newdata[k] = empty_list
        newdata["summed_waves"] = [0 for i in range(len(ddf))]
        if len(all_waves)>0:
            i = 0
            for wave in all_waves:
                newdata["datetime"] = wave["datetime_date"]
                newdata["wave_{:02d}".format(i)] = wave["wave_values"]
                newdata["summed_waves"] = list(map(add,newdata["summed_waves"],wave["wave_values"])) # https://stackoverflow.com/questions/18713321/element-wise-addition-of-2-lists
                i += 1
            try:
                newdata["trend"] = dfData[(min(newdata["datetime"])<=dfData["datetime_date"]) & (dfData["datetime_date"] <= max(newdata["datetime"]))]["trend"]
            except:
                pass # strange error
        self.cds_gumbel_waves.data = newdata

        i = 0
        for r in self.gumbel_wave_renderers:
            r.visible = True

        try:
            ddf =  pd.read_sql("SELECT DISTINCT cluster_id FROM stringency_index_clustering where country='{}'".format(new),conn)
        except:
            ddf = pd.DataFrame()
        if len(ddf) > 0:
            cluster_id = ddf.cluster_id.values[0]
            df = pd.read_sql("SELECT country,state_value,state_date,cluster_id FROM stringency_index_clustering where cluster_id='{}'".format(cluster_id),conn)
            df["alpha"] = 0.25
            df.at[df.country==new,"alpha"] = 1.0
            country_list = ", ".join(sorted(df.country.unique(),reverse=False))

            newdata = {}
            self.p_oxcluster.visible = True
            for k in self.cds_oxcluster.data.keys():
                newdata[k] = df[k].values
            self.cds_oxcluster.data = newdata
            self.p_oxcluster.height = 50+15*len(df.country.unique())
            self.p_oxcluster.y_range = FactorRange(factors=sorted(df.country.unique(),reverse=True))
        else:
            self.p_oxcluster.visible = False

        conn.close()




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
        self.refresh_data = Button(label="Load Data",disabled=False,width=100)
        self.refresh_data.on_event(ButtonClick, self.refresh_data_callback)
        conn = self.engine.connect()
        df = pd.read_sql("SELECT DISTINCT data_source FROM cookiecutter_case_data",conn).dropna()
        conn.close()
        self.dataset_select=Select(options=sorted(df.data_source.values),width=250)
        self.dataset_select.on_change("value",self.change_dataset)
        self.country_select = Select(options=[""],width=250)
        self.country_select.on_change("value",self.change_country)
        self.scenario_name = TextInput(value="country wave 2")
        self.scenario_type_label = Paragraph(text="Select Type")
        #self.scenario_type = RadioGroup(labels=["Wave","Calm"],width=100)
        self.scenario_type = RadioButtonGroup(labels=["Wave","Calm"],width=100,active=0)
        self.scenario_number_label = Paragraph(text=" Occurence #")
        self.scenario_number = Spinner(low=1, step=1, value=1,width=50)
        self.save = Button(label="Save",disabled=True,width=100)
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
        # Gumbel distribution waves
        empty_data.update(dict(zip(["wave_{:02d}".format(i) for i in range(self.max_gumbel_waves)],[[] for i in range(100)])))
        for c in ['datetime_date','new_cases','new_cases_rel','trend','stringencyindex','summed_waves']: 
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
        empty_data["summed_waves"] = []
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

        tooltips_top = [("datetime_date","@datetime_date{%Y-%m-%d}"),
                    ("New Cases","@new_cases{0}"),
                    ("New Cases Trend","@trend{0}"),
                    #("Active Cases","@active{0}"),
                    #("Deaths","@deaths{0.00}"),
                    ]
        hover_top = HoverTool(tooltips = tooltips_top,
                          formatters={
                              "@datetime_date": "datetime"
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

        self.cds_oxcluster = ColumnDataSource({"country":[],"state_value":[],"state_date":[],"cluster_id":[],"alpha":[]})
        self.p_oxcluster = figure(plot_width=1200, plot_height=65, x_axis_type="datetime",title="Temporal clusters of measures for ",
                            x_range = self.p_top.x_range,
                            y_range=[], toolbar_location=None, tools="", 
                            output_backend="webgl")
        ox_palette = ["#e41a1c","#377eb8","#4daf4a","#ffffb2"]
        ox_mapper = LinearColorMapper(palette=ox_palette, low=0, high=3)
        self.p_oxcluster.rect(y="country", x="state_date", width=86400000, height=1, source=self.cds_oxcluster,fill_alpha="alpha",
                            line_color=None, fill_color=transform('state_value', ox_mapper))



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
        r1 = self.p_gumbel.x(x="datetime",y="summed_waves",source=self.cds_gumbel_waves,alpha=0.75,color="#000000")
        self.p_gumbel.add_layout(Legend(items=[("trend", [r0]),
            ("summed gumbel waves", [r1])],location="top_left"))

        self.gumbel_choices = MultiChoice(value=[],options=[],title="keep waves")
        self.gumbel_choices.on_change("value",self.gumbel_choices_callback)

        #r = [None for i in range(self.max_gumbel_waves)]
        legend_items = []
        stack_y = []
        colors = []
        for i in range(self.max_gumbel_waves):
            stack_y.append("wave_{:02d}".format(i))
            colors.append(Colorblind8[i%len(Colorblind8)])
        self.gumbel_wave_renderers = self.p_gumbel.varea_stack(stack_y,x="datetime",source=self.cds_gumbel_waves,color=colors,fill_alpha=0.5)
        for i in range(self.max_gumbel_waves):
            legend_items.append(("{:02d}".format(i), [self.gumbel_wave_renderers[i]]))
        self.p_gumbel_legend = Legend(items=legend_items,click_policy="hide",orientation="horizontal",label_text_font_size="6pt",
            label_height=1,label_standoff=1,label_text_line_height=1,margin=1,spacing=0,
            background_fill_color="#252425",label_text_color="#FFFFFF",padding=0)
        self.p_gumbel.add_layout(self.p_gumbel_legend,"below")
        for renderer in self.gumbel_wave_renderers:
            renderer.on_change('visible',self.gumbel_plot_callback)


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

        self.voting_table = DataTable(source=self.cds_votes,columns=columns,width=400,height=600)

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
        self.p_oxcluster = self.theme.apply_theme_defaults(self.p_oxcluster)
        self.p_oxcluster.background_fill_color = "#ffffff"
        self.p_gumbel = self.theme.apply_theme_defaults(self.p_gumbel)
        self.p_gumbel.background_fill_color = "#ffffff"
        self.voting_table = self.theme.apply_theme_defaults(self.voting_table)
        self.p_votes = self.theme.apply_theme_defaults(self.p_votes)
        self.p_votes.background_fill_color = "#ffffff"
        self.gumbel_choices = self.theme.apply_theme_defaults(self.gumbel_choices)


        self.p_histogram_wave = self.theme.apply_theme_defaults(self.p_histogram_wave)
        self.p_histogram_wave.background_fill_color = "lightsteelblue"
        self.p_histogram_calm = self.theme.apply_theme_defaults(self.p_histogram_calm)
        self.p_histogram_calm.background_fill_color = "lightsteelblue"
        self.p_duration_heatmap = self.theme.apply_theme_defaults(self.p_duration_heatmap)
        self.p_duration_heatmap.background_fill_color = "lightsteelblue"

        self.user_id = TextInput(value="nobody@{}".format(socket.gethostname()),title="Name to save your results")

        self.help_text = Div(text="""<H3>How to use this tool</H3>
            Select a dataset from the dropdowns to browse infection numbers by country (or state). The trend is a 7 days de-seasoned indicator computed by this toolset.
            Stringcency indices are from the <a href="https://github.com/OxCGRT/covid-policy-tracker" style="color:#DDDDDD;">Oxford Covid-19 Government Response Tracker (OxCGRT)</a>.
            Gumbel wave fits are based on work from <a href="https://gitlab.com/dzwietering" style="color:#DDDDDD;">Damiaan Zwietering</a>,  see also
            <a href="https://gitlab.com/dzwietering/corona/-/tree/master/pydata" style="color:#DDDDDD;">https://gitlab.com/dzwietering/corona/-/tree/master/pydata</a>.
            The histogram and heatmap data are populated based on your votes. Refer to <a href="https://github.com/emergent-analytics/workstreams" style="color:#DDDDDD;">emergent-analytics 
            workstreams github repository</a> for license terms, and further documentation.
            """)

        # The return value is a row/column/widget array that defines the arrangement of the various elements.
        return(row([column([self.progress_bar,
                            row([self.blank,
                                #self.refresh_data,
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
                        self.p_oxcluster,
                        self.p_gumbel,
                        self.p_votes,
                            row([Div(text='<div class="horizontalgap" style="width:200px"><h2>Statistics</h2></div>'),
                                self.p_histogram_wave,self.p_histogram_calm,self.p_duration_heatmap]),
                        ]),
                    column([self.user_id,
                        self.voting_table,
                        self.gumbel_choices,
                        self.help_text]),
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

        """if not self.engine.dialect.has_table(self.engine,"un_population_data_2020_estimates"):
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
            conn.close()"""



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


        
        print("Finished download")

                

    def process_data(self):
        """The heavy lifting of  processing the infection numbers of Johns Hopkins and the OxCGRT data.
        """
        # This look slike a duplicate
        # TODO: cleanup
        pass


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
        ### TODO this code would not work anymore
        #score = {}
        #for country in self.country_select.options:
        #    score[country] = self.adfCountryData[country].infection_rate_7.values[-1]
        #score_sorted = {k: v for k, v in sorted(score.items(), key=lambda item: item[1],reverse=True)}
        #self.country_select.options = list(score_sorted.keys())
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
            conn = self.engine.connect()

            df = pd.read_sql("select distinct data_source from cookiecutter_case_data;",conn)
            self.dataset_select.options = sorted(df.data_source.values)
            self.dataset_select.value = "Johns Hopkins global"
            print("DATASETS {}".format(self.dataset_select.options ))

            df = pd.read_sql("SELECT DISTINCT name FROM cookiecutter_case_data  WHERE data_source='Johns Hopkins global' ORDER BY name;",conn)
            self.country_select.options = list(df.name.values)

            sql_query = "SELECT name,identifier,count(CASE WHEN kind='begin' THEN kind END) as waves, count(CASE WHEN kind='end' THEN kind END) as episodes, count(*) from cookiecutter_computed_waves_chgpoint GROUP BY name,cookiecutter_computed_waves_chgpoint.identifier"
            dfWaves = pd.read_sql(sql_query,conn)

            try:
                sql_query = "SELECT count(r.identifier) as votes,r.identifier FROM (SELECT identifier FROM cookiecutter_verdicts GROUP BY vote_id,identifier) as r GROUP BY r.identifier;"
                dfVotes = pd.read_sql(sql_query,conn)
            except:
                dfVotes=pd.DataFrame({"identifier":[],"votes":[],"count":[]})
            dfVotesContent = dfWaves.merge(dfVotes,on="identifier",how="outer").fillna(0)
            dfVotesContent.votes = dfVotesContent.votes.astype(int)
            dfVotesContent["need_vote"] = dfVotesContent.waves+dfVotesContent.episodes > dfVotesContent.votes
            dfVotesContent["count"] = dfVotesContent.waves+dfVotesContent.episodes 
            dfVotesContent = dfVotesContent.sort_values("count",ascending=False)


            if len(dfVotesContent)>0:
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
            sql_url = "postgresql://cookiecutter:cookiecutter@database:5432/cookiec"
        else:
            sql_url = os.environ["SQL_CONNECT"]
        #print("SQL_CONNECT {}".format(sql_url))
        self.engine = create_engine(sql_url, pool_size=10, max_overflow=20)
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
        categories = []
        try:
            result = conn.execute("SELECT DISTINCT category FROM economic_indicators;")
            categories.extend([c[0] for c in result.fetchall()])
            conn.close()
        except:
            pass
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
        sql_query = "SELECT datetime_date,parameter_value FROM economic_indicators WHERE category='{}' and parameter_name='{}' ORDER BY datetime_date;".format(category,key)
        df = pd.read_sql(sql_query,conn)
        print("CHANGE_KEY from '{}' to '{}'".format(old,new))
        print(sql_query)
        #print(df)
        self.cds_proxy_data.data = {"datetime":df["datetime_date"].values,"value":df["parameter_value"].values}
        self.p_values.title.text = "{} - {}".format(category,key)
        self.p_values.x_range=DataRange1d(pd.to_datetime(self.start_date.value).date(),pd.to_datetime(self.end_date.value).date())
        #self.value_axis.bounds=DataRange1d(df.value.min(),df.value.max())
        value_range = df["parameter_value"].max()-df["parameter_value"].min()
        self.p_values.extra_y_ranges["value"].start = df["parameter_value"].min()-value_range*0.05
        self.p_values.extra_y_ranges["value"].end = df["parameter_value"].max()+value_range*0.05
        self.p_values.yaxis[1].axis_label = new
        df = pd.read_sql("SELECT DISTINCT explanation,explanation_text FROM economic_indicators WHERE category='{}' and parameter_name='{}';".format(category,key),conn)
        conn.close()
        url_shown = df["explanation"].values[0]
        url = "https://translate.google.com/translate?hl=en&sl=auto&tl=en&u={}".format(urllib.parse.quote_plus(url_shown))
        self.explanation.text="<H1>{}</H1><H2>{}</H2>See <A HREF=\"{}\" style=\"color:#DDDDDD;\">{}</A> for more details".format(category,key,url,url_shown)
        self.explanation_text.text = df["explanation_text"].values[0]


    def get_keys(self,category=""):
        if category == "":
            category = self.category_select.value
        conn = self.engine.connect()
        result = conn.execute("SELECT DISTINCT parameter_name FROM economic_indicators WHERE category='{}' ORDER BY parameter_name;".format(category))
        keys = []
        keys.extend([k[0] for k in result.fetchall()])
        #print("KEYS {}".format(keys))
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
        conn = self.engine.connect()
        try:
            ddf = pd.read_sql("SELECT DISTINCT name FROM input_output_tables",conn)
            regions = sorted(ddf.name.values)
        except:
            regions = ["Global","Europe","National"]
        self.scenario_region.options = regions
        self.scenario_region.value = regions

        try:
            ddf = pd.read_sql("SELECT DISTINCT row_sector FROM input_output_tables",conn)
            sectors = sorted(ddf.row_sector.values)
        except:
            sectors = sorted(["Air Transport","Hotel","Finance","Industry","Sales","Services"])
        self.scenario_sector.options = sectors
        self.scenario_sector.value = [random.choice(sectors)]
        conn.close()


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


    def scenario_name_callback(self,attr,old,new): #################
        print("SCENARIO_NAME_CALLBACK",old,new)
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
            #output_backend="webgl"
            )
        self.p_values.extra_y_ranges = {"value":Range1d()}
        self.p_values = self.theme.apply_theme_defaults(self.p_values)
        self.p_values.background_fill_color = "#ffffff"
        self.p_values.line(x="datetime",y="value",source=self.cds_proxy_data,y_range_name="value")
        #self.value_axis = LinearAxis(y_range_name="value")
        self.p_values.add_layout(LinearAxis(y_range_name="value",
            axis_label_text_color=self.theme.text_color,
            major_label_text_color=self.theme.text_color,
            axis_line_color=self.theme.plot_color), 'right')

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
        regions = [] 
        self.scenario_region = MultiChoice(title="Region the scenario applies to",options=regions,value=regions)
        sectors = [] 
        self.scenario_sector = MultiChoice(title="Sector the scenario applies to",options=sectors,value=sectors,width=400)
        dummy_text = "Scenario {:%Y-%m-%d %H:%M} using ".format(datetime.datetime.now()) 
        self.scenario_name = TextInput(title="Title to save your scenario",value=dummy_text)# Only single line TextInput works with value_input typed keys
        self.scenario_name.on_change("value_input",self.scenario_name_callback) # Yay! This is not really documented anywhere

        self.save_scenario = Button(label="Save Scenario",disabled=True)
        self.save_scenario.on_event(ButtonClick, self.save_scenario_callback)

        self.explanation = Div(text="<H2>Explanation of dataset</H2>Select a category and parameter. You can draw an approximation using the mouse.")
        self.explanation_text = Div(text="<H2>Explanation of dataset</H2>Select a category and parameter. You can draw an approximation using the mouse.")

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
                    row([self.category_select,self.key_select]), #self.start_date,self.end_date]),
                    row([self.p_values]),
                    row([self.proxy_table,
                        column([self.clear_drawing,self.delete_selected_point])])]),            
                    column([self.user_id,self.scenario_region,self.scenario_sector,self.scenario_name,self.save_scenario,self.explanation,self.explanation_text])])


class GUIwhatif():
    def __init__(self):
        self.theme = THEME()
        if "SQL_CONNECT" not in list(os.environ.keys()):
            sql_url = "postgresql://cookiecutter:cookiecutter@database:5432/cookiec"
        else:
            sql_url = os.environ["SQL_CONNECT"]
        self.engine = create_engine(sql_url, pool_size=10, max_overflow=20)
        print(self.engine)


    def end_of_wave_2_callback(self,attr,old,new):
        df = self.cds.to_df().dropna()
        self.cds_end_of_wave_2.data = {"datetime_date":[pd.to_datetime(df.datetime_date.max()),pd.to_datetime(new)],"value":[df.trend.values[-2],0]}
        #try:
        wave3_start = pd.to_datetime(new)+pd.Timedelta(days=self.calm_durations[1])
        wave3_end   = wave3_start + pd.Timedelta(days=self.wave_durations[1])
        print("Wave 3 start {} end {}".format(wave3_start,wave3_end))

        calm3_end = wave3_end+pd.Timedelta(days=self.calm_durations[2])
        print("Calm 3 start {} end {}".format(wave3_end,calm3_end))

        wave4_end = calm3_end + pd.Timedelta(days=self.wave_durations[1])
        print("Wave 4 start {} end {}".format(calm3_end,wave4_end))

        #self.cassandra.text = """<H1>Scenario Forecast</H1>
        #    Wave 3 start {:%Y-%m-%d} end {:%Y-%m-%d}<P>
        #    Calm 3 start {:%Y-%m-%d} end {:%Y-%m-%d}<P>
        #    Wave 4 start {:%Y-%m-%d} end {:%Y-%m-%d}""".format(wave3_start,wave3_end,wave3_end,calm3_end,calm3_end,wave4_end)

        self.cds_cassandra.data = {"category":["Wave 3","Calm 3","Wave 4"],"from_dt":[wave3_start,wave3_end,calm3_end],"to_dt":[wave3_end,calm3_end,wave4_end]}
        pass


    def select_country_callback(self,attr,old,new):
        conn = self.engine.connect()
        sql = "SELECT datetime_date,sum(new_cases) as new_cases,sum(trend) AS trend FROM cookiecutter_case_data WHERE name IN ("+",".join(["'{}'".format(c) for c in new])+") GROUP BY datetime_date ORDER BY datetime_date;"
        df = pd.read_sql(sql,conn)
        datetime_dates = list(df.datetime_date.values)
        self.cds.data = {"datetime_date":df.datetime_date.dt.date.values,
                        "new_cases":df.new_cases.values,
                        "trend":df.trend.values,
            }

        self.p_top.x_range.update(end=df.datetime_date.max()+pd.Timedelta(days=60))

        df = pd.read_sql("SELECT DISTINCT adm0_a3 AS identifier FROM johns_hopkins_country_mapping WHERE name IN ("+",".join(["'{}'".format(c) for c in new])+")",conn)
        identifiers = list(df.identifier.values)
        try:
            sql = "SELECT DISTINCT from_dt,to_dt,user,kind,vote_id,rel_peak_new_cases,duration,kind_counter FROM cookiecutter_verdicts WHERE identifier IN ("+",".join(["'{}'".format(c) for c in identifiers])+")"
            ddf = pd.read_sql(sql,conn)
        except:
            ddf = pd.DataFrame()
        if len(ddf)>0:
            #ADM0_A3 = self.dfMapping[self.dfMapping.name == new].ADM0_A3.values[0]
            #ddf = dfVotesContent[self.dfVotesContent.ADM0_A3 == ADM0_A3][["from","to","kind"]].copy().drop_duplicates()
            ddf["from"] = pd.to_datetime(ddf["from_dt"])
            ddf["to"] = pd.to_datetime(ddf["to_dt"])
            ddf["color"] = None
            ddf.loc[ddf["kind"] == "Wave","color"] = "tomato"
            ddf.loc[ddf["kind"] == "Wave_act","color"] = "orange"
            ddf.loc[ddf["kind"] == "Calm","color"] = "mediumseagreen"
            ddf.loc[ddf["kind"] == "Calm_act","color"] = "lightgreen"
            ddf["height"] = [random.random()/3+0.1 for i in ddf.index]
            ddf["y"] = [0.5 for i in ddf.index]
            self.cds_votes_ranges.data = {"from":ddf["from"].values,"to":ddf.to.values,"y":ddf.y.values,
                                            "height":ddf.height.values,"color":ddf.color.values}
        else:
            self.cds_votes_ranges.data = {"from":[],"to":[],"y":[],
                                            "height":[],"color":[]}

        if len(ddf)>0:
            wave_duration_q10 = ddf[(ddf["kind"] == "Wave")&(ddf["kind_counter"] == 1)].duration.quantile(0.1)
            wave_duration_mean = ddf[(ddf["kind"] == "Wave")&(ddf["kind_counter"] == 1)].duration.mean()
            wave_duration_q95 = ddf[(ddf["kind"] == "Wave")&(ddf["kind_counter"] == 1)].duration.quantile(0.95)
            self.wave_durations = [wave_duration_q10,wave_duration_mean,wave_duration_q95]
            print("Duration wave [{:.1f},{:.1f},{:.1f}]".format(wave_duration_q10,wave_duration_mean,wave_duration_q95))

            calm_duration_q10 = ddf[ddf["kind"] == "Calm"].duration.quantile(0.1)
            calm_duration_mean = ddf[ddf["kind"] == "Calm"].duration.mean()
            calm_duration_q90 = ddf[ddf["kind"] == "Calm"].duration.quantile(0.9)
            self.calm_durations = [calm_duration_q10,calm_duration_mean,calm_duration_q90]
            print("Duration calm [{:.1f},{:.1f},{:.1f}]".format(calm_duration_q10,calm_duration_mean,calm_duration_q90))
            
            start_dates_current_waves = pd.Series(pd.to_datetime(ddf[ddf["kind"] == "Wave_act"].from_dt))
            start_dates_current_waves_q10 = start_dates_current_waves.quantile(0.1)
            start_dates_current_waves_mean = start_dates_current_waves.mean()
            start_dates_current_waves_q90 = start_dates_current_waves.quantile(0.9)
            print("Start current wave [{},{},{}]".format(start_dates_current_waves_q10,start_dates_current_waves_mean,start_dates_current_waves_q90))

            print("End current wave [{},{},{}]".format(start_dates_current_waves_mean+pd.Timedelta(days=wave_duration_q10),
                start_dates_current_waves_mean+pd.Timedelta(days=wave_duration_mean),
                start_dates_current_waves_mean+pd.Timedelta(days=wave_duration_q95),
                ))
            # I have no idea why the top version throws an error
            try:
                self.end_of_wave_2.value = (start_dates_current_waves_mean+pd.Timedelta(days=wave_duration_q95)).date()
            except:
                pass
            


        conn.close()


    def select_subregion_callback(self,attr,old,new):
        conn = self.engine.connect()
        sql = "SELECT DISTINCT country FROM neighbourhood_relations_world_region_level WHERE subregion IN ("+",".join(["'{}'".format(c) for c in new])+") ORDER BY country"
        df = pd.read_sql(sql,conn)
        self.select_country.options = list(df.country.values)
        conn.close()


    def select_continent_callback(self,attr,old,new):
        conn = self.engine.connect()
        sql = "SELECT DISTINCT subregion FROM neighbourhood_relations_world_region_level WHERE continent IN ("+",".join(["'{}'".format(c) for c in new])+") ORDER BY subregion"
        df = pd.read_sql(sql,conn)
        self.select_subregion.options = list(df.subregion.values)
        conn.close()


    def load_data(self):
        conn = self.engine.connect()
        df = pd.read_sql("SELECT DISTINCT continent FROM neighbourhood_relations_world_region_level ORDER BY continent;",conn)
        self.select_continent.options = list(df.continent.values)
        conn.close()

    def prepopulate(self):
        self.select_continent.value = ["Europe"]
        self.select_subregion.value = ["Northern Europe","Western Europe","Southern Europe"]
        self.select_country.value = ["Austria","Belgium","Denmark","Germany","Luxembourg","Netherlands","Switzerland"]


    def create(self):
        self.select_continent = MultiChoice(title="Continent",options=[""],value=[""])
        self.select_continent.on_change("value",self.select_continent_callback)

        self.select_subregion = MultiChoice(title="Subregion",options=[""],value=[""])
        self.select_subregion.on_change("value",self.select_subregion_callback)

        self.select_country   = MultiChoice(title="Country",options=[""],value=[""])
        self.select_country.on_change("value",self.select_country_callback)

        self.end_of_wave_2 = DatePicker(title="Model end of Wave 2",value=datetime.date.today())
        self.end_of_wave_2.on_change("value",self.end_of_wave_2_callback)

        self.cds = ColumnDataSource({"datetime_date":[],"new_cases":[],"trend":[]})

         # The top time series vbar and line plots
        self.p_top = figure(plot_width=1200, plot_height=225,x_axis_type='datetime',title="Cumulative Number of Cases for Selected Countries",
            tools="pan,box_zoom,box_select,reset",active_drag="box_select",
            output_backend="webgl")
        self.p_top.toolbar.logo=None
        self.p_top.vbar(x="datetime_date",top="new_cases",source=self.cds,#y_range_name="new_cases",
            #width=86400*750,alpha=0.75,color='#ffbb78',legend_label="New Cases")
            width=86400*750,alpha=0.75,color='darkblue',legend_label="New Cases")
        self.p_top.cross(x="datetime_date",y="trend",source=self.cds,#y_range_name="new_cases",
                            alpha=0.75,color="#a85232",legend_label="Trend")
        self.p_top.legend.location="top_left"
        self.p_top.legend.click_policy="hide"
        self.cds_end_of_wave_2 = ColumnDataSource({"datetime_date":[],"value":[]})
        self.p_top.line(x="datetime_date",y="value",color="tomato",source=self.cds_end_of_wave_2)
        self.p_top.circle(x="datetime_date",y="value",color="tomato",size=15,source=self.cds_end_of_wave_2)


        self.cds_votes_ranges = ColumnDataSource({"from":[],"to":[],"height":[],"color":[]})
        self.p_votes = figure(plot_width=1200, plot_height=75, x_axis_type="datetime",title="Previous Selections/Votes",
                                x_range = self.p_top.x_range,toolbar_location=None, tools="",output_backend="webgl")
        self.p_votes.hbar(left="from",right="to",y="y",height="height",source=self.cds_votes_ranges,
                            color="color",fill_alpha=0.2,line_color=None)
        self.p_votes.yaxis.visible = False
        self.p_votes.ygrid.visible = False

        tooltips_top = [("Date","@datetime_date{%F}"),
                    ("New Cases","@new_cases{0}"),
                    ("New Cases Trend","@trend{0}"),
                    #("Active Cases","@active{0}"),
                    #("Deaths","@deaths{0.00}"),
                    ]
        hover_top = HoverTool(tooltips = tooltips_top,
                          formatters={
                              "@datetime_date": "datetime",
                              }
                         )
        self.p_top.add_tools(hover_top)

        self.p_top = self.theme.apply_theme_defaults(self.p_top)
        self.p_top.background_fill_color = "#ffffff"

        self.p_votes = self.theme.apply_theme_defaults(self.p_votes)
        self.p_votes.background_fill_color = "#ffffff"

        self.cassandra = Div(text="<H1>Scenario Forecast</H1>")

        columns = [
            TableColumn(field="category",title="Episode"),
            TableColumn(field="from_dt",title="From",formatter=DateFormatter()),
            TableColumn(field="to_dt",title="To",formatter=DateFormatter()),
            ]

        self.cds_cassandra = ColumnDataSource({"category":[],"from_dt":[],"to_dt":[]})
        self.cassandra_table = DataTable(source=self.cds_cassandra,columns=columns,width=400,height=300,index_position=None)

        self.sql_connect = Div(text="<small>{}</small>".format(self.engine))

        return (column([row([
                                self.select_continent,self.select_subregion,self.select_country,self.end_of_wave_2,
                            ]),
                        self.p_top,
                        self.p_votes,
                        self.cassandra,
                        self.cassandra_table,
                        self.sql_connect,
                        ]))


# just to be safe
#Path("./data").mkdir(exist_ok=True,parents=True)
# instantiate main class
gui_health = GUIHealth()
# create widgets which will not have data to be displayed just yet
content_health = Panel(child=gui_health.create(),title="Health and Stringency")

gui_economy = GUIEconomy()
content_economy = Panel(child=gui_economy.create(),title="Economy and Economic Proxies")

gui_whatif = GUIwhatif()
content_whatif = Panel(child=gui_whatif.create(),title="Fly-Forward Modeller")

content = Tabs(tabs=[content_health,content_economy,content_whatif])

#print("gui_economy.load_data()")
gui_economy.load_data()
gui_whatif.load_data()
gui_whatif.prepopulate()
# put it on the canvas/web browser
curdoc().add_root(content)
# check if we have data...
gui_health.compute_data_status()
# load it and map it onto the plots. The latter is done by selecting a default country, which triggers the country callback
gui_health.load_data()

