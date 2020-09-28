# Cookiecutter tool
#
# Copyright (C) Dr. Klaus G. Paul 2020 Rolls-Royce Deutschland Ltd & Co KG
# Made on behalf of Emergent Alliance Ltd
#
# Notice to users: This is a Minimum Viable Product designed to elicit user requirements. The code
# quality is write-once, reuse only if deemed necessary
#
from bokeh.models import Button, Plot, TextInput
from bokeh.palettes import RdYlBu3, RdYlBu11, Turbo256, Plasma256
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.models.widgets import Select
from bokeh.layouts import row, column, layout
from bokeh.models import Range1d, HoverTool, LinearAxis, Label, NumeralTickFormatter, PrintfTickFormatter, Div, LinearColorMapper, ColorBar, BasicTicker
from bokeh.models import BoxAnnotation, DataTable, DateFormatter, TableColumn, RadioGroup, Spinner, Paragraph, RadioButtonGroup
from bokeh.palettes import Category20, Category10, RdYlBu3, Greys4
from bokeh.client.session import push_session, show_session
from bokeh.events import SelectionGeometry, ButtonClick
from bokeh.transform import transform
import datetime
import pandas as pd
import os
import sys
import gzip
import json
import pickle
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import base64
from pathlib import Path
import pycountry
import socket
import random


class GUI():
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
        self.set_theme("RR")
        self.data_status = "no_data" # "stale", "current"
        pass

    def set_theme(self,theme="RR"):
        """This is used to set some colors of the UI. Note there is another aspect to this
        in the cookiecutter/templates/index.html file which specifies CSS styles for
        UI elements that are trickier to override.
        """
        if theme == "RR":
            self.background_color="#00498F"
            self.background_fill_color="#00498F"
            self.plot_color="#FFFFFF"
            self.text_font="NotoSans"
            self.text_color="#FFFFFF"
            self.tick_line_color="#FFFFFF"
            self.axis_label_text_color="#FFFFFF"
            self.axis_line_color="#FFFFFF"
            self.label_text_color="#FFFFFF"
            self.tick_line_color="#FFFFFF"
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


    def refresh_data_callback(self,event):
        """This handles downloading data and recomputing some of the derived values.
        The idea was to provide user feedback via a progress bar, unfortunately, bokeh
        does not allow for UI updates while handling a callback (which makes a lot of sense,
        but is unfortunate in our case).
        """
        print("REFRESH")
        self.download_data()
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
            dates.append(self.cds.data["date"][i])
        selection = []
        i = 0
        for d in self.cds_OxCGRTHeatmap.data["date"]:
            if d in dates:
                selection.append(i)
            i += 1
        self.cds_OxCGRTHeatmap.selected.indices = selection

    def compute_metrics(self,bins=25):
        """using self.dfVotesContent, this computes stats for display
        """
        if len(self.dfVotesContent) > 1:
            ddf = self.dfVotesContent[["from","to","user","kind","filename","rel_peak_new_cases","duration"]].drop_duplicates()

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
        df = self.cds.to_df().iloc[self.cds.selected.indices]
        df["rel_peak_new_cases"] = max(df["new_cases_rel"])
        df["kind"] = self.scenario_type.labels[self.scenario_type.active]
        df["kind_counter"] = int(self.scenario_number.value)
        df["from"] = df.date.min()
        df["to"] = df.date.max()
        df["duration"] = (pd.to_datetime(df.date.max())-pd.to_datetime(df.date.min())).total_seconds()/86400
        df["user"] = self.user_id.value
        ADM0_A3 = self.dfMapping[self.dfMapping.name == self.country_select.value].ADM0_A3.values[0]
        df["ADM0_A3"] = ADM0_A3
        filename = "data/{}.{}.{}.{}.{:%Y%m%d}.{:%Y%m%d}.{}.csv".format(ADM0_A3,
                self.country_select.value.replace("*","").replace("'","_"), # Taiwan* Cote d'Ivoire
                self.scenario_type.labels[self.scenario_type.active],self.scenario_number.value,
                pd.to_datetime(df.date.values[0]),
                pd.to_datetime(df.date.values[-1]),
                self.user_id.value)
        df.to_csv(filename,index=False)
        # reset selection
        self.cds.selected.indices = []
        # update message field
        self.progress_bar_info_message.text = "Saved selection to {}".format(filename)
        self.progress_bar_data.data["color"] = ["#389c6b"]
        # reset scenario field
        self.scenario_name.value = self.country_select.value + " wave calm #"

        self.dfVotesContent = self.dfVotesContent.append(df)
        #self.dfVotesContent[self.dfVotesContent.infection_rate_7 > 1000.] = 0.
        self.dfVotesContent["filename"] = filename
        self.dfVotesContent.to_pickle("./data/votes.pickle",protocol=3)
        #print(self.dfVotesContent)
        self.compute_metrics()


    def change_country(self,attr,old,new):
        """Handle change of country to be displayed.

        This generates a dict style data structure that can be used to overwrite the various ColumnDataSource s
        that are used to display the values. This is the common bokeh pattern to update a display.

        Commonly made mistake is to re-create a new ColumnDataSource and try to squeeze in a pandas
        dataframe directly, this, however, will not update the plot.
        """
        # fresh data for the time series plots
        newdata = {}
        for column in self.cds.data.keys():
            if column == "index" or column == "Timestamp":
                newdata[column] = self.adfCountryData[new].index
            else:
                newdata[column] = np.nan_to_num(self.adfCountryData[new][column])
        self.cds.data = newdata # overwrite
        # reset scenario text with country name
        #self.scenario_name.value = new+" wave calm #"
        self.scenario_number.value = 1
        self.scenario_type.active = 0
        # rescale the y axes that require it
        df = self.cds.to_df()
        self.p_top.extra_y_ranges["active"].end = df.active.dropna().values.max()*1.1
        self.p_top.extra_y_ranges["new_cases"].end = df.new_cases.dropna().values.max()*1.1

        # now the same thing for the OxCGRT heatmap
        newdata = {}
        for column in self.cds_OxCGRTHeatmap.data.keys():
            if column == "index" or column == "Timestamp":
                newdata[column] = self.adfMeasures[new].index
            else:
                newdata[column] = np.nan_to_num(self.adfMeasures[new][column])
        self.cds_OxCGRTHeatmap.data = newdata

        # now revisit the background boxes, initially make them all invisible
        for b in self.wave_boxes:
            b.visible = False
            b.fill_color = "#FFFFFF"

        # each background box is a BoxAnnotation. Loop through the episode data and color/display them as required.
        dfWaves = self.adfWaves[new]
        last = "new"
        box_no = 0
        for i,row in dfWaves.iterrows():
            if row["kind"] == "begin":
                left = row["date"]
                if last == "end":
                    self.wave_boxes[box_no].left = right
                    self.wave_boxes[box_no].right = left
                    self.wave_boxes[box_no].visible = True
                    self.wave_boxes[box_no].fill_color = "#00FF00"
                    self.wave_boxes[box_no].fill_alpha = 0.05
                    box_no += 1
                last = "begin"
            elif row["kind"] == "end":
                right = row["date"]
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
        if len(self.dfVotesContent)>0:
            ADM0_A3 = self.dfMapping[self.dfMapping.name == new].ADM0_A3.values[0]
            ddf = self.dfVotesContent[self.dfVotesContent.ADM0_A3 == ADM0_A3][["from","to","kind"]].copy().drop_duplicates()
            ddf["from"] = pd.to_datetime(ddf["from"])
            ddf["to"] = pd.to_datetime(ddf["to"])
            ddf["color"] = None
            ddf.loc[ddf["kind"] == "Wave","color"] = "tomato"
            ddf.loc[ddf["kind"] == "Calm","color"] = "mediumseagreen"
            ddf["height"] = [random.random()/3+0.1 for i in ddf.index]
            ddf["y"] = [0.5 for i in ddf.index]
            self.cds_votes_ranges.data = {"from":ddf["from"].values,"to":ddf.to.values,"y":ddf.y.values,
                                            "height":ddf.height.values,"color":ddf.color.values}


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
        self.country_select = Select(options=[""])
        self.country_select.on_change("value",self.change_country)
        self.scenario_name = TextInput(value="country wave 2")
        self.scenario_type_label = Paragraph(text="Select Class")
        #self.scenario_type = RadioGroup(labels=["Wave","Calm"],width=100)
        self.scenario_type = RadioButtonGroup(labels=["Wave","Calm"],width=100,active=0)
        self.scenario_number_label = Paragraph(text=" Occurence #")
        self.scenario_number = Spinner(low=1, step=1, value=1,width=50)
        self.save = Button(label="Save",disabled=True,width=150)
        self.save.on_event(ButtonClick, self.save_callback)
        
        # fields of the OxCGRT dataset we actually use
        self.fields_of_interest = ['C1_School closing','C2_Workplace closing','C3_Cancel public events','C4_Restrictions on gatherings',
                              'C6_Stay at home requirements','C7_Restrictions on internal movement',
                              'C8_International travel controls', 'E1_Income support', 'E2_Debt/contract relief', 'E3_Fiscal measures',
                              'E4_International support', 'H1_Public information campaigns','H2_Testing policy', 'H3_Contact tracing',
                              'H4_Emergency investment in healthcare', 'H5_Investment in vaccines']
        self.cds_OxCGRTHeatmap = ColumnDataSource(pd.DataFrame({"date":[],"class":[],"level":[]}))

        # the main time series data source, created empty initially
        empty_data = dict(zip(self.fields_of_interest,[[] for i in range(len(self.fields_of_interest))]))
        for c in ['date','confirmed','deaths','recovered','active','new_cases','trend','stringency_index','infection_rate_7','new_cases_rel']: 
            empty_data[c] = []
        #self.cds = ColumnDataSource(pd.DataFrame({'date':[], 'confirmed':[], 'deaths':[], 'recovered':[], 'active':[], 'new_cases':[],
        #       'trend':[], 'stringency_index':[],'infection_rate_7':[], 'new_cases_rel':[], 
        #       }))
        self.cds = ColumnDataSource(empty_data)

        # May need a different name for "votes", this is the data structure that captures the labelling activity
        self.cds_votes = ColumnDataSource({"country":[],"waves":[],"complete":[],"votes":[],"need_vote":[]})

        # The backdrops
        self.wave_boxes = [BoxAnnotation(left=None,right=None,visible=False,fill_alpha=0.1,fill_color="#FFFFFF") for i in range(10)]

        # Previous votes
        self.cds_votes_ranges = ColumnDataSource({"from":[],"to":[],"y":[],"height":[],"color":[]})

        # The wave and calm histograms
        self.cds_wave_duration_histogram = ColumnDataSource({"x":[],"y":[],"color":[],"width":[]})
        self.cds_calm_duration_histogram = ColumnDataSource({"x":[],"y":[],"color":[],"width":[]})

        # Heatmap
        self.cds_votes_heatmap = ColumnDataSource({"n":[],"rpc":[],"d":[],"h":[],"w":[]})

        # The top time series vbar and line plots
        self.p_top = figure(plot_width=1200, plot_height=225,x_axis_type='datetime',title="Select Wave/Calm regions here",
            tools="pan,box_zoom,box_select,reset",active_drag="box_select",
            output_backend="webgl")
        self.p_top.toolbar.logo=None
        self.p_top.yaxis.visible=False

        self.p_top.extra_y_ranges = {"new_cases": Range1d(start=0,end=100),
                                     "active": Range1d(start=0,end=100)}
        self.p_top.vbar(x="date",top="new_cases",source=self.cds,y_range_name="new_cases",
            width=86400*750,alpha=0.75,color='#ffbb78',legend_label="New Cases")
        self.p_top.cross(x="date",y="trend",source=self.cds,y_range_name="new_cases",
                            alpha=0.75,color="#a85232",legend_label="Trend")
        self.p_top.add_layout(LinearAxis(y_range_name="new_cases"), 'left')

        self.p_top.line(x="date",y="active",source=self.cds,y_range_name="active",
            color="#1f77b4",legend_label="Active Cases")
        self.p_top.add_layout(LinearAxis(y_range_name="active"), 'left')
        self.p_top.legend.location="top_left"
        self.p_top.legend.click_policy="hide"
        self.p_top.yaxis[1].formatter=PrintfTickFormatter(format="%i")
        self.p_top.yaxis[2].formatter=PrintfTickFormatter(format="%i")

        tooltips_top = [("date","@date{%Y-%m-%d}"),
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
        self.p_stringency.step(x="date",y="stringency_index",source=self.cds,color="#000000",legend_label="Stringency Index")
        #self.p_stringency.legend.location = "top_left"
        #self.p_stringency.legend.label_text_font="NotoSans"
        self.p_stringency.add_tools(hover_top)

        for i in range(len(self.wave_boxes)):
            self.p_stringency.add_layout(self.wave_boxes[i])

        # The OxCGRT detail heatmap
        colors = list(reversed(Greys4)) 
        mapper = LinearColorMapper(palette=colors, low=0, high=4)
        self.p_hmap = figure(plot_width=1200, plot_height=300, x_axis_type="datetime",title="Stringencyi/Measures Detail",
                            x_range = self.p_top.x_range,
                            y_range=list(reversed(self.fields_of_interest)), toolbar_location=None, tools="", 
                            output_backend="webgl")

        self.p_hmap.rect(y="class", x="date", width=86400000, height=1, source=self.cds_OxCGRTHeatmap,
                            line_color=None, fill_color=transform('level', mapper))

        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=4),
                             formatter=PrintfTickFormatter(format="%d"))
        color_bar.background_fill_color = self.background_color
        color_bar.major_label_text_color = "#FFFFFF"

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


        for i in range(len(self.wave_boxes)):
            self.p_hmap.add_layout(self.wave_boxes[i])

        # The data table that displays stats for countries and how many datasets have been labelled
        columns = [
            TableColumn(field="country",title="Country"),
            TableColumn(field="waves",title="Waves"),
            TableColumn(field="complete",title="Eposides"),
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
        self.progress_bar = self.apply_theme_defaults(self.progress_bar)
        self.country_select = self.apply_theme_defaults(self.country_select)
        self.scenario_name = self.apply_theme_defaults(self.scenario_name)
        self.save = self.apply_theme_defaults(self.save)
        self.p_top = self.apply_theme_defaults(self.p_top)
        self.p_top.background_fill_color = "#ffffff"
        self.p_stringency = self.apply_theme_defaults(self.p_stringency)
        self.p_stringency.background_fill_color = "#ffffff"
        self.p_hmap = self.apply_theme_defaults(self.p_hmap)
        self.p_hmap.background_fill_color = "#ffffff"
        self.voting_table = self.apply_theme_defaults(self.voting_table)
        self.p_votes = self.apply_theme_defaults(self.p_votes)
        self.p_votes.background_fill_color = "#ffffff"


        self.p_histogram_wave = self.apply_theme_defaults(self.p_histogram_wave)
        self.p_histogram_wave.background_fill_color = "lightsteelblue"
        self.p_histogram_calm = self.apply_theme_defaults(self.p_histogram_calm)
        self.p_histogram_calm.background_fill_color = "lightsteelblue"
        self.p_duration_heatmap = self.apply_theme_defaults(self.p_duration_heatmap)
        self.p_duration_heatmap.background_fill_color = "lightsteelblue"

        self.user_id = TextInput(value="nobody@{}".format(socket.gethostname()),title="Name to save your results")

        # The return value is a row/column/widget array that defines the arrangement of the various elements.
        return(row([column([self.progress_bar,
                            row([self.blank,
                                self.refresh_data,
                                self.country_select,
                                self.scenario_type_label,
                                self.scenario_type,
                                self.scenario_number_label,
                                self.scenario_number,
                                self.save]),
                        self.p_top,
                        self.p_stringency,
                        self.p_hmap,
                        self.p_votes,
                            row([Div(text='<div class="horizontalgap" style="width:200px"><h2>Statistics</h2></div>'),
                                self.p_histogram_wave,self.p_histogram_calm,self.p_duration_heatmap]),
                        ]),
                    column([self.user_id,
                        self.voting_table]),
                    ]))


    def download_data(self):
        """This downloads directly, from github, the Johns Hopkins and Oxford university data sets. Not error handling is performed
        as it is unclear as to how to proceed in these cases (and no existing data would be overwritten)

        Note this code still contains progress bar code which will not cause visual feedback, as this code
        is called during a callback handling routine which prevents GUI updates. I left it in as this could change
        in a future version breaking the agile rule of no provision for future growth and the sensible rule
        of not having untested code.
        """
        saved_color = self.progress_bar_data.data["color"]
        self.progress_bar_data.data["color"] = ["#FFFFFF"]
        self.progress_bar_data.data["right"][0] = 1.
        self.dfConfirmed = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",low_memory=False)
        self.progress_bar_data.data["color"] =  saved_color
        self.progress_bar_data.data["right"][0] = 0.2
        self.dfDeaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",low_memory=False)
        self.progress_bar_data.data["right"][0] = 0.4
        self.dfRecovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv",low_memory=False)
        self.progress_bar_data.data["right"][0] = 0.6
        self.dfMapping = pd.read_csv("https://github.com/rolls-royce/EMER2GENT/raw/master/data/sun/geo/country_name_mapping.csv",low_memory=False)
        self.progress_bar_data.data["right"][0] = 0.8
        self.dfOxCGRT = pd.read_csv("https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv",low_memory=False)
        self.progress_bar_data.data["right"][0] = 1.
        dfPopulation = pd.read_excel("https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/1_Population/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx",
                            sheet_name="ESTIMATES",skiprows=16,usecols="E,BZ")
        alldata = []
        for i,row in dfPopulation.iterrows():
            try:
                result = pycountry.countries.get(numeric="{:03d}".format(row["Country code"]))
            except:
                print(row["Country code"],end="..")
                continue
            if result:
                alldata.append({"ADM0_A3":result.alpha_3,"population":row["2020"]*1000})
            else:
                try:
                    result = pycountry.countries.search_fuzzy(row["Region, subregion, country or area *"])
                    print(row["Country code"],result,end="..")
                    alldata.append({"ADM0_A3":result.alpha_3,"population":row["2020"]*1000})
                except:
                    continue
        self.dfPopulation = pd.DataFrame(alldata)
        self.blob = {"last_update":"{}".format(datetime.datetime.now())}
                

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
            print("{} POPULATION {}".format(ADM0_A3,population))
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
        with gzip.open("data/datafile.pckld.gz","w+t") as f:
            self.blob["data"] =base64.b64encode(pickle.dumps(self.adfCountryData,protocol=4)).decode("ascii")
            self.blob["mapping"] = base64.b64encode(pickle.dumps(self.dfMapping,protocol=4)).decode("ascii")
            self.blob["stringency"] = base64.b64encode(pickle.dumps(self.adfOxCGRT,protocol=4)).decode("ascii")
            self.blob["measures"] = base64.b64encode(pickle.dumps(self.adfMeasures,protocol=4)).decode("ascii")
            self.blob["waves"] = base64.b64encode(pickle.dumps(self.adfWaves,protocol=4)).decode("ascii")
            self.blob["population"] = base64.b64encode(pickle.dumps(self.dfPopulation,protocol=4)).decode("ascii")
            data = json.dumps(self.blob)
            f.write(data)


    def compute_data_status(self):
        """Determine if data are stale, or don't even exist. Staleness is determined by a "last download" item.
        """
        datafile = "data/datafile.pckld.gz"
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
        

    def load_data(self):
        """Loading the data but also, as a temprary fix, checking which user files can be found in cookiecutter/data/*.csv
        TODO: The latter needs to be ported to SQL
        """
        print(self.data_status)
        if self.data_status == "no_data":
            self.dfVotesContent = pd.DataFrame()
            return
        else:
            with gzip.open("data/datafile.pckld.gz","rt") as f:
                data = f.read()
                data = json.loads(data)
                self.adfCountryData = pickle.loads(base64.b64decode(data["data"]))
                self.dfMapping = pickle.loads(base64.b64decode(data["mapping"]))
                self.adfOxCGRT = pickle.loads(base64.b64decode(data["stringency"]))
                self.adfMeasures = pickle.loads(base64.b64decode(data["measures"]))
                self.adfWaves = pickle.loads(base64.b64decode(data["waves"]))
                self.dfPopulation = pickle.loads(base64.b64decode(data["population"]))
            self.country_select.options=sorted(self.adfCountryData.keys())
            self.sort_countries_by_relevance()

            alldata = []
            for country in self.adfWaves.keys():
                dfWave = self.adfWaves[country]
                if len(dfWave) > 0:
                    alldata.append({"country":country,
                                    "waves":len(dfWave[dfWave.kind == "begin"]),
                                    "complete":len(dfWave[dfWave.kind == "end"]),
                                    "votes":0,
                                    "need_vote":True})
                else:
                    alldata.append({"country":country,
                                    "waves":0,
                                    "complete":0,
                                    "votes":0,
                                    "need_vote":False})
            self.dfVotes = pd.DataFrame(alldata).sort_values(["complete","country"],ascending=[False,True])
            newdata = {}
            files = os.listdir("./data")
            votes = {}
            alldata = []
            for f in files:
                if f.endswith(".csv"):
                    ddf = pd.read_csv("./data/"+f)
                    ddf["filename"] = f
                    alldata.append(ddf)
                    fields = f.split(".")
                    if len(fields)>=3:
                        country = fields[1]
                        if country in votes.keys():
                            votes[country] += 1
                        else:
                            votes[country] = 1
            for i,row in self.dfVotes.iterrows():
                if row["country"] in votes.keys():
                    self.dfVotes.at[i,"votes"] = votes[row["country"]]
                    if votes[row["country"]] < row["waves"]+row["complete"]:
                        self.dfVotes.at[i,"need_vote"] = False
            for c in self.dfVotes.columns:
                newdata[c] = self.dfVotes[c].values
            self.cds_votes.data = newdata
            self.dfVotesContent = pd.DataFrame().append(alldata)
            self.dfVotesContent.to_pickle("./data/votes.pickle",protocol=3)
            self.compute_metrics()

            self.country_select.value = "Germany"
            self.cds.selected.on_change('indices',self.on_selection_change_callback)
        pass

# just to be safe
Path("./data").mkdir(exist_ok=True,parents=True)
# instantiate main class
gui = GUI()
# create widgets which will not have data to be displayed just yet
content = gui.create()
# put it on the canvas/web browser
curdoc().add_root(content)
# check if we have data...
gui.compute_data_status()
# load it and map it onto the plots. The latter is done by selecting a default country, which triggers the country callback
gui.load_data()

