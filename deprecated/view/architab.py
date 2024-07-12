# coding: utf8
import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output


class ArchiTab(dcc.Tab):

    def __init__(self, label='Archi', value='architab'):

        dcc.Tab.__init__(self, label=label, value=value)
