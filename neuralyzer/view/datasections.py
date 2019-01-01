# coding: utf8
import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output


class FileSection(html.Div):

    def __init__(self, id='DataFileSection'):

        html.Div.__init__(self, id=id)

        # I'll put dcc objects in children

        # append title (H3 html text)
        self.children.append(html.H3('Files'))

        # append subtitle (H2 html text)
        self.children.append(html.H2('Training'))

        upload_style = {'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'}

        # append file upload from dcc for training files
        # a div, with dcc upload and another div as children
        train_uploader = dcc.Upload(id='TrainingUploader', children=html.Div(['Drag and Drop or ', html.A('Select Files')]), style=upload_style)
        train_foldername = html.Div(id='TrainingFolderName')
        self.children.append(html.Div(id='Training', children=[train_uploader, train_foldername]))

        # append subtitle (H2 html text)
        self.children.append(html.H2('Training'))

        # append file upload from dcc for validation files
        # a div, with dcc upload and another div as children
        valid_uploader = dcc.Upload(id='ValidationUploader', children=html.Div(['Drag and Drop or ', html.A('Select Files')]), style=upload_style)
        valid_foldername = html.Div(id='ValidationFolderName')
        self.children.append(html.Div(id='Validation', children=[train_uploader, train_foldername]))


class PreprocessingSection(html.Div):

    def __init__(self, id='DataPreprocessingSelection'):

        html.Div.__init__(self, id=id)

        # append title (H3 html text)
        self.children.append(html.H3('Preprocessing'))

        dropoptions = [{'label': 'normalize', 'value': 'NORM'}]
        drop = dcc.Dropdown(id='PreprocessingChoice', options=dropoptions, multi=True, value='NORM')
        descriptions = html.Div(id='PreprocessingDescription')
        self.children.append(html.Div(id='Preprocessing', children=[drop, descriptions]))


class LabelSection(html.Div):

    def __init__(self, id='DataLabelSelection'):

        html.Div.__init__(self, id=id)

        # append title (H3 html text)
        self.children.append(html.H3('Labels'))

        dropoptions = [{'label': 'Keras Generator', 'value': 'KERAS'}]
        drop = dcc.Dropdown(id='LabelGeneratorChoice', options=dropoptions, value='KERAS')
        description = html.Div(id='LabelGeneratorDescription')
        self.children.append(html.Div(id='LabelGenerator', children=[drop, description]))
