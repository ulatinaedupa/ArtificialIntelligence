# Import dependencies
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd 

# Create an instance of the app 
app = Dash('Cluster Dashboard')

# Load data using pandas
df = pd.read_csv('cluster_results.csv')

# Create plots with plotly express 
fig1 = px.scatter(df, x='num_comments', y='num_reactions', color='cluster')
fig2 = px.scatter(df, x='num_shares', y='num_likes', color='cluster')
fig3 = px.scatter(df, x='num_angrys', y='num_sads', color='cluster')
fig4 = px.scatter(df, x='num_loves', y='num_wows', color='cluster')

# Create app layout 
app.layout = html.Div(
    children=[
        html.H1(children='Cluster Dashboard'), 
        html.Div(children='Showcasing your results using Dash'),
        html.Div(children=[
            html.Div(dcc.Graph(id='comments', figure=fig1), style={'display':'inline-block'}), 
            html.Div(dcc.Graph(id='shares', figure=fig2), style={'display':'inline-block'})
        ]), 
        html.Div(children=[
            html.Div(dcc.Graph(id='haha', figure=fig3), style={'display':'inline-block'}), 
            html.Div(dcc.Graph(id='loves', figure=fig4), style={'display':'inline-block'})
        ])
    ], 
    style={'font-family':'Roboto', 'margin':'0 auto', 
    'width':'1400px'}
)

# Run the application
if __name__ == '__main__': 
    app.run_server(debug=True)