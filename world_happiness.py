#import libaries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff 
import io

#import data
@st.cache_data
def imp():
    df1=pd.read_csv("world-happiness-report.csv")
    df2=pd.read_csv("world-happiness-report-2021.csv")
    df_new=pd.read_csv("df_new.csv")
    pred_results=pd.read_csv("pred_results.csv")
    rfr_results=pd.read_csv("rfr_results.csv")
    feat_imp_rfr=pd.read_csv("feat_imp_rfr.csv")
    return (df1, df2, df_new, pred_results, rfr_results, feat_imp_rfr)

df1, df2, df_new, pred_results, rfr_results, feat_imp_rfr = imp()

#title
st.title("World Happiness Project")

#sidebar content
st.sidebar.title("Menu")
pages=["About the Project", "Used Datasets", "My EDA", "How is the ML model?", "The End"]
page=st.sidebar.radio("Go to", pages)

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")

variables=["Happiness Score", "GDP", "Social Support", "Life Expectancy", "Freedom", "Generosity", "Corruption"]

st.sidebar.title("Data Limits")
selected_var_box = st.sidebar.selectbox("Choose a variable to visualize:", variables)

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")

st.sidebar.title("Impressum")
st.sidebar.markdown(
    """
    <div style="font-size: 2px; style="line-height: 1.5;">
        Author: Sebastian HARTEL<br>
        Guided by: Lucas VARELA
    </div>
    """, unsafe_allow_html=True
)

#for a better viz
translate={"Happiness Score": "life_ladder", 
           "GDP": "log_gdp_per_capita",
           "Social Support": "social_support", 
           "Life Expectancy": "healthy_life_expectancy_at_birth", 
           "Freedom": "freedom_to_make_life_choices", 
           "Generosity": "generosity",
           "Corruption": "perceptions_of_corruption"}

selected_var = translate.get(selected_var_box)
translabel = selected_var_box

translate2={"life_ladder": "Happiness Score", 
            "log_gdp_per_capita": "GDP",
            "social_support": "Social Support", 
            "healthy_life_expectancy_at_birth": "Life Expectancy", 
            "freedom_to_make_life_choices": "Freedom", 
            "generosity": "Generosity",
            "perceptions_of_corruption": "Corruption", 
            "positive_affect": "Positive Affect", 
            "negative_affect": "Negative Affect",
            "regional_indicator": "Region",
            "country_name": "Country",
            "abs": "Abs. Diff",
            "life_ladder_cat": "Cluster",
            "0": "PCA 1",
            "1": "PCA 2",
            "year": "Year",
            "cluster": "Cluster",
            "rel": "in %",
            "Unnamed: 0": "Feature",
            "regional_indicator_Latin America and Caribbean": "Region Latin America & Caribbean"}

color={ "Central and Eastern Europe": px.colors.qualitative.Plotly[0],
        "Commonwealth of Independent States": px.colors.qualitative.Plotly[1],
        "East Asia": px.colors.qualitative.Plotly[2],
        "Latin America and Caribbean": px.colors.qualitative.Plotly[3],
        "Middle East and North Africa": px.colors.qualitative.Plotly[4],
        "North America and ANZ": px.colors.qualitative.Plotly[5],
        "South Asia": px.colors.qualitative.Plotly[6],
        "Southeast Asia": px.colors.qualitative.Plotly[7],
        "Sub-Saharan Africa": px.colors.qualitative.Plotly[8],
        "Western Europe": px.colors.qualitative.Plotly[9]}

color1=px.colors.diverging.RdYlGn
color1_r=px.colors.diverging.RdYlGn_r

color_scale=[
        [0, "rgb(251,106,74)"],
        [0.4, "rgb(247,254,174)"],
        [0.5, "rgb(0,109,44)"],
        [0.6, "rgb(247,254,174)"],
        [1, "rgb(251,106,74)"]
        ]

num_col = {"life_ladder", "log_gdp_per_capita", "social_support", "healthy_life_expectancy_at_birth", "freedom_to_make_life_choices",
           "generosity", "perceptions_of_corruption"}

@st.cache_data
def tick(selected_var):
    without_dec=["life_ladder", "log_gdp_per_capita", "healthy_life_expectancy_at_birth"]
    with_dec=["social_support", "freedom_to_make_life_choices", "generosity", "perceptions_of_corruption"]

    if selected_var in without_dec:
        axis_tickformat=".f"
    else:
        axis_tickformat=".1f"
    return axis_tickformat


if page == pages[0]:
    st.subheader("Context")
    text="""
    The World Happiness Report is a landmark survey of the state of global happiness. The report continues to gain global 
    recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making
    decisions. Leading experts across fields - economics, psychology, survey analysis, national statistics, health, public policy 
    and more - describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review 
    the state of happiness in the world today and show how the new science of happiness explains personal and national variations 
    in happiness.
    """
    st.write(text)
    st.write(" ")
    
    st.subheader("Objectives")
    text2="""
    The objectives of this project are multifaceted. Firstly, it aims to analyze the rankings of various 
    variables across countries or regions, identifying which countries have experienced the 
    most significant changes over time and uncovering trends in happiness. Secondly, the project seeks 
    to investigate the most significant factors influencing happiness. Finally, it involves developing 
    a supervised machine learning model to predict happiness scores.
    """
    st.write(text2)


if page == pages[1]:
    st.subheader("Raw data")
    text3="""
    The happiness scores and rankings use data from the Gallup World Poll. In 2005, Gallup began its 
    World Poll, which continually surveys citizens in 160 countries, representing more than 98% of the 
    world´s adult population. The Gallup World Poll consists of more than 100 global questions as well as 
    region-specific items.  
    The columns following the happiness score estimate the extent to which each of six factors – 
    economic production, social support, life expectancy, freedom, absence of corruption, and generosity."""
    st.write(text3)
    
    st.markdown("""
    For this project, I have used the free available data sourced on Kaggle [click here to visit Kaggle](https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021) with a CSV-file spanning
    observations from 2005 to 2020 and a already analyzed file with data from 2021.
    """)    
    
    with st.expander("df1 'world-happiness-report'", expanded=False):
        st.dataframe(df1, height=250)
    with st.expander("df2 'world-happiness-report-2021'", expanded=False):
        st.dataframe(df2, height=250)
    st.write(" ")

    st.subheader("Merged data")
    st.write("""The merged data contains 2098 rows and 12 columns.""")
    st.dataframe(df_new, height=250)
    st.write(" ")
    
    buffer = io.StringIO()
    df_new.info(buf=buffer)
    info = buffer.getvalue()
    st.write("""Following information about type and missing values:""")
    st.text(info)
    st.write(" ")

    st.write("""Below is a statistical description of the numerical variables:""")
    df_new_col=df_new.iloc[:,2:]
    st.dataframe(df_new_col.describe())
    st.write(" ")

    st.write("""The life ladder serves as our target variable also called as happiness score, 
             while the others will function as our explanatory variables in the ML models.""")

    with st.expander("Read me for more details about the variables", expanded=False):
        text4="""
        <u>GDP:</u> <br>Country´s GDP divided by its total (mid-year) population.

        <u>Social Support:</u><br> 
        It refers to assistance or support provided by members of social networks to an individual. 
        If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not? 
        National average of the binary responses (0 = no, 1 = yes) to the question of social support.

        <u>Life Expectancy:</u><br> 
        It is the average life in good health - that is to say without 
        irreversible limitation of activity in daily life or incapacities - of a fictitious generation subject to the conditions 
        of mortality and morbidity prevailing that year. World Health Organization´s (WHO): Data at source are available for the 
        years 2000, 2010, 2015 and 2019. To match with the sample periods, interpolation and extrapolation are used.

        <u>Freedom:</u><br> 
        Personal freedom to make key life decisions (democracy, authoritarianisms). Are you satisfied 
        or dissatisfied with your freedom to choose what you do with your life? National average of the binary responses (0 = no, 1 = yes)
        to the question of freedom.

        <u>Generosity:</u><br> 
        How generous are people in terms of donations, volunteering, and offering help? These activities are indicators of how well 
        the community is functioning and whether people are connecting with each other. Consider the question: "Have you donated money 
        to a charity in the past month?" We examine the residuals from regressing the national average responses to this generosity 
        question.

        <u>Corruption:</u><br> 
        Is there corruption in governance and businesses? Do people trust their government to make the right decisions? Can we rely 
        on the legal system? National average of the binary responses (0 = no, 1 = yes)
        to the question of corruption."""
    
        st.markdown(text4, unsafe_allow_html=True)

if page == pages[2]:
    fol1, fol2, fol3 = st.tabs(["Distributions", "Rankings & Trends", "Feature Importance & Cluster"])

    with fol1:
        #FIGURE1
        @st.cache_data
        def fig1(df_new):
            df_fig1=df_new.sort_values(by="regional_indicator")
            return(df_fig1)

        df_fig1=fig1(df_new)
        fig = px.histogram(df_fig1, 
                           x="year", 
                           color="regional_indicator", 
                           title="Frequency Distribution by Regions",
                           color_discrete_map=color,
                           labels={"year": "Year", "regional_indicator": "Region"}
        )
        st.plotly_chart(fig)
        st.write("""
        The report began with a few countries, showing an initial rise in distribution. Over time, its significance increased, with a yearly survey covering over 140 countries, a figure that has remained stable. 
        In 2020, there was a notable decline likely due to COVID and stricter regulations.""")
        st.write(" ")

        #FIGURE3
        @st.cache_data
        def fig3(df_new, selected_var):
            df_fig3=df_new.dropna(axis=0, subset=selected_var)
            return df_fig3

        df_fig3=fig3(df_new, selected_var)
        fig = ff.create_distplot([df_fig3[selected_var]], [selected_var], show_hist=False)
        fig.update_layout(title="Frequency Distribution of selected [Variable]", 
                          autosize=True, 
                          showlegend=False, 
                          xaxis_title=f"{translabel}", 
                          yaxis_title="Density"
        )
        fig.update_traces(hovertemplate=f'{translabel}: %{{x:.2f}}<br>Density: %{{y:.2f}}')

        st.plotly_chart(fig)
        st.write("""
                 Since our variables do not follow a normal distribution, the mean is not a suitable 
                 measure of central tendency. The median is preferred as it is less influenced by 
                 extreme values and the distribution shape. For missing values, using the median or 
                 other robust measures ensures data integrity. This characteristic also affects model 
                 selection and indicates the presence of outliers. Additionally, generosity values have 
                 changed over time.""")
        st.write(" ")


        #FIGURE4
        yaxis_tickformat=tick(selected_var)

        fig = px.box(fig1(df_new), 
                     x="regional_indicator", 
                     y=selected_var, 
                     hover_data=["country_name", "year"], 
                     title="Boxplots of selected [Variable] across Regions", 
                     color="regional_indicator", 
                     color_discrete_map=color,
                     labels={selected_var: translabel, "regional_indicator": "Region", "country_name": "Country", "year": "Year"}
        )
        fig.update_layout(autosize=True, 
                          xaxis=dict(title="", showticklabels=False), 
                          yaxis_tickformat=yaxis_tickformat
        )
        st.plotly_chart(fig)
        st.write("""
        From a statistical point of view, and as shown in the boxplot, the dataset contains many extreme values and outliers. 
        Some of these values are true. For instance, Haiti recorded the smallest value of 32.3 years for life expectancy 
        in 2010, which is noteworthy considering the significant earthquake that struck Haiti, destroying crucial infrastructure 
        such as hospitals. Other values seems to be false, such as the small perception of corruption in Rwanda probably caused of a measurement error. 
        Given to our small data size and the fact that our chosen model is less sensitive to outliers/extreme values, removing them is not necessary.
        """)


    with fol2:
        #FIGURE5
        num_countries=st.select_slider(" ", options=["highest", "lowest"])
        
        @st.cache_data
        def fig5(df_new, selected_var, num_countries):
            df_fig5=df_new.dropna(axis=0, subset=selected_var)
            df_fig5=df_fig5.groupby(by=["country_name", "regional_indicator"])[selected_var].median().round(2)
            df_fig5=df_fig5.reset_index()
            df_fig5=df_fig5.sort_values(by=selected_var, ascending=False)        

            if num_countries == "highest":
                df_countries=df_fig5.head(15)
                title="Ranking of the highest Countries of selected [Variable] over time<br><sup>measured by Median"
            else:
                df_countries=df_fig5.tail(15)
                title="Ranking of the lowest Countries of selected [Variable] over time<br><sup>measured by Median"

            return (df_countries, title, selected_var)

        df_countries, title, selected_var=fig5(df_new, selected_var, num_countries)
        yaxis_tickformat=tick(selected_var)

        if selected_var == "generosity":
            st.warning("Please select another variable")
        else:
            fig = px.bar(df_countries, 
                        x="country_name", 
                        y=selected_var, 
                        color="regional_indicator", 
                        color_discrete_map=color, 
                        category_orders={"country_name": df_countries["country_name"].values}, 
                        labels=translate2
            )
            y_axis_min = df_countries[selected_var].min() - df_countries[selected_var].min()*0.05
            y_axis_max = df_countries[selected_var].max()
            yaxis=dict(range=[y_axis_min, y_axis_max])
            fig.update_layout(autosize=True, 
                                title=title, 
                                xaxis_title="",
                                yaxis=yaxis
            )
            st.plotly_chart(fig)
            st.write("""Over the past decade, regions such as "Western Europe" and "North America & ANZ" consistently report higher levels of happiness. Conversely, many parts of Africa struggle with significant 
                    challenges that impact the overall happiness and well-being of their people.""")
            st.write(" ")
        

        #FIGURE7
        st.write("""""")
        slider_fig2=st.select_slider(" ", options=["best", "worst"])

        @st.cache_data
        def fig7(df_new, selected_var, slider_fig2):
            df_fig7=df_new[["country_name", "regional_indicator", "year", selected_var]]
            df_fig7[selected_var] = df_fig7[selected_var].round(2)
            df_fig7=df_fig7.loc[(df_fig7["year"] == 2011) | (df_fig7["year"] == 2021)]

            df_pivot=df_fig7.pivot(index=['country_name', 'regional_indicator'], columns='year', values=selected_var).reset_index()
            df_pivot.columns = ['country_name', 'regional_indicator', "2011", "2021"]
            df_pivot=df_pivot.dropna()
                    
            df_pivot["abs"]=round(df_pivot["2021"] - df_pivot["2011"], 1)
            df_pivot["rel"]=round((df_pivot["abs"]/df_pivot["2011"])*100, 0)
            df_pivot["rel2"]=df_pivot["rel"]
            df_pivot=df_pivot.sort_values(by="rel", ascending=True)

            if slider_fig2 == "best":
                color_fig7=px.colors.sequential.Greens
                if selected_var == "perceptions_of_corruption":
                    df_pivot=df_pivot.head(6)
                    df_pivot["rel2"]=df_pivot["rel2"]*-1
                    color_fig7=px.colors.sequential.Greens_r
                else:
                    df_pivot=df_pivot.tail(6)
                    color_fig7=px.colors.sequential.Greens
                
            else:
                if selected_var == "perceptions_of_corruption":
                    df_pivot=df_pivot.tail(6)
                    color_fig7=px.colors.sequential.Reds
                elif selected_var == "healthy_life_expectancy_at_birth":
                    df_pivot=df_pivot.head(2)
                    df_pivot["rel2"]=df_pivot["rel2"]*-1
                    color_fig7=px.colors.sequential.Reds_r
                else:
                    df_pivot=df_pivot.head(6)
                    df_pivot["rel2"]=df_pivot["rel2"]*-1
                    color_fig7=px.colors.sequential.Reds_r

            return(df_pivot, color_fig7, selected_var)
        
        df_pivot, color_fig7, selected_var=fig7(df_new, selected_var, slider_fig2)
    
        if selected_var == "generosity":
            st.warning("Please select another variable")
        else:
            fig = px.scatter(df_pivot,
                            x="2021",
                            y="2021",
                            size="rel2",
                            color="rel",
                            color_continuous_scale=color_fig7,
                            hover_name="country_name",
                            labels=translate2,
                            title="Biggest Movers of selected [Variable] from 2011 to 2021<br><sup>measured by the Differential Ratios"
            )
            for index, row in df_pivot.iterrows():
                fig.add_annotation(x=row["2021"],
                                    y=row["2021"],
                                    text=row["country_name"],
                                    showarrow=True,
                                    arrowwidth=1,
                                    arrowhead=1,
                                    arrowcolor="white",
                                    ax=0,
                                    ay=-50,
                                    font=dict(size=12)
                )
            fig.update_layout(yaxis_title=" ", 
                            xaxis_title=f"{translabel} in 2021",
                            showlegend=False
            )
            fig.update_yaxes(showticklabels=False)
            fig.update_traces(hovertemplate=(
                                "Country: %{customdata[2]}<br>"
                                '2021: %{x}<br>'
                                '2011: %{customdata[0]}<br>'
                                'Abs.: %{customdata[1]}<br>'
                                "Rel.: %{customdata[3]}%<br>"
                                ),
                                customdata=df_pivot[['2011', "abs", "country_name", "rel"]].values
            )
            st.plotly_chart(fig)
            st.write("""It is noticeable that the United States is one of the few countries where life expectancy has declined over 
                     the years, despite the high GDP and happiness score suggesting otherwise. Even Afghanistan, which has been 
                     one of the worst affected in almost all categories since the Taliban takeover, has not seen a decline in 
                     life expectancy.""")
            st.write(" ")

        
        #FIGURE6
        st.write("""""")
        slider=st.select_slider(" ", options=["happiest", "unhappiest"])

        @st.cache_data
        def fig6(df_new, slider):
            df_2021=df_new.loc[df_new["year"] == 2021]
            store_2021=df_2021["country_name"].unique()
            df_fig6=df_new.loc[df_new["country_name"].isin(store_2021)]

            df_fig6=df_fig6.groupby(by="country_name")["life_ladder"].mean()
            df_fig6.sort_values(inplace=True)
                
            if slider == "happiest":
                store=df_fig6.tail(10).index
            else:
                store=df_fig6.head(10).index
            
            return store

        store=fig6(df_new, slider)
        traces = []

        for store_name in store:
            #calculate mean
            store_name_mean = df_new[df_new["country_name"] == store_name]["life_ladder"].mean()

            #value 2021 without missings
            try:
                store_name_2021 = df_new[(df_new["country_name"] == store_name) & (df_new["year"] == 2021)]["life_ladder"].iloc[0].round(2)
            except IndexError:
                store_name_2021 = None

            #filter data for the current store
            store_data = df_new[df_new["country_name"] == store_name]
            store_data["life_ladder"]=store_data["life_ladder"].round(2)

            #create a list of y values (store name) for each data point with offset
            y_values = [store_name] * len(store_data)

            #create a list of text labels for each data point
            hover_text = [f"Country: {store_name}<br>Year: {year}<br>Life Ladder: {life_ladder}" 
                         for year, life_ladder in zip(store_data["year"], store_data["life_ladder"])]

            #create a scatter plot trace for the current store
            trace_historic = go.Scatter(x=store_data["life_ladder"],
                                        y=y_values,
                                        mode='markers',
                                        marker=dict(color="gray", size=10, opacity=0.3),
                                        showlegend=False,
                                        hovertext=hover_text,
                                        hoverinfo="text"
            )

            #sreate a scatter plot trace for the mean value
            trace_mean = go.Scatter(x=[store_name_mean],
                                    y=[store_name],
                                    mode='markers',
                                    marker=dict(color="yellow", size=12, opacity=0.3),
                                    showlegend=False,
                                    hovertext=f"Country: {store_name}<br>Mean Life Ladder: {store_name_mean:.2f}",
                                    hoverinfo="text",
                                    name="mean"
            )

            #create a scatter plot trace for the 2021 value if it exists
            if store_name_2021 is not None:
                trace_2021_color = color1[0] if store_name_mean > store_name_2021 else color1[10]
                trace_2021 = go.Scatter(x=[store_name_2021],
                                        y=[store_name],
                                        mode='markers',
                                        marker=dict(color=trace_2021_color, size=12, line=dict(color='white', width=2)),
                                        showlegend=False,
                                        hovertext=f"Country: {store_name}<br>2021 Life Ladder: {store_name_2021:.2f}",
                                        hoverinfo="text",
                                        name="2021"
                )
                traces.append(trace_2021)

            #append the traces
            traces.append(trace_historic)
            traces.append(trace_mean)

        #create legend traces
        legend_traces = [
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color='yellow', size=12, opacity=0.5),
                showlegend=True, name='Mean Life Ladder'
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color='red', size=12, line=dict(color='white', width=2)),
                showlegend=True, name='2021 Life Ladder (Decrease)'
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color='green', size=12, line=dict(color='white', width=2)),
                showlegend=True, name='2021 Life Ladder (Increase)'
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(color='gray', size=10, opacity=0.3),
                showlegend=True, name='Historical Life Ladder'
            )
        ]

        #add the legend traces to the plot
        traces.extend(legend_traces)

        #create the figure
        fig = go.Figure(data=traces)
        fig.update_layout(title="Trend of Happiness for various Countries")
        st.plotly_chart(fig)
        st.write("""It appears that people are becoming unhappier compared to previous years. The latest measured happiness scores in 
                 many countries are lower than the historical mean scores. Could this be a result of COVID-19? Investigating this 
                 question using data from 2022 and now would be interesting.""")


    with fol3:
        #FIGURE8
        @st.cache_data
        def fig8(df_new):
            rename_dict={"life_ladder": "Happiness Score", 
                            "log_gdp_per_capita": "GDP",
                            "social_support": "Social Support", 
                            "healthy_life_expectancy_at_birth": "Life Expectancy", 
                            "freedom_to_make_life_choices": "Freedom", 
                            "generosity": "Generosity",
                            "perceptions_of_corruption": "Corruption"}
            num_col2=["Happiness Score", "GDP","Social Support", "Life Expectancy", "Freedom", "Generosity", "Corruption"]
            df_corr=df_new.rename(columns=rename_dict)
            correlation_matrix = df_corr[num_col2].corr().round(2)
            return correlation_matrix
        
        correlation_matrix=fig8(df_new)
        fig = px.imshow(correlation_matrix, 
                        color_continuous_scale=color1,
                        title="Correlation Matrix",
                        color_continuous_midpoint=0
        )
        annotations = []
        for i, row in enumerate(correlation_matrix.values):
            for j, val in enumerate(row):
                font_color = 'white' if abs(val) > 0.75 and abs(val) > -0.25 else 'black'
                annotations.append(dict(text=f"{val:.2f}", x=j, y=i, xref='x', yref='y', showarrow=False, font=dict(color=font_color)))

        fig.update_layout(annotations=annotations, width=650, height=650)
        fig.update_traces(hovertemplate="Variable 1: %{x}<br>Variable 2: %{y}<br>Correlation: %{z:.2f}")
        st.plotly_chart(fig)
        st.write("""The heatmap reveals that the features with the biggest impact on the happiness score are GDP and life expectancy - both showing a positive correlation.""")
        st.write(" ")


        #FIGURE9
        st.write(" ")
        col1, col2=st.columns(2)
        with col1:
            selected_x = st.selectbox("Select X:", variables)
        with col2:
            selected_y = st.selectbox("Select Y:", variables)
        
        selected_x = translate.get(selected_x)
        selected_y = translate.get(selected_y)
        
        @st.cache_data
        def fig9(df_new, selected_x, selected_y):
            df_new1 = df_new.groupby(by=["year", "regional_indicator"], as_index=False).agg({
                selected_x: "mean",
                selected_y: "mean"
                }).round(2)
            df_new1 = df_new1.sort_values(by="regional_indicator", ascending=True)
            return (df_new1, selected_x, selected_y)

        df_new1, selected_x, selected_y=fig9(df_new, selected_x, selected_y)
        fig = px.scatter(df_new1, 
                         x=selected_x, 
                         y=selected_y, 
                         color="regional_indicator", 
                         title="Relationship<br><sup>between selected [Variables] across Regions by Year",
                         color_discrete_map=color, 
                         hover_data={"year": True},
                         labels=translate2
        )
        st.plotly_chart(fig)
        st.write("""When examining the relationships between the two features with the greatest impact on happiness and the happiness score, we can identify three distinct clusters. 
                 For better visualization, I manually set the data into these three clusters by comparing the attributes on the next plot.""")


        #FIGURE9a
        @st.cache_data
        def fig9a(df_new, selected_x, selected_y):
            df_new1=df_new
            df_new1["life_ladder_cat"]=0

            def new_feature(x):
                if x < 5:
                    return "low"
                elif x >= 5 and x < 6:
                    return "middle"
                else:
                    return "top"

            df_new1.life_ladder_cat=df_new1.life_ladder.apply(new_feature)

            a="country_name"
            b="year"

            return (df_new1, selected_x, selected_y, a, b)
        
        df_new1, selected_x, selected_y, a, b=fig9a(df_new, selected_x, selected_y)

        top="rgb(0,109,44)"
        middle="rgb(247,254,174)"
        low="rgb(251,106,74)"
        color_fig9a={"top": top, "middle": middle, "low": low}

        fig = px.scatter(df_new1, 
                         x=selected_x, 
                         y=selected_y, 
                         color="life_ladder_cat", 
                         title="Relationship<br><sup>between selected [Variables] by colored clusters",
                         color_discrete_map=color_fig9a,
                         hover_data={a: True, selected_x: ':.2f', selected_y: ':.2f', b: True},
                         labels=translate2
        )
        st.plotly_chart(fig)
        st.write("""Comparing the variables life expectancy and GDP, we can observe a high 
                 positive correlation. This is evident from the multiple lines that depict how 
                 individual countries experience simultaneous increases in both variables.""")


if page == pages[3]:
    fol4, fol5 = st.tabs(["Pre-Work", "Model Results"])

    with fol4:
        st.subheader("Preprocessing Steps")
        st.write("""To ensure the robustness of our ML model, I avoided reducing the sample size, as small training 
        sets may not capture all data variations and small test sets might not evaluate all scenarios adequately. I removed 
        unnecessary variables like "Generosity". For filling missing values in the numerical subset, I used the KNN Imputer, 
        which imputes values based on similar samples, preserving underlying data relationships. To standardize feature 
        scales, I applied the Standard Scaler, preventing features with larger magnitudes from dominating the learning 
        process and ensuring proper Hyper Parameter reactions.""")
        st.write(" ")
        
        st.subheader("Linear Regression vs. Random Forest Regressor")
        st.write("""Both models have their advantages, but in this case I believe the benefits of RFR outweigh those of LR. RFR 
                 capture non-linear relationships more effectively and are more robust to outliers, which I didn’t eliminate. 
                 However, they act as a black box, lacking interpretable coefficients. In contrast, LR provides clear coefficients 
                 for interpretation and performs well on training and test sets. While the Random Forest model is slightly overfitted, 
                 it shows better results on the MAE. Here are the results for both models.""")

        st.write(" ")
        with st.expander("Results of LR", expanded=False):
            ln_results=pd.DataFrame({"Version": ["By default", "PCA"],
                                     "MSE": [0.28, 0.34],
                                     "RSME": [0.53,0.58],
                                     "MAE": [0.41,0.46],
                                     "R2 Train": [0.78,0.74],
                                     "R2 Test": [0.77,0.72]}, index=[1,2])
            st.dataframe(ln_results)
        with st.expander("Results of RFR", expanded=False):
            rfr_results=rfr_results.reset_index(drop=True)
            rfr_results.index=rfr_results.index=[1,2,3,4]
            rfr_results = rfr_results.drop(index=[2, 4])
            rfr_results=rfr_results.reset_index(drop=True)
            rfr_results.index=rfr_results.index=[1,2]
            rfr_results=rfr_results.rename(columns={"Unnamed: 0": "Version"})
            st.dataframe(rfr_results)


    with fol5:
        #FIGURE11
        color_scale2=[
            [0, "rgb(251,106,74)"],
            [0.5, "rgb(247,254,174)"],
            [1, "rgb(0,109,44)"]
        ]
                
        st.write("""Below are the renderings about the results of the selected model, Random Forest Regressor with the best parameters. """)
        fig11=feat_imp_rfr.iloc[:5]
        fig11=fig11.replace(translate2)
        fig11=fig11.sort_values(by=fig11.columns[1], ascending=True).round(2)
        fig = px.bar(fig11, 
                     y=fig11.columns[0], 
                     x=fig11.columns[1], 
                     labels=translate2, 
                     title="Bar Chart of Feature Importance",
                     color=fig11.columns[1],
                     color_continuous_scale=color_scale2,
                     color_continuous_midpoint=0
                     )
        fig.update_layout(yaxis_title=" ", coloraxis_showscale=False)
        st.plotly_chart(fig)
        st.write("""Here are the most important feature of the model results, which I already identified in my EDA. 
                 GDP has the biggest impact to the happiness score.""")
        st.write(" ")
        
        #FIGURE12
        pred_results['normalized_error'] = (pred_results['residuals'] - pred_results['residuals'].min()) / (-pred_results['residuals'].min() - pred_results['residuals'].min())
        pred_results["residuals"]=pred_results["residuals"].round(2)
                
        color_scale=[
            [0, "rgb(251,106,74)"],
            [0.4, "rgb(247,254,174)"],
            [0.5, "rgb(0,109,44)"],
            [0.6, "rgb(247,254,174)"],
            [1, "rgb(251,106,74)"]
        ]

        fig1 = px.scatter(pred_results, 
                        y=round(pred_results['pred'], 2), 
                        x=round(pred_results['life_ladder'], 2), 
                        labels={"life_ladder": "True Value", "pred": "Predicted Value", "color": "Error", "x": "True Value",
                               "y": "Predicted Value", "country_name": "Country", "year": "Year",
                               "residuals": "Error", "normalized_error": "Normalized Error"},
                        title="Results of the Test Set",
                        color=pred_results["normalized_error"],
                        color_continuous_scale=color_scale,
                        color_continuous_midpoint=0.5,                      
                        hover_data={"country_name": True, "year": True, "residuals": True, "normalized_error": False}
                        )
        
        #add corridor of MAE
        true_values = pred_results['life_ladder']
        predicted_values = pred_results['pred']
        mae = 0.33

        upper_bound = true_values + mae
        lower_bound = true_values - mae

        #sort values for proper plotting
        sorted_indices = true_values.argsort()
        true_values_sorted = true_values.iloc[sorted_indices].round(2)
        upper_bound_sorted = upper_bound.iloc[sorted_indices].round(2)
        lower_bound_sorted = lower_bound.iloc[sorted_indices].round(2)

        #add filled area for the MAE corridor
        fig1.add_trace(go.Scatter(
            x=true_values_sorted,
            y=upper_bound_sorted,
            mode='lines',
            line=dict(color='white', dash="dash"),
            showlegend=True,
            name='MAE Bound'
        ))

        fig1.add_trace(go.Scatter(
            x=true_values_sorted,
            y=lower_bound_sorted,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(211, 211, 211, 0.5)',
            line=dict(color='white', dash="dash"),
            showlegend=False,
            name='MAE Bound'
        ))

        #add perfect prediction line
        min_value = round(min(true_values_sorted.min(), predicted_values.min()),2)
        max_value = round(max(true_values_sorted.max(), predicted_values.max()),2)

        fig1.add_trace(go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode='lines',
            line=dict(color='white'),
            showlegend=True,
            name='Perfect Prediction'
        ))
        fig1.update_layout(title="Visualizing Model Accuracy: Predicted vs. True Values",
                        yaxis_title="Predicted Value",
                        xaxis_title="True Value",
                        legend=dict(
                                yanchor="top",
                                y=1.1,
                                xanchor="center",
                                x=0.5))
        st.plotly_chart(fig1)
        st.write("""In the previous tab, RFR achieved a Mean Absolute Error of 0.33, indicating that, 
                 on average, the predictions are 0.33 points away from the true Happiness Score. The red points represent 
                 results with larger distance from the ideal line, while the green points indicate closer approximations. I can also
                 conclude, that countries with a small actual score were predicted to have a higher score, while countries with a higher actual score
                 were predicted to have a lower score.""")
        st.write(" ")
        
        
        #FIGURE 13
        @st.cache_data
        def fig13(pred_results):
            pred_results=pred_results.groupby(by="country_name")["residuals"].mean().round(2)
            pred_results=pd.DataFrame(pred_results).reset_index()
            max_abs_residual = max(abs(pred_results['residuals'].min()), abs(pred_results['residuals'].max()))
            return (pred_results, max_abs_residual)

        df_fig2, max_abs_residual=fig13(pred_results)
        df_fig2['normalized_error'] = (df_fig2['residuals'] - df_fig2['residuals'].min()) / (-df_fig2['residuals'].min() - df_fig2['residuals'].min())
        df_fig2["residuals"]=df_fig2["residuals"].round(2)

        fig = px.choropleth(df_fig2,
                            locations="country_name",
                            locationmode="country names",
                            color=df_fig2["normalized_error"],
                            color_continuous_scale=color_scale,
                            color_continuous_midpoint=0.5,
                            title="Mean Absolute Error across Countries",
                            labels={"country_name": "Country", "residuals": "MAE", "normalized_error": "Normalized MAE"},
                            hover_data={"residuals": True, "normalized_error": False}
        )
        st.plotly_chart(fig)
        st.write("""This map displays the MAE for each country in the test set. The bad predictions are concentrated in a 
                 few countries such as Afghanistan, Pakistan, and South Sudan. These countries are also among the most significant 
                 movers in the dataset. This observation suggests that the model is more sensitive to countries experiencing 
                 substantial changes, such as shifts in government, civil wars, or natural disasters.""")


if page == pages[4]:
    st.subheader("Conclusion")
    text_final="""
    Based on the data, the happiest people predominantly reside in Scandinavia (Western Europe), while African nations generally score 
    lower on well-being. Additionally, happiness appears to be declining in various countries, potentially as a result of COVID-19. 
    My analyses have also identified GDP and life expectancy as the most significant factors influencing the happiness score, 
    a finding corroborated by our machine learning model.

    To improve happiness, governments should focus on implement policies that promote sustainable 
    economic growth, job creation and fair distribution of wealth. Investing in infrastructure, education 
    and technology can drive economic development and increase GDP. Strengthening healthcare 
    systems to improve life expectancy is also crucial. This includes ensuring access to quality healthcare, 
    promoting preventive care and addressing public health issues. Investments in healthcare 
    infrastructure and medical research can lead to better health outcomes and increased life expectancy. """
    st.write(text_final)

    st.subheader("Criticism")
    st.write("""The report heavily relies on the Cantril Ladder, where respondents rate their life 
             satisfaction on a scale from 0 to 10. Critics argue that this single-question approach 
             oversimplifies the complex concept of happiness and may not capture all its dimensions
             [click here to visit Africa Check](https://africacheck.org/fact-checks/blog/analysis-what-does-un-happiness-report-really-measure).
             \nStudies have also shown that the Cantril Ladder might prompt respondents to think about their 
             socio-economic status, thus skewing the results towards materialistic measures of 
             happiness rather than holistic well-being
             [click here to visit Phys.org](https://phys.org/news/2024-03-world-happiness-wrong.html).""")