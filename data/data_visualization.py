import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go

def plot_kde(attrition):
    f, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=False, sharey=False)
    
    # KDE Plots
    s = np.linspace(0, 3, 10)
    cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)
    sns.kdeplot(data=attrition, x='Age', y='TotalWorkingYears', cmap=cmap, fill=True, cut=5, ax=axes[0,0])
    axes[0,0].set(title='Age against Total working years')

    cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)
    sns.kdeplot(data=attrition, x='Age', y='DailyRate', cmap=cmap, fill=True, ax=axes[0,1])
    axes[0,1].set(title='Age against Daily Rate')

    cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)
    sns.kdeplot(data=attrition, x='YearsInCurrentRole', y='Age', cmap=cmap, fill=True, ax=axes[0,2])
    axes[0,2].set(title='Years in role against Age')

    cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)
    sns.kdeplot(data=attrition, x='DailyRate', y='DistanceFromHome', cmap=cmap, fill=True, ax=axes[1,0])
    axes[1,0].set(title='Daily Rate against Distance from Home')

    cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)
    sns.kdeplot(data=attrition, x='DailyRate', y='JobSatisfaction', cmap=cmap, fill=True, ax=axes[1,1])
    axes[1,1].set(title='Daily Rate against Job Satisfaction')

    cmap = sns.cubehelix_palette(start=1.666666666667, light=1, as_cmap=True)
    sns.kdeplot(data=attrition, x='YearsAtCompany', y='JobSatisfaction', cmap=cmap, fill=True, ax=axes[1,2])
    axes[1,2].set(title='Years at Company against Job Satisfaction')

    cmap = sns.cubehelix_palette(start=2.0, light=1, as_cmap=True)
    sns.kdeplot(data=attrition, x='YearsAtCompany', y='DailyRate', cmap=cmap, fill=True, ax=axes[2,0])
    axes[2,0].set(title='Years at Company against Daily Rate')

    cmap = sns.cubehelix_palette(start=2.333333333333, light=1, as_cmap=True)
    sns.kdeplot(data=attrition, x='RelationshipSatisfaction', y='YearsWithCurrManager', cmap=cmap, fill=True, ax=axes[2,1])
    axes[2,1].set(title='Relationship Satisfaction vs Years with Manager')

    cmap = sns.cubehelix_palette(start=2.666666666667, light=1, as_cmap=True)
    sns.kdeplot(data=attrition, x='WorkLifeBalance', y='JobSatisfaction', cmap=cmap, fill=True, ax=axes[2,2])
    axes[2,2].set(title='Work Life Balance against Job Satisfaction')

    f.tight_layout()
    plt.show()

def plot_correlation_heatmap(attrition):
    data = [
        go.Heatmap(
            z=attrition.corr().values,  # Generating the Pearson correlation
            x=attrition.columns.values,
            y=attrition.columns.values,
            colorscale='Viridis',
            reversescale=False,
            opacity=1.0
        )
    ]

    layout = go.Layout(
        title='Pearson Correlation of numerical features',
        xaxis=dict(ticks='', nticks=36),
        yaxis=dict(ticks=''),
        width=900, height=700,
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='labelled-heatmap')
