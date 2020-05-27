from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, FileInput, Button, \
    RadioGroup, Div, CheckboxButtonGroup, Select, TableColumn, DataTable
from bokeh.plotting import figure

from metafeatures import Meta
from autoclustering import AutoClustering

# Set up plot
data = dict(
        metafeature=[],
        value=[],
    )
source = ColumnDataSource(data)

columns = [
        TableColumn(field="metafeature", title="MetaFeature"),
        TableColumn(field="value", title="value"),
    ]
data_table = DataTable(source=source, columns=columns, width=200, height=280)


# Set up widgets
header_1 = Div(text="<h2> <b> Automated Clustering <b> </h2>")
select = Select(title="Sample Data:", value="Glass", options=["Glass", "Iris", "Pathbased", "Spherical_6_2"])
subheader_1 = Div(text="<h3> Select Configurations </h3>")
gen = Slider(title="Generations", value=10, start=-10, end=50, step=5)
pop = Slider(title="Population", value=10, start=-10, end=100, step=5)
button = Button(label="Search", button_type="success")
subheader_2 = Div(text="<h3> Select Meta-Models </h3>")
radio_group = RadioGroup(
        labels=["Meta-learning for Evaluation Metrics & Algorithm", "Meta-Learning for Initial Configuration"], active=0)
subheader_3 = Div(text="<h3> Other Preferences </h3>")
checkbox_button_group = CheckboxButtonGroup(
        labels=["Show all non-dominated configurations"], active=[0])


file_input = FileInput()

# Set up callbacks

def display():
    file = "./datasets/{}.csv".format(select.value.lower())
    meta = Meta(file)
    df = meta.extract_metafeatures(file, "distance")


    columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns]
    data_table = DataTable(columns=columns, source=ColumnDataSource(df), width=1200, height=50)
    subheader_4 = Div(text="<br> <h4> Metafeatures </h4>")

    # Search configurations
    _gen = int(gen.value)
    _pop = int(pop.value)

    meta = radio_group.active

    if meta == 0:
        auto = AutoClustering(file, generations=_gen, population=_pop, pre_cvi=True)
    else:
        auto = AutoClustering(file, generations=_gen, population=_pop, pre_config=True)

    # Get configuration information
    algorithm, evaluation = auto.get_parameters()
    data = dict(
        parameters=["Algorithm", "Evaluation Metrics"],
        values=[algorithm, evaluation],
    )

    source = ColumnDataSource(data)

    columns = [
        TableColumn(field="parameters", title="Parameters"),
        TableColumn(field="values", title="Values"),
    ]

    subheader_5 = Div(text="<h4> Configuration </h4>")
    config_table = DataTable(source=source, columns=columns, width=800, height=80)

    # Display dataset characteristics
    # curdoc().clear()
    # curdoc().add_root(row(inputs, column(subheader_4, data_table, subheader_5, config_table), width=1200))

    pops, hof = auto.search()

    # Display best HOF
    if len(checkbox_button_group.active) == 0:
        hof = hof[0]

    str_pops = [str(pop) for pop in hof]

    # Display results
    data = dict(
        configurations=str_pops,
    )

    source = ColumnDataSource(data)

    columns = [
        TableColumn(field="configurations", title="Configurations"),
    ]

    subheader_6 = Div(text="<h4> Results... </h4>")
    results_table = DataTable(source=source, columns=columns, width=1200)
    # print(pops)

    curdoc().clear()
    curdoc().add_root(row(inputs, column(subheader_4, data_table, subheader_5, config_table, subheader_6, results_table), width=1200))


# Button callback
button.on_click(display)



# Set up layouts and add to document
inputs = column(header_1, select, subheader_1, gen, pop, subheader_2, radio_group, subheader_3, checkbox_button_group, button)


mainLayout = column(row(inputs, width=1200), name='mainLayout')
curdoc().add_root(mainLayout)
# curdoc().add_root(row(inputs, width=1200))
curdoc().title = "AutoClustering"
