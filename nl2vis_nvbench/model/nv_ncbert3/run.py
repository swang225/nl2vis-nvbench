from nebula.model.nv_ncbert3 import nvncBert3
from nebula.data.nvbench.vega import to_vega_lite

path1 = "C:/Users/aphri/Documents/t0002/pycharm/repo/nebula/model/ncnet_no_af2/result/model_best.pt"
# path1 = "C:/Users/aphri/Documents/t0002/pycharm/data/ncnet/output_models/model_best.pt"
temp_path1 = "C:/Users/aphri/Documents/t0002/pycharm/data/ncnet/temp_data"
m1 = nvncBert3(
    trained_model_path=path1,
    temp_dataset_path=temp_path1
)

path2 = "C:/Users/aphri/Documents/t0002/pycharm/python/data/nvBench-main/databases/database/car_1/car_1.sqlite"
m1.specify_dataset(
    data_type='sqlite3',
    db_url=path2,
    table_name='cars_data'
)

m1.show_dataset(top_rows=3)

res = m1.predict(
    nl_question="What is the average weight and year for each year. Plot them as line chart.",
    show_progress=False
)

res_vis = to_vega_lite(res)

print(res)