from nl2vis_nvbench.model.nv_bert import nvBert
import os.path as osp
from nl2vis_nvbench import root

temp_path1 = "C:/Users/aphri/Documents/t0002/pycharm/data/ncnet/temp_data"
m1 = nvBert(temp_dataset_path=temp_path1)

best_model_path = osp.join(root(), "model/nv_bert/result/model_best.pt")
m1.load_model(best_model_path)

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

print(res)